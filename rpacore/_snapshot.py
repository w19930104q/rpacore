"""Offline consistency snapshots for paired transaction and queue databases.

The framework keeps transaction truth and queue truth in two separate SQLite
files. Copying them independently cannot promise one logical point in time: a
retry or lease renewal committing between the two copies leaves the pair
inconsistent. For each source this module opens two cooperating connections:
a writable sentry parked in ``BEGIN IMMEDIATE`` whose RESERVED lock blocks any
other connection from starting a write transaction for the whole window, and a
separate read-only connection whose explicit read transaction is copied with
the SQLite online backup mechanism. With a sentry held on both files before
either copy starts, no writer commit can land on either database while the
pair is assembled, in both rollback-journal and WAL modes. The sentry never
writes a row; releasing it rolls the empty transaction back, so the source
databases are not modified.

Publication is all-or-nothing: copies, the versioned manifest and the
completion marker are assembled in a temporary sibling directory and only
renamed into place after every checksum and integrity check succeeds.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import sqlite3
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from threading import Event
from typing import TypedDict

from importlib import metadata

from rpacore._snapshot_errors import (
    SnapshotBusyError,
    SnapshotCancelledError,
    SnapshotError,
    SnapshotVerificationError,
)
from rpacore._sqlite import (
    QUEUE_SCHEMA_VERSION,
    SCHEMA_VERSION_TABLE,
    TRANSACTION_SCHEMA_VERSION,
    connect_sqlite,
    ensure_database_compatibility,
    require_current_schema,
    validate_durable_sqlite_path,
)

SNAPSHOT_FORMAT_VERSION = 1
TRANSACTION_SNAPSHOT_NAME = "transactions.db"
QUEUE_SNAPSHOT_NAME = "queue.db"
MANIFEST_NAME = "snapshot-manifest.json"
COMPLETION_MARKER_NAME = "snapshot.complete"
_BACKUP_PAGES = 256
_BACKUP_BUSY_RETRIES = 3
_BACKUP_RETRY_SLEEP_SECONDS = 0.05
_SQLITE_RESULT_OK = 0
_SQLITE_RESULT_BUSY = 5
_SQLITE_RESULT_LOCKED = 6
_SHA256_CHUNK_SIZE = 1024 * 1024
_TRANSACTION_COMPONENT = "transactions"
_QUEUE_COMPONENT = "queue"
_SQLITE_MASTER_TABLES = ("sqlite_sequence",)


class _LeaseManifest(TypedDict):
    in_progress_items: int
    complete_claims: int
    incomplete_claims: int
    open_attempts: int


class _DatabaseManifest(TypedDict, total=False):
    component: str
    source_path: str
    snapshot_file: str
    schema_version: int
    journal_mode: str
    sha256: str
    size_bytes: int
    row_counts: dict[str, int]
    foreign_key_violations: int
    integrity_check: str
    lease: _LeaseManifest


class _SnapshotManifest(TypedDict):
    format_version: int
    created_at: str
    framework_version: str
    completion_marker: str
    databases: list[_DatabaseManifest]


def _framework_version() -> str:
    try:
        return metadata.version("rpacore")
    except metadata.PackageNotFoundError:
        return "0.0.0+local"


@dataclass(frozen=True)
class SnapshotDatabaseResult:
    """One verified database copy inside a published snapshot."""

    component: str
    source_path: str
    snapshot_file: str
    schema_version: int
    journal_mode: str
    sha256: str
    size_bytes: int
    row_counts: dict[str, int]
    foreign_key_violations: int
    integrity_check: str


@dataclass(frozen=True)
class SnapshotResult:
    """The published location and per-database verification facts."""

    snapshot_dir: str
    format_version: int
    created_at: str
    databases: tuple[SnapshotDatabaseResult, ...]


@dataclass(frozen=True)
class RestoredDatabase:
    """One verified database written to an explicit new restore path."""

    component: str
    path: str
    schema_version: int
    journal_mode: str
    sha256: str
    size_bytes: int
    row_counts: dict[str, int]
    foreign_key_violations: int
    integrity_check: str
    lease: "QueueLeaseState | None"


@dataclass(frozen=True)
class QueueLeaseState:
    """Queue claim facts captured alongside the snapshot for audit."""

    in_progress_items: int
    complete_claims: int
    incomplete_claims: int
    open_attempts: int


@dataclass(frozen=True)
class SnapshotRestoreResult:
    """The new restore paths and their verification facts."""

    snapshot_dir: str
    restored_at: str
    databases: tuple[RestoredDatabase, ...]


def create_consistency_snapshot(
    *,
    transaction_db_path: str,
    queue_db_path: str,
    output_dir: str,
    cancel_event: Event | None = None,
) -> SnapshotResult:
    """Back up the transaction and queue databases as one consistent pair.

    A sentry write lock and a pinned read transaction are acquired on both
    sources before either copy starts, so no writer commit can land on either
    database while the pair is assembled. The output directory must not already
    exist; all work happens in a temporary sibling directory and is atomically
    renamed into place only after both copies pass integrity, foreign-key,
    schema and checksum verification. Any failure removes the staging
    directory without touching existing files.
    """
    transaction_source = _validate_source_path(
        transaction_db_path, field="transaction_db_path"
    )
    queue_source = _validate_source_path(queue_db_path, field="queue_db_path")
    destination = _validate_output_dir(output_dir)
    if transaction_source == queue_source:
        raise SnapshotError(
            "transaction and queue database paths must name different files; "
            "paired consistency snapshots require the separate-file layout"
        )
    _raise_if_cancelled(cancel_event)

    staging_path = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent)
    )
    guards: list["_DatabaseGuard"] = []
    try:
        # Each source gets two cooperating connections: a writable sentry
        # holding BEGIN IMMEDIATE (a RESERVED lock that blocks every other
        # connection from acquiring a write lock for the whole window) and a
        # separate read-only connection whose explicit read transaction is the
        # snapshot the online backup copies. The two sentries together close
        # the cross-file window in both rollback-journal and WAL modes: no
        # retry or lease renewal can commit on either database while the pair
        # is assembled.
        transaction_guard = _open_database_guard(
            transaction_source, component=_TRANSACTION_COMPONENT
        )
        guards.append(transaction_guard)
        _raise_if_cancelled(cancel_event)
        queue_guard = _open_database_guard(
            queue_source, component=_QUEUE_COMPONENT
        )
        guards.append(queue_guard)
        _raise_if_cancelled(cancel_event)
        transaction_snapshot, transaction_lease = _backup_database(
            reader=transaction_guard.reader,
            source_path=transaction_source,
            staging_path=staging_path / TRANSACTION_SNAPSHOT_NAME,
            component=_TRANSACTION_COMPONENT,
            cancel_event=cancel_event,
        )
        queue_snapshot, queue_lease = _backup_database(
            reader=queue_guard.reader,
            source_path=queue_source,
            staging_path=staging_path / QUEUE_SNAPSHOT_NAME,
            component=_QUEUE_COMPONENT,
            cancel_event=cancel_event,
        )
        created_at = datetime.now(timezone.utc).isoformat()
        manifest = _build_manifest(
            transaction_snapshot=transaction_snapshot,
            queue_snapshot=queue_snapshot,
            transaction_lease=transaction_lease,
            queue_lease=queue_lease,
            created_at=created_at,
        )
        _raise_if_cancelled(cancel_event)
        _publish_snapshot(staging_path, destination, manifest)
        return SnapshotResult(
            snapshot_dir=str(destination),
            format_version=SNAPSHOT_FORMAT_VERSION,
            created_at=created_at,
            databases=(transaction_snapshot, queue_snapshot),
        )
    except SnapshotError:
        _remove_staging(staging_path)
        raise
    except sqlite3.OperationalError as exc:
        _remove_staging(staging_path)
        if _is_busy(exc):
            raise SnapshotBusyError(
                "Consistency snapshot could not acquire a stable read of both "
                f"databases: {exc}"
            ) from exc
        raise SnapshotError(f"Consistency snapshot failed: {exc}") from exc
    except Exception:
        _remove_staging(staging_path)
        raise
    finally:
        for guard in reversed(guards):
            guard.close()


def restore_consistency_snapshot(
    *,
    snapshot_dir: str,
    transaction_target_path: str,
    queue_target_path: str,
) -> SnapshotRestoreResult:
    """Restore a published snapshot to explicit new database paths.

    Neither target path may already exist, so an original live database can
    never be overwritten. The snapshot directory is only read; its completion
    marker, manifest format and recorded SHA-256 digests are verified before
    any target file is written. Each restored copy is then reopened to verify
    schema markers, table row counts, foreign-key integrity and queue lease
    facts against the manifest. A failure removes only files this call
    created.
    """
    snapshot_path = _validate_existing_snapshot(snapshot_dir)
    transaction_target = _validate_restore_target(
        transaction_target_path, field="transaction_target_path"
    )
    queue_target = _validate_restore_target(
        queue_target_path, field="queue_target_path"
    )
    if transaction_target == queue_target:
        raise SnapshotError(
            "restore target paths must name different files"
        )
    manifest = _load_manifest(snapshot_path)
    _verify_snapshot_files(snapshot_path, manifest)

    created: list[Path] = []
    try:
        restored: list[RestoredDatabase] = []
        for database in manifest["databases"]:
            component = str(database["component"])
            target = (
                transaction_target
                if component == _TRANSACTION_COMPONENT
                else queue_target
            )
            source_file = snapshot_path / str(database["snapshot_file"])
            _copy_with_fsync(source_file, target)
            created.append(target)
            restored.append(
                _verify_restored_database(target, component, database)
            )
        return SnapshotRestoreResult(
            snapshot_dir=str(snapshot_path),
            restored_at=datetime.now(timezone.utc).isoformat(),
            databases=tuple(restored),
        )
    except SnapshotError:
        _cleanup_created(created)
        raise
    except sqlite3.Error as exc:
        _cleanup_created(created)
        raise SnapshotVerificationError(
            f"Restored database failed SQLite verification: {exc}"
        ) from exc
    except Exception:
        _cleanup_created(created)
        raise


def _validate_source_path(value: str, *, field: str) -> Path:
    validate_durable_sqlite_path(value, field=field)
    resolved = Path(value).resolve()
    if not resolved.is_file():
        raise SnapshotError(f"Source database does not exist: {resolved}")
    if not _sqlite_header(resolved):
        raise SnapshotError(f"Source database is not a SQLite file: {resolved}")
    return resolved


def _validate_output_dir(value: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise SnapshotError("output_dir must be a non-empty path")
    destination = Path(value).resolve()
    if destination.exists():
        raise SnapshotError(
            f"Snapshot output already exists; refusing to overwrite: {destination}"
        )
    if not destination.parent.is_dir():
        raise SnapshotError(
            f"Snapshot output parent directory does not exist: {destination.parent}"
        )
    return destination


def _validate_existing_snapshot(value: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise SnapshotError("snapshot_dir must be a non-empty path")
    snapshot_path = Path(value).resolve()
    if not snapshot_path.is_dir():
        raise SnapshotError(f"Snapshot directory does not exist: {snapshot_path}")
    if not (snapshot_path / COMPLETION_MARKER_NAME).is_file():
        raise SnapshotVerificationError(
            f"Snapshot is incomplete; missing {COMPLETION_MARKER_NAME}: {snapshot_path}"
        )
    return snapshot_path


def _validate_restore_target(value: str, *, field: str) -> Path:
    validate_durable_sqlite_path(value, field=field)
    target = Path(value).resolve()
    if target.exists():
        raise SnapshotError(
            f"Restore target already exists; refusing to overwrite: {target}"
        )
    if not target.parent.is_dir():
        raise SnapshotError(
            f"Restore target parent directory does not exist: {target.parent}"
        )
    return target


def _sqlite_header(path: Path) -> bool:
    with path.open("rb") as handle:
        return handle.read(16) == b"SQLite format 3\x00"


class _DatabaseGuard:
    """The two connections that freeze one source database for the window.

    ``sentry`` is a writable connection parked inside ``BEGIN IMMEDIATE``. Its
    RESERVED lock prevents any other connection from starting a write
    transaction for the lifetime of the guard, in both rollback-journal and
    WAL modes. It never writes a row; closing it rolls the empty transaction
    back and releases the lock.

    ``reader`` is the separate read-only connection whose pinned read
    transaction the online backup copies. The backup API cannot run on the
    sentry connection itself because holding the RESERVED lock blocks its own
    page-copy steps, so the two roles must use distinct connections.
    """

    def __init__(
        self, sentry: sqlite3.Connection, reader: sqlite3.Connection
    ) -> None:
        self.sentry = sentry
        self.reader = reader

    def close(self) -> None:
        # Release the read snapshot first, then the write lock.
        self.reader.close()
        self.sentry.rollback()
        self.sentry.close()


def _open_database_guard(path: Path, *, component: str) -> _DatabaseGuard:
    """Acquire the sentry write lock and pin the backup read transaction."""
    try:
        sentry = connect_sqlite(str(path), field=f"snapshot.{component}")
        try:
            sentry.execute("PRAGMA busy_timeout = 0")
            # BEGIN IMMEDIATE takes the RESERVED lock at once; a busy result
            # here means another connection already owns the write side and
            # the snapshot must fail rather than wait indefinitely.
            sentry.execute("BEGIN IMMEDIATE")
            sentry.execute("SELECT 1").fetchone()
        except sqlite3.OperationalError:
            sentry.close()
            raise
        try:
            reader = _open_pinned_reader(path, component=component)
        except BaseException:
            sentry.rollback()
            sentry.close()
            raise
        return _DatabaseGuard(sentry=sentry, reader=reader)
    except sqlite3.OperationalError as exc:
        if _is_busy(exc):
            raise SnapshotBusyError(
                f"Source database is locked and cannot be snapshotted: {path} ({exc})"
            ) from exc
        raise


def _open_pinned_reader(path: Path, component: str) -> sqlite3.Connection:
    """Open a read-only connection and pin an explicit read transaction."""
    connection = connect_sqlite(
        str(path), field=f"snapshot.{component}", readonly=True
    )
    ensure_database_compatibility(connection)
    supported = (
        TRANSACTION_SCHEMA_VERSION
        if component == _TRANSACTION_COMPONENT
        else QUEUE_SCHEMA_VERSION
    )
    require_current_schema(
        connection,
        component=component,
        supported_version=supported,
        label=component,
    )
    connection.execute("BEGIN")
    # A first read establishes the snapshot before the connection is used for
    # backup; later statements stay inside the same read transaction.
    connection.execute("SELECT 1").fetchone()
    return connection


def _backup_database(
    *,
    reader: sqlite3.Connection,
    source_path: Path,
    staging_path: Path,
    component: str,
    cancel_event: Event | None,
) -> tuple[SnapshotDatabaseResult, QueueLeaseState | None]:
    target_connection = sqlite3.connect(str(staging_path))
    busy_retries = 0
    try:
        def progress(rc: int, _remaining: int, _page_count: int) -> None:
            # CPython invokes the backup progress handler as
            # ``progress(rc, remaining, page_count)``; ``rc`` is the raw
            # SQLite result code of the page-copy step. SQLITE_BUSY and
            # SQLITE_LOCKED mean a concurrent writer blocked this step and the
            # backup API would otherwise keep retrying indefinitely.
            nonlocal busy_retries
            if rc in (_SQLITE_RESULT_BUSY, _SQLITE_RESULT_LOCKED):
                busy_retries += 1
                if busy_retries > _BACKUP_BUSY_RETRIES:
                    raise SnapshotBusyError(
                        f"Backup of {component} database repeatedly met a "
                        "conflicting lock while copying pages"
                    )
            _raise_if_cancelled(cancel_event)

        reader.backup(
            target_connection,
            pages=_BACKUP_PAGES,
            sleep=_BACKUP_RETRY_SLEEP_SECONDS,
            progress=progress,
        )
    except SnapshotCancelledError:
        staging_path.unlink(missing_ok=True)
        raise
    except SnapshotBusyError:
        staging_path.unlink(missing_ok=True)
        raise
    except sqlite3.OperationalError as exc:
        if _is_busy(exc):
            raise SnapshotBusyError(
                f"Backup of {component} database failed on a lock: {exc}"
            ) from exc
        raise SnapshotError(f"Backup of {component} database failed: {exc}") from exc
    finally:
        target_connection.close()

    try:
        verification = _verify_copy(
            staging_path,
            component=component,
            source_path=source_path,
            snapshot_file=staging_path.name,
        )
    except Exception:
        staging_path.unlink(missing_ok=True)
        raise
    return verification


def _verify_copy(
    path: Path,
    *,
    component: str,
    source_path: Path,
    snapshot_file: str,
) -> tuple[SnapshotDatabaseResult, QueueLeaseState | None]:
    with sqlite3.connect(str(path)) as connection:
        connection.row_factory = sqlite3.Row
        ensure_database_compatibility(connection)
        supported = (
            TRANSACTION_SCHEMA_VERSION
            if component == _TRANSACTION_COMPONENT
            else QUEUE_SCHEMA_VERSION
        )
        schema_version = _component_version(connection, component)
        if schema_version != supported:
            raise SnapshotVerificationError(
                f"{component} snapshot schema version {schema_version} does not "
                f"match the supported version {supported}"
            )
        integrity = str(
            connection.execute("PRAGMA integrity_check").fetchone()[0]
        )
        if integrity.lower() != "ok":
            raise SnapshotVerificationError(
                f"{component} snapshot integrity check failed: {integrity}"
            )
        foreign_key_violations = len(
            connection.execute("PRAGMA foreign_key_check").fetchall()
        )
        if foreign_key_violations:
            raise SnapshotVerificationError(
                f"{component} snapshot contains {foreign_key_violations} "
                "foreign-key violations"
            )
        row_counts = _table_row_counts(connection)
        lease = _queue_lease_state(connection) if component == _QUEUE_COMPONENT else None
        # Materialize WAL content into the rollback-journal file so the
        # published copy is self-contained without a -wal sidecar and matches
        # the framework's storage policy on restore.
        journal_mode = str(
            connection.execute("PRAGMA journal_mode = DELETE").fetchone()[0]
        ).lower()
        if journal_mode != "delete":
            raise SnapshotVerificationError(
                f"{component} snapshot could not be normalized to rollback "
                f"journal mode; got {journal_mode!r}"
            )
    digest, size_bytes = _sha256_file(path)
    return SnapshotDatabaseResult(
        component=component,
        source_path=str(source_path),
        snapshot_file=snapshot_file,
        schema_version=schema_version,
        journal_mode=journal_mode,
        sha256=digest,
        size_bytes=size_bytes,
        row_counts=row_counts,
        foreign_key_violations=foreign_key_violations,
        integrity_check=integrity,
    ), lease


def _component_version(connection: sqlite3.Connection, component: str) -> int:
    row = connection.execute(
        f"SELECT version FROM {SCHEMA_VERSION_TABLE} WHERE component = ?",
        (component,),
    ).fetchone()
    if row is None:
        raise SnapshotVerificationError(
            f"Snapshot is missing the schema version marker for {component}"
        )
    return int(row[0])


def _table_row_counts(connection: sqlite3.Connection) -> dict[str, int]:
    rows = connection.execute(
        "SELECT name FROM sqlite_master WHERE type = 'table' "
        "AND name NOT LIKE 'sqlite_%' ORDER BY name"
    ).fetchall()
    counts: dict[str, int] = {}
    for row in rows:
        table_name = str(row[0])
        counts[table_name] = int(
            connection.execute(f'SELECT COUNT(*) FROM "{table_name}"').fetchone()[0]
        )
    for internal in _SQLITE_MASTER_TABLES:
        present = connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?",
            (internal,),
        ).fetchone()
        if present is not None:
            counts[internal] = int(
                connection.execute(f"SELECT COUNT(*) FROM {internal}").fetchone()[0]
            )
    return dict(sorted(counts.items()))


def _queue_lease_state(connection: sqlite3.Connection) -> QueueLeaseState:
    in_progress = int(
        connection.execute(
            "SELECT COUNT(*) FROM queue_items WHERE status = 'in_progress'"
        ).fetchone()[0]
    )
    complete_claims = int(
        connection.execute(
            "SELECT COUNT(*) FROM queue_items WHERE status = 'in_progress' "
            "AND claimed_by <> '' AND claim_token <> '' AND claimed_at IS NOT NULL"
        ).fetchone()[0]
    )
    open_attempts = int(
        connection.execute(
            "SELECT COUNT(*) FROM queue_attempts WHERE outcome = ''"
        ).fetchone()[0]
    )
    return QueueLeaseState(
        in_progress_items=in_progress,
        complete_claims=complete_claims,
        incomplete_claims=in_progress - complete_claims,
        open_attempts=open_attempts,
    )


def _sha256_file(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(_SHA256_CHUNK_SIZE)
            if not chunk:
                break
            digest.update(chunk)
            size += len(chunk)
    return digest.hexdigest(), size


def _build_manifest(
    *,
    transaction_snapshot: SnapshotDatabaseResult,
    queue_snapshot: SnapshotDatabaseResult,
    transaction_lease: QueueLeaseState | None,
    queue_lease: QueueLeaseState | None,
    created_at: str,
) -> _SnapshotManifest:
    databases: list[_DatabaseManifest] = []
    for result, lease in (
        (transaction_snapshot, transaction_lease),
        (queue_snapshot, queue_lease),
    ):
        entry: _DatabaseManifest = {
            "component": result.component,
            "source_path": result.source_path,
            "snapshot_file": result.snapshot_file,
            "schema_version": result.schema_version,
            "journal_mode": result.journal_mode,
            "sha256": result.sha256,
            "size_bytes": result.size_bytes,
            "row_counts": result.row_counts,
            "foreign_key_violations": result.foreign_key_violations,
            "integrity_check": result.integrity_check,
        }
        if lease is not None:
            entry["lease"] = {
                "in_progress_items": lease.in_progress_items,
                "complete_claims": lease.complete_claims,
                "incomplete_claims": lease.incomplete_claims,
                "open_attempts": lease.open_attempts,
            }
        databases.append(entry)
    return {
        "format_version": SNAPSHOT_FORMAT_VERSION,
        "created_at": created_at,
        "framework_version": _framework_version(),
        "completion_marker": COMPLETION_MARKER_NAME,
        "databases": databases,
    }


def _publish_snapshot(
    staging_path: Path, destination: Path, manifest: _SnapshotManifest
) -> None:
    manifest_path = staging_path / MANIFEST_NAME
    _write_json_atomic(manifest_path, manifest)
    marker_path = staging_path / COMPLETION_MARKER_NAME
    _write_bytes_atomic(
        marker_path,
        f"rpacore snapshot format v{SNAPSHOT_FORMAT_VERSION}\n".encode("utf-8"),
    )
    # The rename is the single publication point. A snapshot becomes visible
    # only when every file, the manifest and the marker are already durable.
    os.replace(staging_path, destination)


def _write_json_atomic(path: Path, payload: _SnapshotManifest) -> None:
    data = json.dumps(payload, sort_keys=True, indent=2).encode("utf-8")
    _write_bytes_atomic(path, data)


def _write_bytes_atomic(path: Path, data: bytes) -> None:
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary_path = Path(temporary)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise


def _load_manifest(snapshot_path: Path) -> _SnapshotManifest:
    manifest_path = snapshot_path / MANIFEST_NAME
    if not manifest_path.is_file():
        raise SnapshotVerificationError(
            f"Snapshot is missing {MANIFEST_NAME}: {snapshot_path}"
        )
    try:
        raw_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise SnapshotVerificationError(
            f"Snapshot manifest cannot be read: {exc}"
        ) from exc
    if not isinstance(raw_manifest, dict):
        raise SnapshotVerificationError("Snapshot manifest has an invalid shape")
    if raw_manifest.get("format_version") != SNAPSHOT_FORMAT_VERSION:
        raise SnapshotVerificationError(
            "Snapshot manifest format version "
            f"{raw_manifest.get('format_version')!r} is not supported; expected "
            f"{SNAPSHOT_FORMAT_VERSION}"
        )
    databases = raw_manifest.get("databases")
    if not isinstance(databases, list) or len(databases) != 2:
        raise SnapshotVerificationError(
            "Snapshot manifest must describe exactly two databases"
        )
    if not all(isinstance(entry, dict) for entry in databases):
        raise SnapshotVerificationError(
            "Snapshot manifest database entries have an invalid shape"
        )
    components = {str(entry.get("component")) for entry in databases}
    if components != {_TRANSACTION_COMPONENT, _QUEUE_COMPONENT}:
        raise SnapshotVerificationError(
            "Snapshot manifest must describe the transaction and queue components"
        )
    return raw_manifest  # type: ignore[return-value]


def _verify_snapshot_files(
    snapshot_path: Path, manifest: _SnapshotManifest
) -> None:
    for database in manifest["databases"]:
        snapshot_file = snapshot_path / str(database["snapshot_file"])
        if not snapshot_file.is_file():
            raise SnapshotVerificationError(
                f"Snapshot database file is missing: {snapshot_file}"
            )
        digest, size = _sha256_file(snapshot_file)
        if digest != database["sha256"]:
            raise SnapshotVerificationError(
                f"Snapshot checksum mismatch for {database['component']}"
            )
        if size != int(database["size_bytes"]):
            raise SnapshotVerificationError(
                f"Snapshot size mismatch for {database['component']}"
            )
        # No journal sidecars may accompany a published snapshot.
        for suffix in ("-wal", "-shm", "-journal"):
            sidecar = snapshot_path / (str(database["snapshot_file"]) + suffix)
            if sidecar.exists():
                raise SnapshotVerificationError(
                    f"Snapshot contains an unexpected journal sidecar: {sidecar.name}"
                )


def _copy_with_fsync(source: Path, target: Path) -> None:
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
    )
    temporary_path = Path(temporary)
    try:
        with os.fdopen(descriptor, "wb") as output, source.open("rb") as input_handle:
            shutil.copyfileobj(input_handle, output, length=_SHA256_CHUNK_SIZE)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary_path, target)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise


def _verify_restored_database(
    target: Path, component: str, recorded: _DatabaseManifest
) -> RestoredDatabase:
    digest, size = _sha256_file(target)
    if digest != recorded["sha256"] or size != recorded["size_bytes"]:
        raise SnapshotVerificationError(
            f"Restored {component} database checksum does not match the snapshot"
        )
    with sqlite3.connect(str(target)) as connection:
        connection.row_factory = sqlite3.Row
        ensure_database_compatibility(connection)
        schema_version = _component_version(connection, component)
        if schema_version != recorded["schema_version"]:
            raise SnapshotVerificationError(
                f"Restored {component} schema version {schema_version} does not "
                f"match the snapshot marker {recorded['schema_version']}"
            )
        integrity = str(
            connection.execute("PRAGMA integrity_check").fetchone()[0]
        )
        if integrity.lower() != "ok" or integrity != str(recorded["integrity_check"]):
            raise SnapshotVerificationError(
                f"Restored {component} integrity check failed: {integrity}"
            )
        violations = len(connection.execute("PRAGMA foreign_key_check").fetchall())
        if violations != recorded["foreign_key_violations"]:
            raise SnapshotVerificationError(
                f"Restored {component} foreign-key state changed: {violations}"
            )
        row_counts = _table_row_counts(connection)
        if row_counts != recorded["row_counts"]:
            raise SnapshotVerificationError(
                f"Restored {component} row counts do not match the snapshot"
            )
        journal_mode = str(
            connection.execute("PRAGMA journal_mode").fetchone()[0]
        ).lower()
        if journal_mode != str(recorded["journal_mode"]):
            raise SnapshotVerificationError(
                f"Restored {component} journal mode {journal_mode!r} does not "
                f"match the snapshot {recorded['journal_mode']!r}"
            )
        lease = None
        if component == _QUEUE_COMPONENT:
            lease = _queue_lease_state(connection)
            recorded_lease = recorded.get("lease")
            if not isinstance(recorded_lease, dict):
                raise SnapshotVerificationError(
                    "Snapshot is missing the recorded queue lease state"
                )
            typed_lease: _LeaseManifest = recorded_lease  # type: ignore[assignment]
            if (
                lease.in_progress_items != typed_lease["in_progress_items"]
                or lease.complete_claims != typed_lease["complete_claims"]
                or lease.incomplete_claims != typed_lease["incomplete_claims"]
                or lease.open_attempts != typed_lease["open_attempts"]
            ):
                raise SnapshotVerificationError(
                    "Restored queue lease state does not match the snapshot"
                )
    return RestoredDatabase(
        component=component,
        path=str(target),
        schema_version=schema_version,
        journal_mode=journal_mode,
        sha256=digest,
        size_bytes=size,
        row_counts=row_counts,
        foreign_key_violations=violations,
        integrity_check=integrity,
        lease=lease,
    )


def _raise_if_cancelled(cancel_event: Event | None) -> None:
    if cancel_event is not None and cancel_event.is_set():
        raise SnapshotCancelledError("Consistency snapshot was cancelled")


def _is_busy(error: sqlite3.OperationalError) -> bool:
    message = str(error).lower()
    return "locked" in message or "busy" in message


def _remove_staging(path: Path) -> None:
    shutil.rmtree(path, ignore_errors=True)


def _cleanup_created(paths: list[Path]) -> None:
    for path in paths:
        try:
            path.unlink(missing_ok=True)
        except OSError:
            continue
