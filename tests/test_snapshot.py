"""Regression tests for offline consistency snapshots and verified restores.

Every database here lives under pytest's temporary directory; the repository's
own storage files are never touched. Concurrency cases use real SQLite
connections and real threads so lock and WAL behavior is genuine.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path

import pytest

from rpacore import (
    QueueItem,
    QueueStatus,
    SnapshotBusyError,
    SnapshotCancelledError,
    SnapshotError,
    SnapshotVerificationError,
    SqliteQueue,
    Transaction,
    create_consistency_snapshot,
    restore_consistency_snapshot,
    save_transaction,
)
import rpacore._snapshot as snapshot_module
from rpacore._snapshot import (
    COMPLETION_MARKER_NAME,
    MANIFEST_NAME,
    QUEUE_SNAPSHOT_NAME,
    SNAPSHOT_FORMAT_VERSION,
    TRANSACTION_SNAPSHOT_NAME,
)
from rpacore._sqlite import QUEUE_SCHEMA_VERSION, TRANSACTION_SCHEMA_VERSION


# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def databases(tmp_path: Path) -> tuple[Path, Path]:
    transaction_db = tmp_path / "live" / "rpacore.db"
    queue_db = tmp_path / "live" / "queue.db"
    transaction_db.parent.mkdir()
    _seed_transactions(transaction_db)
    _seed_queue(queue_db)
    return transaction_db, queue_db


def _seed_transactions(path: Path) -> None:
    save_transaction(
        Transaction(reference="snapshot-ref-1", definition_identity="tests.snapshot/v1"),
        str(path),
    )
    save_transaction(
        Transaction(reference="snapshot-ref-2", definition_identity="tests.snapshot/v1"),
        str(path),
    )


def _seed_queue(path: Path) -> SqliteQueue:
    queue = SqliteQueue({"db_path": str(path), "retry_delay": 0})
    queue.add(QueueItem(reference="queued-1", payload={"order": 1}))
    queue.add(QueueItem(reference="queued-2", payload={"order": 2}))
    claimed = queue.next_item(worker_id="snapshot-worker")
    assert claimed is not None
    # The open claim produces a complete lease and an open attempt row.
    queue.renew_lease(
        claimed.id, claimed_by="snapshot-worker", claim_token=claimed.claim_token
    )
    return queue


def _grow_database(path: Path, cycles: int = 3) -> None:
    """Duplicate queue rows so a backup spans multiple page-copy batches."""
    with sqlite3.connect(str(path)) as connection:
        for cycle in range(cycles):
            connection.execute(
                "INSERT INTO queue_items "
                "(id, reference, payload, status, retry_count, created_at, "
                "claimed_by, claimed_at, transaction_id, claim_token) "
                "SELECT id || '-grow' || ?, reference, payload, status, "
                "retry_count, created_at, claimed_by, claimed_at, "
                "transaction_id, claim_token FROM queue_items",
                (str(cycle),),
            )
            connection.commit()


def _create_snapshot(
    databases: tuple[Path, Path], output: Path, **kwargs
):
    transaction_db, queue_db = databases
    output.parent.mkdir(parents=True, exist_ok=True)
    return create_consistency_snapshot(
        transaction_db_path=str(transaction_db),
        queue_db_path=str(queue_db),
        output_dir=str(output),
        **kwargs,
    )


def _journal_mode(path: Path) -> str:
    with sqlite3.connect(str(path)) as connection:
        return str(connection.execute("PRAGMA journal_mode").fetchone()[0]).lower()


def _row_count(path: Path, table: str) -> int:
    with sqlite3.connect(str(path)) as connection:
        return int(connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])


def _staging_leftovers(directory: Path) -> list[Path]:
    return [path for path in directory.iterdir() if path.name.startswith(".")]


# ---------------------------------------------------------------------------
# Happy path: publication, manifest, marker and verification facts
# ---------------------------------------------------------------------------


class TestSnapshotCreate:
    def test_publishes_self_contained_verified_pair(
        self, databases: tuple[Path, Path], tmp_path: Path
    ) -> None:
        transaction_db, queue_db = databases
        output = tmp_path / "snapshots" / "pair-1"

        result = _create_snapshot(databases, output)

        assert result.format_version == SNAPSHOT_FORMAT_VERSION
        assert Path(result.snapshot_dir) == output
        assert output.is_dir()
        assert (output / COMPLETION_MARKER_NAME).is_file()
        assert (output / MANIFEST_NAME).is_file()
        assert (output / TRANSACTION_SNAPSHOT_NAME).is_file()
        assert (output / QUEUE_SNAPSHOT_NAME).is_file()
        # Normalized rollback-journal copies carry no WAL or journal sidecars.
        assert not list(output.glob("*-wal"))
        assert not list(output.glob("*-shm"))
        assert not list(output.glob("*-journal"))

        manifest = json.loads((output / MANIFEST_NAME).read_text(encoding="utf-8"))
        assert manifest["format_version"] == SNAPSHOT_FORMAT_VERSION
        assert manifest["completion_marker"] == COMPLETION_MARKER_NAME
        components = {entry["component"] for entry in manifest["databases"]}
        assert components == {"transactions", "queue"}
        for entry in manifest["databases"]:
            snapshot_file = output / entry["snapshot_file"]
            assert snapshot_file.stat().st_size == entry["size_bytes"]
            assert _journal_mode(snapshot_file) == "delete"
            assert entry["foreign_key_violations"] == 0
            assert entry["integrity_check"].lower() == "ok"
            assert len(entry["sha256"]) == 64

        by_component = {entry["component"]: entry for entry in manifest["databases"]}
        assert (
            by_component["transactions"]["schema_version"]
            == TRANSACTION_SCHEMA_VERSION
        )
        assert by_component["queue"]["schema_version"] == QUEUE_SCHEMA_VERSION
        assert by_component["transactions"]["row_counts"]["transactions"] == 2
        queue_entry = by_component["queue"]
        assert queue_entry["row_counts"]["queue_items"] == 2
        lease = queue_entry["lease"]
        assert lease["in_progress_items"] == 1
        assert lease["complete_claims"] == 1
        assert lease["incomplete_claims"] == 0
        assert lease["open_attempts"] == 1

    def test_result_dataclasses_match_manifest(
        self, databases: tuple[Path, Path], tmp_path: Path
    ) -> None:
        result = _create_snapshot(databases, tmp_path / "pair")

        assert len(result.databases) == 2
        components = {database.component for database in result.databases}
        assert components == {"transactions", "queue"}
        for database in result.databases:
            assert database.foreign_key_violations == 0
            assert database.integrity_check == "ok"
            assert database.journal_mode == "delete"
            assert database.size_bytes > 0

    def test_does_not_modify_source_databases(
        self, databases: tuple[Path, Path], tmp_path: Path
    ) -> None:
        transaction_db, queue_db = databases
        before = {
            str(transaction_db): transaction_db.read_bytes(),
            str(queue_db): queue_db.read_bytes(),
        }

        _create_snapshot(databases, tmp_path / "pair")

        assert transaction_db.read_bytes() == before[str(transaction_db)]
        assert queue_db.read_bytes() == before[str(queue_db)]
        assert _row_count(transaction_db, "transactions") == 2
        assert _journal_mode(queue_db) == "delete"


# ---------------------------------------------------------------------------
# Repeated snapshots and input validation
# ---------------------------------------------------------------------------


class TestSnapshotRefusals:
    def test_repeated_snapshot_never_overwrites(
        self, databases: tuple[Path, Path], tmp_path: Path
    ) -> None:
        output = tmp_path / "pair"
        first = _create_snapshot(databases, output)
        marker = (output / COMPLETION_MARKER_NAME).read_bytes()

        with pytest.raises(SnapshotError, match="already exists"):
            _create_snapshot(databases, output)

        # The first publication is byte-identical after the refused retry.
        assert (output / COMPLETION_MARKER_NAME).read_bytes() == marker
        assert Path(first.snapshot_dir) == output
        assert not _staging_leftovers(tmp_path)

    def test_identical_source_paths_are_rejected(
        self, databases: tuple[Path, Path], tmp_path: Path
    ) -> None:
        transaction_db, _ = databases
        with pytest.raises(SnapshotError, match="different files"):
            create_consistency_snapshot(
                transaction_db_path=str(transaction_db),
                queue_db_path=str(transaction_db),
                output_dir=str(tmp_path / "pair"),
            )
        assert not (tmp_path / "pair").exists()

    def test_missing_source_fails_without_staging(
        self, databases: tuple[Path, Path], tmp_path: Path
    ) -> None:
        _, queue_db = databases
        with pytest.raises(SnapshotError):
            create_consistency_snapshot(
                transaction_db_path=str(tmp_path / "missing.db"),
                queue_db_path=str(queue_db),
                output_dir=str(tmp_path / "pair"),
            )
        assert not (tmp_path / "pair").exists()
        assert not _staging_leftovers(tmp_path)


# ---------------------------------------------------------------------------
# Cancellation
# ---------------------------------------------------------------------------


class TestSnapshotCancellation:
    def test_pre_set_event_fails_before_publication(
        self, databases: tuple[Path, Path], tmp_path: Path
    ) -> None:
        event = threading.Event()
        event.set()
        with pytest.raises(SnapshotCancelledError):
            _create_snapshot(databases, tmp_path / "pair", cancel_event=event)
        assert not (tmp_path / "pair").exists()

    def test_cancellation_during_backup_removes_staging(
        self, databases: tuple[Path, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        event = threading.Event()
        original_backup = snapshot_module._backup_database

        def cancel_then_backup(**kwargs):
            if kwargs["component"] == "queue":
                # The pinned readers already exist; cancelling here forces the
                # queue backup's progress handler to raise on its first call.
                event.set()
            return original_backup(**kwargs)

        monkeypatch.setattr(
            snapshot_module, "_backup_database", cancel_then_backup
        )

        with pytest.raises(SnapshotCancelledError):
            _create_snapshot(databases, tmp_path / "pair", cancel_event=event)

        assert not (tmp_path / "pair").exists()
        assert not _staging_leftovers(tmp_path)


# ---------------------------------------------------------------------------
# Real lock conflicts under both rollback-journal and WAL policies
# ---------------------------------------------------------------------------


class TestLockConflict:
    @pytest.mark.parametrize("journal_mode", ["delete", "wal"])
    def test_held_write_lock_fails_busy_without_half_set(
        self,
        databases: tuple[Path, Path],
        tmp_path: Path,
        journal_mode: str,
    ) -> None:
        transaction_db, queue_db = databases
        if journal_mode == "wal":
            with sqlite3.connect(str(queue_db)) as connection:
                if (
                    str(
                        connection.execute(
                            "PRAGMA journal_mode = WAL"
                        ).fetchone()[0]
                    ).lower()
                    != "wal"
                ):
                    pytest.skip("WAL journal mode is not available on this build")
        output = tmp_path / "pair"

        # A connection already holding the write lock prevents the snapshot
        # sentry from acquiring BEGIN IMMEDIATE; the failure is immediate
        # because the sentry runs with busy_timeout disabled.
        writer = sqlite3.connect(str(queue_db), timeout=0)
        writer.execute("PRAGMA busy_timeout = 0")
        writer.execute("BEGIN IMMEDIATE")
        writer.execute("UPDATE queue_items SET retry_count = retry_count + 1")
        try:
            with pytest.raises(SnapshotBusyError, match="[Ll]ock"):
                _create_snapshot(databases, output)
        finally:
            writer.rollback()
            writer.close()

        assert not output.exists()
        assert not _staging_leftovers(tmp_path)
        # Existing storage is untouched and still usable.
        assert _row_count(queue_db, "queue_items") == 2

    @pytest.mark.parametrize("journal_mode", ["delete", "wal"])
    def test_sentry_blocks_concurrent_writer_commits_for_the_window(
        self,
        databases: tuple[Path, Path],
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        journal_mode: str,
    ) -> None:
        transaction_db, queue_db = databases
        if journal_mode == "wal":
            with sqlite3.connect(str(queue_db)) as connection:
                if (
                    str(
                        connection.execute(
                            "PRAGMA journal_mode = WAL"
                        ).fetchone()[0]
                    ).lower()
                    != "wal"
                ):
                    pytest.skip("WAL journal mode is not available on this build")
        _grow_database(queue_db)

        writer_outcome: dict[str, object] = {}
        original_open = snapshot_module._open_database_guard
        observed_guards: list[snapshot_module._DatabaseGuard] = []

        def observe_guard(path: Path, *, component: str):
            guard = original_open(path, component=component)
            observed_guards.append(guard)
            if component == "queue":
                # The sentry is now held. A real writer on another connection
                # must be refused the write lock for as long as the window is
                # open, in both journal modes.
                writer = sqlite3.connect(
                    str(queue_db), timeout=0, check_same_thread=False
                )
                writer.execute("PRAGMA busy_timeout = 0")
                try:
                    writer.execute("BEGIN IMMEDIATE")
                    writer.execute(
                        "UPDATE queue_items SET retry_count = retry_count + 1"
                    )
                    writer.commit()
                    writer_outcome["committed"] = True
                except sqlite3.OperationalError as exc:
                    writer_outcome["committed"] = False
                    writer_outcome["error"] = str(exc)
                    writer.rollback()
                finally:
                    writer.close()
            return guard

        monkeypatch.setattr(snapshot_module, "_open_database_guard", observe_guard)
        result = _create_snapshot(databases, tmp_path / "pair")

        assert Path(result.snapshot_dir).is_dir()
        assert writer_outcome["committed"] is False
        assert "locked" in str(writer_outcome["error"]).lower()
        # Guards were released when the snapshot returned.
        for guard in observed_guards:
            with pytest.raises(sqlite3.ProgrammingError):
                guard.sentry.execute("SELECT 1")

    def test_lock_acquired_mid_backup_exhausts_progress_retries(
        self, databases: tuple[Path, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        transaction_db, queue_db = databases
        _grow_database(queue_db)
        output = tmp_path / "pair"
        original_backup = snapshot_module._backup_database

        class LockInjectingReader:
            """Proxies the pinned reader and holds an EXCLUSIVE lock at backup.

            The queue sentry's RESERVED lock is released for the duration of
            this backup and replaced by an EXCLUSIVE lock on this separate
            connection. The backup's page-copy steps then return SQLITE_BUSY
            until the bounded progress retry count raises SnapshotBusyError.
            """

            def __init__(
                self, real_reader: sqlite3.Connection, sentry: sqlite3.Connection
            ) -> None:
                self._real_reader = real_reader
                self._sentry = sentry
                self.holder: sqlite3.Connection | None = None

            def backup(self, target, *args, **kwargs):  # type: ignore[no-untyped-def]
                self._sentry.rollback()
                self.holder = sqlite3.connect(
                    str(queue_db), timeout=0, check_same_thread=False
                )
                self.holder.execute("PRAGMA busy_timeout = 0")
                self.holder.execute("BEGIN EXCLUSIVE")
                self.holder.execute("SELECT COUNT(*) FROM queue_items").fetchone()
                try:
                    return self._real_reader.backup(target, *args, **kwargs)
                finally:
                    assert self.holder is not None
                    self.holder.rollback()
                    self.holder.close()
                    self.holder = None

        injected: dict[str, LockInjectingReader] = {}

        def lock_then_backup(**kwargs):
            if kwargs["component"] == "queue":
                # Recover the live guard's sentry through the create flow.
                sentry = injected_sentry["connection"]
                proxy = LockInjectingReader(kwargs["reader"], sentry)
                injected["queue"] = proxy
                kwargs["reader"] = proxy
            return original_backup(**kwargs)

        injected_sentry: dict[str, sqlite3.Connection] = {}
        original_guard = snapshot_module._open_database_guard

        def capture_guard(path: Path, *, component: str):
            guard = original_guard(path, component=component)
            if component == "queue":
                injected_sentry["connection"] = guard.sentry
            return guard

        monkeypatch.setattr(snapshot_module, "_open_database_guard", capture_guard)
        monkeypatch.setattr(snapshot_module, "_backup_database", lock_then_backup)

        with pytest.raises(SnapshotBusyError, match="repeatedly met"):
            _create_snapshot(databases, output)
        assert not output.exists()
        assert not _staging_leftovers(tmp_path)
        assert injected["queue"].holder is None

    def test_progress_handler_reads_raw_sqlite_result_code(self) -> None:
        # Regression guard: CPython invokes the handler as
        # progress(rc, remaining, page_count). Treating the third argument as a
        # boolean retry flag (an older misreading) would count every ordinary
        # step as busy and break snapshots on healthy databases.
        observed: list[tuple[int, int, int]] = []

        def handler(rc: int, remaining: int, page_count: int) -> None:
            observed.append((rc, remaining, page_count))

        handler(snapshot_module._SQLITE_RESULT_OK, 10, 20)
        handler(snapshot_module._SQLITE_RESULT_BUSY, 10, 20)
        assert observed == [(0, 10, 20), (5, 10, 20)]
        assert snapshot_module._SQLITE_RESULT_LOCKED == 6


# ---------------------------------------------------------------------------
# WAL mode: sentry locking and self-contained normalized copies
# ---------------------------------------------------------------------------


class TestWalMode:
    def _ensure_wal(self, queue_db: Path) -> None:
        with sqlite3.connect(str(queue_db)) as connection:
            mode = str(
                connection.execute("PRAGMA journal_mode = WAL").fetchone()[0]
            ).lower()
        if mode != "wal":
            pytest.skip("WAL journal mode is not available on this build")
        assert _journal_mode(queue_db) == "wal"

    def test_wal_snapshot_is_consistent_and_self_contained(
        self, databases: tuple[Path, Path], tmp_path: Path
    ) -> None:
        transaction_db, queue_db = databases
        self._ensure_wal(queue_db)
        before_items = _row_count(queue_db, "queue_items")

        result = _create_snapshot(databases, tmp_path / "wal-pair")

        queue_snapshot = Path(result.snapshot_dir) / QUEUE_SNAPSHOT_NAME
        # The published copy is normalized to rollback journal and carries no
        # WAL or shared-memory sidecars.
        assert _journal_mode(queue_snapshot) == "delete"
        assert not (
            Path(result.snapshot_dir) / (QUEUE_SNAPSHOT_NAME + "-wal")
        ).exists()
        assert not (
            Path(result.snapshot_dir) / (QUEUE_SNAPSHOT_NAME + "-shm")
        ).exists()
        with sqlite3.connect(str(queue_snapshot)) as connection:
            assert (
                int(connection.execute("SELECT COUNT(*) FROM queue_items").fetchone()[0])
                == before_items
            )
        # The live source stays in WAL mode and is unmodified in row count.
        assert _journal_mode(queue_db) == "wal"
        assert _row_count(queue_db, "queue_items") == before_items
        # Repeated snapshots remain refused on the same output path.
        with pytest.raises(SnapshotError, match="already exists"):
            _create_snapshot(databases, tmp_path / "wal-pair")

    def test_wal_snapshot_restores_to_a_verified_rollback_journal_copy(
        self, databases: tuple[Path, Path], tmp_path: Path
    ) -> None:
        transaction_db, queue_db = databases
        self._ensure_wal(queue_db)
        result = _create_snapshot(databases, tmp_path / "wal-pair")

        restore_result = restore_consistency_snapshot(
            snapshot_dir=result.snapshot_dir,
            transaction_target_path=str(tmp_path / "rt.db"),
            queue_target_path=str(tmp_path / "rq.db"),
        )
        for restored in restore_result.databases:
            assert restored.integrity_check == "ok"
            assert restored.foreign_key_violations == 0
            assert restored.journal_mode == "delete"
        assert _journal_mode(tmp_path / "rq.db") == "delete"
        assert _journal_mode(queue_db) == "wal"

    def test_concurrent_wal_writer_is_blocked_then_snapshot_succeeds(
        self,
        databases: tuple[Path, Path],
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        transaction_db, queue_db = databases
        self._ensure_wal(queue_db)
        _grow_database(queue_db)

        start_writer = threading.Event()
        writer_finished = threading.Event()
        blocked_at_least_once = threading.Event()

        def writer_worker() -> None:
            start_writer.wait(5)
            connection = sqlite3.connect(str(queue_db), timeout=0)
            connection.execute("PRAGMA busy_timeout = 0")
            attempts = 0
            while attempts < 100:
                try:
                    connection.execute("BEGIN IMMEDIATE")
                    connection.execute(
                        "UPDATE queue_items SET retry_count = retry_count + 1"
                    )
                    connection.commit()
                except sqlite3.OperationalError:
                    blocked_at_least_once.set()
                    connection.rollback()
                attempts += 1
            connection.close()
            writer_finished.set()

        original_guard = snapshot_module._open_database_guard

        def start_writer_when_both_guards_held(path: Path, *, component: str):
            guard = original_guard(path, component=component)
            if component == "queue":
                # Both sentries are now held; release a background writer to
                # hammer the write lock for the remainder of the backup window.
                start_writer.set()
            return guard

        monkeypatch.setattr(
            snapshot_module,
            "_open_database_guard",
            start_writer_when_both_guards_held,
        )
        writer_thread = threading.Thread(target=writer_worker)
        writer_thread.start()
        try:
            result = _create_snapshot(databases, tmp_path / "wal-busy-pair")
        finally:
            writer_thread.join()
            assert writer_finished.wait(5)

        assert blocked_at_least_once.is_set()
        assert Path(result.snapshot_dir).is_dir()
        restore_result = restore_consistency_snapshot(
            snapshot_dir=result.snapshot_dir,
            transaction_target_path=str(tmp_path / "rt.db"),
            queue_target_path=str(tmp_path / "rq.db"),
        )
        for restored in restore_result.databases:
            assert restored.integrity_check == "ok"
            assert restored.foreign_key_violations == 0


# ---------------------------------------------------------------------------
# Restore: verification, new paths, originals untouched
# ---------------------------------------------------------------------------


class TestRestore:
    def test_restore_to_new_paths_verifies_every_recorded_fact(
        self, databases: tuple[Path, Path], tmp_path: Path
    ) -> None:
        transaction_db, queue_db = databases
        original_transaction_bytes = transaction_db.read_bytes()
        original_queue_bytes = queue_db.read_bytes()
        snapshot = _create_snapshot(databases, tmp_path / "pair")

        restored_dir = tmp_path / "restored"
        restored_dir.mkdir()
        restored_transaction = restored_dir / "rpacore.db"
        restored_queue = restored_dir / "queue.db"

        result = restore_consistency_snapshot(
            snapshot_dir=snapshot.snapshot_dir,
            transaction_target_path=str(restored_transaction),
            queue_target_path=str(restored_queue),
        )

        assert restored_transaction.is_file() and restored_queue.is_file()
        assert len(result.databases) == 2
        queue_result = next(
            database for database in result.databases if database.component == "queue"
        )
        assert queue_result.lease is not None
        assert queue_result.lease.in_progress_items == 1
        assert queue_result.lease.complete_claims == 1
        assert queue_result.lease.open_attempts == 1
        for database in result.databases:
            assert database.foreign_key_violations == 0
            assert database.integrity_check == "ok"
            assert database.journal_mode == "delete"

        # Restored copies are independently readable with matching row counts.
        assert _row_count(restored_transaction, "transactions") == 2
        assert _row_count(restored_queue, "queue_items") == 2
        with sqlite3.connect(str(restored_queue)) as connection:
            in_progress = int(
                connection.execute(
                    "SELECT COUNT(*) FROM queue_items WHERE status = ?",
                    (QueueStatus.IN_PROGRESS,),
                ).fetchone()[0]
            )
        assert in_progress == 1

        # Originals are untouched.
        assert transaction_db.read_bytes() == original_transaction_bytes
        assert queue_db.read_bytes() == original_queue_bytes

    def test_restore_refuses_existing_targets(
        self, databases: tuple[Path, Path], tmp_path: Path
    ) -> None:
        snapshot = _create_snapshot(databases, tmp_path / "pair")
        existing = tmp_path / "existing.db"
        existing.write_text("not a database", encoding="utf-8")

        with pytest.raises(SnapshotError, match="already exists"):
            restore_consistency_snapshot(
                snapshot_dir=snapshot.snapshot_dir,
                transaction_target_path=str(existing),
                queue_target_path=str(tmp_path / "new-queue.db"),
            )
        assert existing.read_text(encoding="utf-8") == "not a database"

    def test_restore_refuses_identical_targets(
        self, databases: tuple[Path, Path], tmp_path: Path
    ) -> None:
        snapshot = _create_snapshot(databases, tmp_path / "pair")
        target = tmp_path / "same.db"
        with pytest.raises(SnapshotError, match="different files"):
            restore_consistency_snapshot(
                snapshot_dir=snapshot.snapshot_dir,
                transaction_target_path=str(target),
                queue_target_path=str(target),
            )

    def test_restore_without_completion_marker_is_rejected(
        self, databases: tuple[Path, Path], tmp_path: Path
    ) -> None:
        snapshot = _create_snapshot(databases, tmp_path / "pair")
        (Path(snapshot.snapshot_dir) / COMPLETION_MARKER_NAME).unlink()

        with pytest.raises(SnapshotVerificationError, match="incomplete"):
            restore_consistency_snapshot(
                snapshot_dir=snapshot.snapshot_dir,
                transaction_target_path=str(tmp_path / "t.db"),
                queue_target_path=str(tmp_path / "q.db"),
            )
        assert not (tmp_path / "t.db").exists()
        assert not (tmp_path / "q.db").exists()

    def test_tampered_snapshot_checksum_fails_before_targets_are_published(
        self, databases: tuple[Path, Path], tmp_path: Path
    ) -> None:
        snapshot = _create_snapshot(databases, tmp_path / "pair")
        queue_copy = Path(snapshot.snapshot_dir) / QUEUE_SNAPSHOT_NAME
        with sqlite3.connect(str(queue_copy)) as connection:
            connection.execute("PRAGMA journal_mode = DELETE")
            existing_id = str(
                connection.execute("SELECT id FROM queue_items LIMIT 1").fetchone()[0]
            )
            connection.execute(
                "INSERT INTO queue_admin_events "
                "(item_id, action, reason, previous_status, new_status, created_at) "
                "VALUES (?, 'fail', 'tampered', 'pending', 'failed', "
                "'2026-11-13T00:00:00+00:00')",
                (existing_id,),
            )
            connection.commit()

        transaction_target = tmp_path / "restored-t.db"
        queue_target = tmp_path / "restored-q.db"
        with pytest.raises(SnapshotVerificationError, match="[Cc]hecksum"):
            restore_consistency_snapshot(
                snapshot_dir=snapshot.snapshot_dir,
                transaction_target_path=str(transaction_target),
                queue_target_path=str(queue_target),
            )
        assert not transaction_target.exists()
        assert not queue_target.exists()

    def test_unknown_manifest_format_is_rejected(
        self, databases: tuple[Path, Path], tmp_path: Path
    ) -> None:
        snapshot = _create_snapshot(databases, tmp_path / "pair")
        manifest_path = Path(snapshot.snapshot_dir) / MANIFEST_NAME
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["format_version"] = 999
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

        with pytest.raises(SnapshotVerificationError, match="format version"):
            restore_consistency_snapshot(
                snapshot_dir=snapshot.snapshot_dir,
                transaction_target_path=str(tmp_path / "t.db"),
                queue_target_path=str(tmp_path / "q.db"),
            )

    def test_snapshot_can_be_restored_twice_to_different_targets(
        self, databases: tuple[Path, Path], tmp_path: Path
    ) -> None:
        snapshot = _create_snapshot(databases, tmp_path / "pair")
        first = restore_consistency_snapshot(
            snapshot_dir=snapshot.snapshot_dir,
            transaction_target_path=str(tmp_path / "a-t.db"),
            queue_target_path=str(tmp_path / "a-q.db"),
        )
        second = restore_consistency_snapshot(
            snapshot_dir=snapshot.snapshot_dir,
            transaction_target_path=str(tmp_path / "b-t.db"),
            queue_target_path=str(tmp_path / "b-q.db"),
        )
        assert [database.sha256 for database in first.databases] == [
            database.sha256 for database in second.databases
        ]
