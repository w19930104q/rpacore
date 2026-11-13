"""CLI commands for offline consistency snapshots and verified restores.

Both commands require explicit database paths; there is no implicit
working-database fallback, so a snapshot can never silently target the wrong
storage. Snapshot publication and restore verification are all-or-nothing.
"""

from __future__ import annotations

import argparse
import json
import sys

from rpacore._snapshot import (
    MANIFEST_NAME,
    SNAPSHOT_FORMAT_VERSION,
    QueueLeaseState,
    SnapshotDatabaseResult,
    SnapshotRestoreResult,
    SnapshotResult,
    create_consistency_snapshot,
    restore_consistency_snapshot,
)
from rpacore._snapshot_errors import (
    SnapshotBusyError,
    SnapshotCancelledError,
    SnapshotError,
    SnapshotVerificationError,
)

SUCCESS = 0
EXECUTION_ERROR = 1
USAGE_ERROR = 2
CLI_SNAPSHOT_FORMAT_VERSION = 1


def _run_snapshot_command(args: argparse.Namespace) -> int:
    if not args.transaction_db or not args.queue_db or not args.output:
        _print_error(
            "snapshot create requires --transaction-db, --queue-db and --output"
        )
        return USAGE_ERROR
    try:
        result = create_consistency_snapshot(
            transaction_db_path=args.transaction_db,
            queue_db_path=args.queue_db,
            output_dir=args.output,
        )
    except SnapshotCancelledError as exc:
        _print_error(f"Snapshot cancelled before publication: {exc}")
        return EXECUTION_ERROR
    except SnapshotBusyError as exc:
        _print_error(
            f"Snapshot failed because a database was busy or locked: {exc}; "
            "no partial snapshot was published"
        )
        return EXECUTION_ERROR
    except SnapshotVerificationError as exc:
        _print_error(f"Snapshot verification failed: {exc}; no partial snapshot was published")
        return EXECUTION_ERROR
    except SnapshotError as exc:
        _print_error(f"Snapshot failed: {exc}; no partial snapshot was published")
        return EXECUTION_ERROR
    except Exception as exc:
        _print_error(
            f"Snapshot failed ({type(exc).__name__}); no partial snapshot was published"
        )
        return EXECUTION_ERROR

    if args.json:
        _write_json(_snapshot_envelope(result))
    else:
        print(f"Consistency snapshot published: {result.snapshot_dir}")
        for database in result.databases:
            print(
                f"  {database.component}: schema v{database.schema_version}, "
                f"{database.size_bytes} bytes, sha256 {database.sha256}"
            )
        print(f"  manifest: {MANIFEST_NAME}, format v{SNAPSHOT_FORMAT_VERSION}")
    return SUCCESS


def _run_restore_command(args: argparse.Namespace) -> int:
    if (
        not args.snapshot_dir
        or not args.transaction_target
        or not args.queue_target
    ):
        _print_error(
            "snapshot restore requires --snapshot-dir, --transaction-target and --queue-target"
        )
        return USAGE_ERROR
    try:
        result = restore_consistency_snapshot(
            snapshot_dir=args.snapshot_dir,
            transaction_target_path=args.transaction_target,
            queue_target_path=args.queue_target,
        )
    except SnapshotVerificationError as exc:
        _print_error(f"Snapshot or restored database failed verification: {exc}")
        return EXECUTION_ERROR
    except SnapshotError as exc:
        _print_error(f"Snapshot restore failed: {exc}")
        return EXECUTION_ERROR
    except Exception as exc:
        _print_error(f"Snapshot restore failed ({type(exc).__name__})")
        return EXECUTION_ERROR

    if args.json:
        _write_json(_restore_envelope(result))
    else:
        print(f"Consistency snapshot restored from: {result.snapshot_dir}")
        for database in result.databases:
            line = (
                f"  {database.component}: {database.path} schema v{database.schema_version} "
                f"journal={database.journal_mode} sha256 {database.sha256}"
            )
            print(line)
            if database.lease is not None:
                lease = database.lease
                print(
                    "    lease: "
                    f"in_progress={lease.in_progress_items} "
                    f"complete_claims={lease.complete_claims} "
                    f"incomplete_claims={lease.incomplete_claims} "
                    f"open_attempts={lease.open_attempts}"
                )
        print("  original databases were not modified")
    return SUCCESS


def _snapshot_envelope(result: SnapshotResult) -> dict[str, object]:
    return {
        "schema_version": CLI_SNAPSHOT_FORMAT_VERSION,
        "command": "snapshot:create",
        "format_version": result.format_version,
        "snapshot_dir": result.snapshot_dir,
        "created_at": result.created_at,
        "databases": [_database_envelope(database) for database in result.databases],
    }


def _restore_envelope(result: SnapshotRestoreResult) -> dict[str, object]:
    return {
        "schema_version": CLI_SNAPSHOT_FORMAT_VERSION,
        "command": "snapshot:restore",
        "snapshot_dir": result.snapshot_dir,
        "restored_at": result.restored_at,
        "databases": [
            {
                "component": database.component,
                "path": database.path,
                "schema_version": database.schema_version,
                "journal_mode": database.journal_mode,
                "sha256": database.sha256,
                "size_bytes": database.size_bytes,
                "row_counts": database.row_counts,
                "foreign_key_violations": database.foreign_key_violations,
                "integrity_check": database.integrity_check,
                **(
                    {"lease": _lease_envelope(database.lease)}
                    if database.lease is not None
                    else {}
                ),
            }
            for database in result.databases
        ],
    }


def _database_envelope(database: SnapshotDatabaseResult) -> dict[str, object]:
    return {
        "component": database.component,
        "source_path": database.source_path,
        "snapshot_file": database.snapshot_file,
        "schema_version": database.schema_version,
        "journal_mode": database.journal_mode,
        "sha256": database.sha256,
        "size_bytes": database.size_bytes,
        "row_counts": database.row_counts,
        "foreign_key_violations": database.foreign_key_violations,
        "integrity_check": database.integrity_check,
    }


def _lease_envelope(lease: QueueLeaseState) -> dict[str, int]:
    return {
        "in_progress_items": lease.in_progress_items,
        "complete_claims": lease.complete_claims,
        "incomplete_claims": lease.incomplete_claims,
        "open_attempts": lease.open_attempts,
    }


def _write_json(data: dict[str, object]) -> None:
    print(json.dumps(data, sort_keys=True, separators=(",", ":")))


def _print_error(message: str) -> None:
    print(f"rpacore: {message}", file=sys.stderr)
