"""CLI tests for the offline consistency snapshot and restore commands."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from rpacore import QueueItem, SqliteQueue, Transaction, save_transaction
from rpacore.cli import main
from rpacore._snapshot import (
    COMPLETION_MARKER_NAME,
    MANIFEST_NAME,
    QUEUE_SNAPSHOT_NAME,
    TRANSACTION_SNAPSHOT_NAME,
)
from tests.cli_helpers import run_cli


@pytest.fixture
def storage(tmp_path: Path) -> tuple[Path, Path]:
    live = tmp_path / "live"
    live.mkdir()
    transaction_db = live / "rpacore.db"
    queue_db = live / "queue.db"
    save_transaction(
        Transaction(reference="cli-ref", definition_identity="tests.snapshot_cli/v1"),
        str(transaction_db),
    )
    queue = SqliteQueue({"db_path": str(queue_db)})
    queue.add(QueueItem(reference="cli-queued", payload={"k": "v"}))
    return transaction_db, queue_db


def test_snapshot_create_then_restore_round_trip_json(
    storage: tuple[Path, Path], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    transaction_db, queue_db = storage
    snapshot_dir = tmp_path / "pair"

    create = main(
        [
            "snapshot",
            "create",
            "--transaction-db",
            str(transaction_db),
            "--queue-db",
            str(queue_db),
            "--output",
            str(snapshot_dir),
            "--json",
        ]
    )
    assert create == 0
    envelope = json.loads(capsys.readouterr().out)
    assert envelope["schema_version"] == 1
    assert envelope["command"] == "snapshot:create"
    assert envelope["snapshot_dir"] == str(snapshot_dir)
    assert (snapshot_dir / COMPLETION_MARKER_NAME).is_file()
    components = {entry["component"] for entry in envelope["databases"]}
    assert components == {"transactions", "queue"}
    for entry in envelope["databases"]:
        assert len(entry["sha256"]) == 64
        assert entry["integrity_check"] == "ok"
        assert entry["foreign_key_violations"] == 0

    transaction_target = tmp_path / "restored" / "rpacore.db"
    queue_target = tmp_path / "restored" / "queue.db"
    transaction_target.parent.mkdir()
    restore = main(
        [
            "snapshot",
            "restore",
            "--snapshot-dir",
            str(snapshot_dir),
            "--transaction-target",
            str(transaction_target),
            "--queue-target",
            str(queue_target),
            "--json",
        ]
    )
    assert restore == 0
    restore_envelope = json.loads(capsys.readouterr().out)
    assert restore_envelope["command"] == "snapshot:restore"
    assert restore_envelope["schema_version"] == 1
    queue_entry = next(
        entry
        for entry in restore_envelope["databases"]
        if entry["component"] == "queue"
    )
    assert "lease" in queue_entry
    assert queue_entry["lease"]["in_progress_items"] == 0
    with sqlite3.connect(str(queue_target)) as connection:
        assert (
            connection.execute("SELECT COUNT(*) FROM queue_items").fetchone()[0] == 1
        )


def test_human_readable_output_reports_components(
    storage: tuple[Path, Path], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    transaction_db, queue_db = storage
    snapshot_dir = tmp_path / "pair"
    assert (
        main(
            [
                "snapshot",
                "create",
                "--transaction-db",
                str(transaction_db),
                "--queue-db",
                str(queue_db),
                "--output",
                str(snapshot_dir),
            ]
        )
        == 0
    )
    out = capsys.readouterr().out
    assert "Consistency snapshot published" in out
    assert "transactions:" in out
    assert "queue:" in out
    assert MANIFEST_NAME in out


def test_missing_snapshot_subcommand_is_a_usage_error(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc_info:
        main(["snapshot"])
    assert exc_info.value.code == 2


def test_create_missing_required_flag_is_a_usage_error(
    capsys: pytest.CaptureFixture[str], storage: tuple[Path, Path]
) -> None:
    transaction_db, queue_db = storage
    with pytest.raises(SystemExit) as exc_info:
        main(
            [
                "snapshot",
                "create",
                "--transaction-db",
                str(transaction_db),
                "--queue-db",
                str(queue_db),
            ]
        )
    assert exc_info.value.code == 2


def test_create_busy_source_returns_execution_error_and_diagnostics_on_stderr(
    storage: tuple[Path, Path], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    transaction_db, queue_db = storage
    holder = sqlite3.connect(str(queue_db), timeout=0)
    holder.execute("PRAGMA busy_timeout = 0")
    holder.execute("BEGIN IMMEDIATE")
    try:
        code = main(
            [
                "snapshot",
                "create",
                "--transaction-db",
                str(transaction_db),
                "--queue-db",
                str(queue_db),
                "--output",
                str(tmp_path / "pair"),
                "--json",
            ]
        )
    finally:
        holder.rollback()
        holder.close()
    assert code == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "no partial snapshot was published" in captured.err
    assert not (tmp_path / "pair").exists()


def test_restore_missing_marker_returns_execution_error(
    storage: tuple[Path, Path], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    transaction_db, queue_db = storage
    snapshot_dir = tmp_path / "pair"
    assert (
        main(
            [
                "snapshot",
                "create",
                "--transaction-db",
                str(transaction_db),
                "--queue-db",
                str(queue_db),
                "--output",
                str(snapshot_dir),
            ]
        )
        == 0
    )
    capsys.readouterr()
    (snapshot_dir / COMPLETION_MARKER_NAME).unlink()
    code = main(
        [
            "snapshot",
            "restore",
            "--snapshot-dir",
            str(snapshot_dir),
            "--transaction-target",
            str(tmp_path / "t.db"),
            "--queue-target",
            str(tmp_path / "q.db"),
        ]
    )
    assert code == 1
    assert "incomplete" in capsys.readouterr().err


def test_subprocess_cli_creates_and_restores_snapshot(
    storage: tuple[Path, Path], tmp_path: Path
) -> None:
    transaction_db, queue_db = storage
    snapshot_dir = tmp_path / "subprocess-pair"
    created = run_cli(
        "snapshot",
        "create",
        "--transaction-db",
        str(transaction_db),
        "--queue-db",
        str(queue_db),
        "--output",
        str(snapshot_dir),
        "--json",
        cwd=tmp_path,
    )
    assert created.returncode == 0, created.stderr
    payload = json.loads(created.stdout)
    assert payload["command"] == "snapshot:create"
    assert (snapshot_dir / TRANSACTION_SNAPSHOT_NAME).is_file()
    assert (snapshot_dir / QUEUE_SNAPSHOT_NAME).is_file()

    restored = run_cli(
        "snapshot",
        "restore",
        "--snapshot-dir",
        str(snapshot_dir),
        "--transaction-target",
        str(tmp_path / "sub-t.db"),
        "--queue-target",
        str(tmp_path / "sub-q.db"),
        "--json",
        cwd=tmp_path,
    )
    assert restored.returncode == 0, restored.stderr
    assert json.loads(restored.stdout)["command"] == "snapshot:restore"
    assert (tmp_path / "sub-q.db").is_file()
