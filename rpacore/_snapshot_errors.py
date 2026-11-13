"""Exceptions raised by offline consistency snapshot and restore operations."""

from __future__ import annotations


class SnapshotError(RuntimeError):
    """Base class for snapshot/restore failures.

    A failed snapshot never publishes a directory and leaves every existing
    filesystem destination untouched; a failed restore removes only the new
    target files it created and never writes to the original databases.
    """


class SnapshotBusyError(SnapshotError):
    """A source database could not be read or backed up because it was locked."""


class SnapshotCancelledError(SnapshotError):
    """A snapshot operation was cancelled cooperatively before publication."""


class SnapshotVerificationError(SnapshotError):
    """A snapshot or restored database failed its recorded verification."""
