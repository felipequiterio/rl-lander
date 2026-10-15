"""Atomic file writes: write to a temp file in the same directory, then os.replace().

A rename within one filesystem is atomic, so a concurrent reader (the viewer, a
training process checking the champion) sees either the old file or the new
one, never a half-written file.
"""

from __future__ import annotations

import json
import os
import tempfile
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any


def replace(src: str, dst: Path) -> None:
    # On Windows, os.replace fails if another process has dst open at that instant
    # (e.g. the viewer reading it). It's a transient condition, so retry briefly.
    for attempt in range(20):
        try:
            os.replace(src, dst)
            return
        except PermissionError:
            if attempt == 19:
                raise
            time.sleep(0.05)


def atomic_write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(data, f, indent=2)
            f.flush()
            os.fsync(f.fileno())
        replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def read_json(path: Path) -> dict[str, Any] | None:
    """Return the parsed file, or None if it doesn't exist (yet) or can't be read."""
    try:
        with open(path) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError, PermissionError):
        return None


def atomic_write_with(path: Path, write: Callable[[str], None], suffix: str = ".tmp") -> None:
    """Atomically create ``path`` using ``write(tmp_path)`` to produce the content."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.stem}.", suffix=suffix)
    os.close(fd)
    try:
        write(tmp)
        replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise
