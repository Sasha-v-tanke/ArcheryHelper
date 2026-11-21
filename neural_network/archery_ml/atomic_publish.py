"""Publish a completed directory without replacing an existing destination."""

from __future__ import annotations

import ctypes
import errno
import os
import sys
from pathlib import Path


def publish_directory(staging: Path, destination: Path) -> None:
    """Atomically expose staging, failing if destination exists at rename time."""
    staging = Path(staging)
    destination = Path(destination)
    if sys.platform == "win32":
        # Windows os.rename already refuses an existing destination.
        os.rename(staging, destination)
        return

    libc = ctypes.CDLL(None, use_errno=True)
    source = os.fsencode(staging)
    target = os.fsencode(destination)
    if sys.platform == "darwin":
        rename = libc.renamex_np
        rename.argtypes = (ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint)
        result = rename(source, target, 0x00000004)  # RENAME_EXCL
    elif sys.platform.startswith("linux"):
        rename = libc.renameat2
        rename.argtypes = (ctypes.c_int, ctypes.c_char_p,
                           ctypes.c_int, ctypes.c_char_p, ctypes.c_uint)
        result = rename(-100, source, -100, target, 1)  # AT_FDCWD, RENAME_NOREPLACE
    else:
        raise RuntimeError("atomic no-replace directory publication is unsupported")
    if result != 0:
        error = ctypes.get_errno()
        if error == errno.EEXIST:
            raise FileExistsError(error, os.strerror(error), destination)
        raise OSError(error, os.strerror(error), destination)
