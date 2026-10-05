"""Default locations on disk."""

import os
import sys
from pathlib import Path

ENV_PREFIX = "RVC_NEXT_"
DATA_DIR_ENV = f"{ENV_PREFIX}DATA_DIR"
CONFIG_DIR_ENV = f"{ENV_PREFIX}CONFIG_DIR"


def default_data_dir() -> Path:
    """Return the data directory: ``RVC_NEXT_DATA_DIR``, or the platform's user data folder."""
    env = os.environ.get(DATA_DIR_ENV)
    if env:
        return Path(env).expanduser().resolve()
    if sys.platform == "win32":
        base = Path(os.environ.get("APPDATA") or Path.home() / "AppData" / "Roaming")
    elif sys.platform == "darwin":
        base = Path.home() / "Library" / "Application Support"
    else:
        base = Path(os.environ.get("XDG_DATA_HOME") or Path.home() / ".local" / "share")
    return (base / "rvc-next").resolve()
