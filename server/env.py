"""Environment loading: real env > .env (local, gitignored) > ghost.env (committed defaults).

`ghost.env` is the project's config file: it makes a bare
`uvicorn server.main:app` start the mind against a local LM Studio. Put
machine-specific overrides and secrets in `.env`; anything exported in the
shell wins over both. GHOST_CONFIG points at a different defaults file;
GHOST_NO_CONFIG=1 loads no file at all.
"""
import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
_loaded = False


def load_env() -> None:
    """Idempotent; call before any os.getenv-backed settings class is defined."""
    global _loaded
    if _loaded:
        return
    _loaded = True
    if os.getenv("GHOST_NO_CONFIG") == "1":
        return  # hermetic runs (the test suite): code defaults only
    # load_dotenv never overrides what is already set, so load in priority order.
    load_dotenv(ROOT / ".env")
    load_dotenv(Path(os.getenv("GHOST_CONFIG") or ROOT / "ghost.env"))
