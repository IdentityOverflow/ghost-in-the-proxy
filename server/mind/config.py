"""Mind middleware configuration (all env-gated; disabled = pure passthrough)."""
import os
from pydantic import BaseModel


class MindConfig(BaseModel):
    enabled: bool = os.getenv("MIND_ENABLED", "0") == "1"
    db_dir: str = os.getenv("MIND_DB_DIR", "var/minds")
    # Target model window. 8k is the design floor (docs/architecture.md);
    # set to the backend's actual loaded context for tighter conditions.
    window: int = int(os.getenv("MIND_WINDOW", "8192"))
    # Fraction of the window reserved for the response and tool round-trips.
    reserve_fraction: float = float(os.getenv("MIND_RESERVE_FRACTION", "0.25"))
    # Run a summarization pass when live events not covered by the running
    # summary and not selectable as texture exceed this many tokens.
    summary_trigger_tokens: int = int(os.getenv("MIND_SUMMARY_TRIGGER_TOKENS", "600"))
    # Cap for the running-summary section of the workspace.
    summary_budget_tokens: int = int(os.getenv("MIND_SUMMARY_BUDGET_TOKENS", "500"))
    # When the steward folds, fold this much PAST the required boundary so
    # the trigger doesn't re-fire every turn (the v0 summarization storm).
    fold_ahead_tokens: int = int(os.getenv("MIND_FOLD_AHEAD_TOKENS", "700"))
    # Never fold the most recent N user turns; they stay verbatim.
    min_keep_turns: int = int(os.getenv("MIND_MIN_KEEP_TURNS", "2"))
    # Model used for summarization/extraction; empty = the request's model.
    extraction_model: str | None = os.getenv("MIND_EXTRACTION_MODEL") or None
    # "open": mind errors fall back to passthrough (production posture).
    # "strict": mind errors fail the request loudly — REQUIRED for eval runs,
    # otherwise a crashed mind silently gets graded as the passthrough.
    fail_mode: str = os.getenv("MIND_FAIL_MODE", "open")


mind_config = MindConfig()
