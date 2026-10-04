"""Mind telemetry: one JSON object per line, for soak analysis.

MIND_METRICS_PATH unset = no file, zero cost. The soak eval (s14) joins
these records with the harness results to plot memory-section size, fold
outcomes and ledger churn against turn index — the instruments that tell
"memory bloat" apart from "steward loss".
"""

import json
import os
import time
from typing import Any

METRICS_PATH = os.getenv("MIND_METRICS_PATH") or None


def emit(kind: str, **fields: Any) -> None:
    if not METRICS_PATH:
        return
    record = {"kind": kind, "wall_ts": round(time.time(), 3), **fields}
    try:
        with open(METRICS_PATH, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    except OSError as error:
        print(f"[mind] metrics write failed ({error!r})", flush=True)
