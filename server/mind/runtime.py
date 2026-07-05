"""Mind runtime: per-request orchestration, fail-mode policy, reply observation.

Flow per request (docs/architecture.md):
  reconcile (session + new events) -> maybe summarize -> assemble workspace.
After the provider responds, the reply is recorded as a provisional event;
the next request's reconciliation confirms what the client retained.

MIND_FAIL_MODE=open   -> any mind error falls back to passthrough (production)
MIND_FAIL_MODE=strict -> mind errors raise; eval runs MUST use strict, or a
                         crashed mind gets silently graded as the passthrough.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .assembler import Workspace, assemble, estimate_tokens
from .config import MindConfig, mind_config
from .perception import reconcile
from .store import MindStore
from .summarizer import update_summary


@dataclass
class PreparedRequest:
    session_id: str
    outcome: str
    messages: list[dict[str, Any]]


class MindRuntime:
    def __init__(self, config: MindConfig):
        self.config = config
        db_path = Path(config.db_dir) / "minds.sqlite3"
        self.store = MindStore(db_path)

    async def prepare(
        self,
        messages: list[dict[str, Any]],
        provider: Any,
        model: str,
    ) -> PreparedRequest:
        recon = reconcile(self.store, messages)
        events = self.store.live_events(recon.session_id)
        client_system = self.store.get_client_system(recon.session_id)
        summary = self.store.latest_summary(recon.session_id)

        workspace = assemble(self.config, client_system, events, summary)
        uncovered = self._uncovered_tokens(events, summary, workspace)
        if uncovered > self.config.summary_trigger_tokens:
            # Fold everything the budget wants evicted into the summary,
            # then assemble again with the smaller, covered history.
            upto = workspace.desired_from_seq - 1
            await update_summary(
                self.config, self.store, recon.session_id, events, provider, model, upto
            )
            summary = self.store.latest_summary(recon.session_id)
            workspace = assemble(self.config, client_system, events, summary)

        print(
            "[mind]",
            json.dumps(
                {
                    "session": recon.session_id,
                    "outcome": recon.outcome,
                    "live_events": len(events),
                    "workspace_tokens_est": workspace.estimated_tokens,
                    "texture_from_seq": workspace.texture_from_seq,
                    "summary_upto": (summary or (0, ""))[0],
                }
            ),
            flush=True,
        )
        return PreparedRequest(recon.session_id, recon.outcome, workspace.messages)

    def observe_reply(
        self, session_id: str, message: dict[str, Any], complete: bool = True
    ) -> None:
        """Record our own reply as a provisional event (confirmed next request)."""
        keep = {
            key: value
            for key, value in message.items()
            if key in ("role", "content", "tool_calls") and value is not None
        }
        self.store.append_event(
            session_id, keep, source="mind", complete=complete, confirmed=False
        )

    def _uncovered_tokens(self, events, summary, workspace: Workspace) -> int:
        """Eviction pressure: tokens the BUDGET wants out but no summary covers."""
        summary_upto = (summary or (0, ""))[0]
        return sum(
            estimate_tokens(event.message)
            for event in events
            if summary_upto < event.seq < workspace.desired_from_seq
        )


_runtime: MindRuntime | None = None


def get_mind_runtime() -> MindRuntime | None:
    """Singleton accessor; None when the mind is disabled (pure passthrough)."""
    global _runtime
    if not mind_config.enabled:
        return None
    if _runtime is None:
        _runtime = MindRuntime(mind_config)
    return _runtime
