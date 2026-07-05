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
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Reasoning is ephemeral deliberation, not conversation truth: think blocks
# never enter the event store (and clients that strip them would otherwise
# desync reconciliation every turn).
THINK_BLOCK = re.compile(r"<think>.*?</think>\s*", flags=re.DOTALL)

from .assembler import Workspace, assemble, estimate_tokens
from .config import MindConfig, mind_config
from .perception import reconcile
from .steward import run_steward
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
        records = self.store.live_records(recon.session_id)
        episodes = self.store.live_episodes(recon.session_id)

        workspace = assemble(self.config, client_system, events, summary, records, episodes)
        uncovered = self._uncovered_tokens(events, summary, workspace)
        if uncovered > self.config.summary_trigger_tokens:
            # Fold everything the budget wants evicted — plus a fold-ahead
            # margin so the trigger doesn't re-fire every turn — into the
            # structured ledger (steward), falling back to the v0 prose
            # summarizer if the steward's proposal doesn't parse.
            upto = self._fold_boundary(events, workspace.desired_from_seq - 1)
            try:
                await run_steward(
                    self.config, self.store, recon.session_id, events, provider, model, upto
                )
            except Exception as error:
                print(f"[mind] steward failed ({error!r}); prose fallback", flush=True)
                await update_summary(
                    self.config, self.store, recon.session_id, events, provider, model, upto
                )
            summary = self.store.latest_summary(recon.session_id)
            records = self.store.live_records(recon.session_id)
            episodes = self.store.live_episodes(recon.session_id)
            workspace = assemble(self.config, client_system, events, summary, records, episodes)

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
                    "ledger": {kind: len(items) for kind, items in records.items()},
                    "episodes": len(episodes),
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
        if isinstance(keep.get("content"), str):
            keep["content"] = THINK_BLOCK.sub("", keep["content"])
        self.store.append_event(
            session_id, keep, source="mind", complete=complete, confirmed=False
        )

    def _fold_boundary(self, events, base_upto: int) -> int:
        """Extend the fold boundary past what the budget strictly requires.

        Walks forward from base_upto accumulating fold_ahead_tokens, landing
        only on turn boundaries (an assistant event followed by a user event)
        and never eating the last min_keep_turns user turns.
        """
        user_seqs = [event.seq for event in events if event.role == "user"]
        keep_from = user_seqs[-self.config.min_keep_turns] if len(user_seqs) >= self.config.min_keep_turns else 0
        extended = base_upto
        accumulated = 0
        for index, event in enumerate(events):
            if event.seq <= base_upto:
                continue
            if event.seq >= keep_from:
                break
            accumulated += estimate_tokens(event.message)
            next_role = events[index + 1].role if index + 1 < len(events) else None
            at_turn_boundary = event.role != "user" and next_role == "user"
            if at_turn_boundary:
                extended = event.seq
                if accumulated >= self.config.fold_ahead_tokens:
                    break
        return extended

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
