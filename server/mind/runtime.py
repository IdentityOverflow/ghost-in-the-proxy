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
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Reasoning is ephemeral deliberation, not conversation truth: think blocks
# never enter the event store (and clients that strip them would otherwise
# desync reconciliation every turn).
THINK_BLOCK = re.compile(r"<think>.*?</think>\s*", flags=re.DOTALL)

from .assembler import ThreadsView, Workspace, assemble, estimate_tokens
from .config import MindConfig, mind_config
from .dynamics import ThreadState, admitted_threads, cued_threads, update_dynamics
from .perception import reconcile
from .recall import RECALL_TOOL, resolve_recall
from .router import scope_tools
from .steward import run_steward
from .store import MindStore, content_text
from .summarizer import update_summary


@dataclass
class PreparedRequest:
    session_id: str
    outcome: str
    messages: list[dict[str, Any]]
    # Tools to forward this request (router-scoped client tools + recall).
    # None means "leave the client's tools untouched".
    tools: list[dict[str, Any]] | None = None
    tools_scoped: bool = False
    # Whether the recall tool is in `tools` this request — the streaming
    # path needs to know before the first chunk whether to hold-and-decide.
    recall_offered: bool = False


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
        tools: list[dict[str, Any]] | None = None,
        now: float | None = None,
    ) -> PreparedRequest:
        """`now` is a client-supplied clock (X-Mind-Clock, fake-clock eval runs
        only); None means the real wall clock."""
        recon = reconcile(self.store, messages, now=now)
        # Chronos (v4): the time the mind renders and reasons with. None keeps
        # every time-aware surface (Now section, gap markers, due status,
        # steward timestamps) inert.
        clock = (now if now is not None else time.time()) if self.config.time_enabled else None
        events = self.store.live_events(recon.session_id)
        client_system = self.store.get_client_system(recon.session_id)
        summary = self.store.latest_summary(recon.session_id)
        records = self.store.live_records(recon.session_id)
        episodes = self.store.live_episodes(recon.session_id)

        threads = self._attention(recon.session_id, events, records)
        workspace = assemble(
            self.config, client_system, events, summary, records, episodes, threads, now=clock
        )
        uncovered = self._uncovered_tokens(events, summary, workspace)
        if uncovered > self.config.summary_trigger_tokens:
            # Fold everything the budget wants evicted — plus a fold-ahead
            # margin so the trigger doesn't re-fire every turn — into the
            # structured ledger (steward), falling back to the v0 prose
            # summarizer if the steward's proposal doesn't parse.
            upto = self._fold_boundary(events, workspace.desired_from_seq - 1)
            try:
                await run_steward(
                    self.config, self.store, recon.session_id, events, provider, model, upto,
                    now=clock,
                )
            except Exception as error:
                print(f"[mind] steward failed ({error!r}); prose fallback", flush=True)
                await update_summary(
                    self.config, self.store, recon.session_id, events, provider, model, upto
                )
            summary = self.store.latest_summary(recon.session_id)
            records = self.store.live_records(recon.session_id)
            episodes = self.store.live_episodes(recon.session_id)
            threads = self._attention(recon.session_id, events, records)
            workspace = assemble(
                self.config, client_system, events, summary, records, episodes, threads, now=clock
            )

        # v3 routing: scope the client's tool pack to this turn, and offer
        # recall once anything has folded out of verbatim view.
        last_user_text = next(
            (
                text
                for event in reversed(events)
                if event.role == "user" and (text := content_text(event.message))
            ),
            "",
        )
        scoped = tools
        tools_scoped = False
        if self.config.tool_router_enabled and tools:
            scoped = scope_tools(tools, events, last_user_text)
            tools_scoped = scoped is not tools
        out_tools = list(scoped) if scoped else []
        folded_upto = (summary or (0, ""))[0]
        recall_offered = self.config.recall_enabled and folded_upto > 0
        if recall_offered:
            out_tools = out_tools + [RECALL_TOOL]
            tools_scoped = True

        print(
            "[mind]",
            json.dumps(
                {
                    "session": recon.session_id,
                    "outcome": recon.outcome,
                    "tools": {
                        "client": len(tools or []),
                        "forwarded": len(scoped or []),
                        "recall": recall_offered,
                    },
                    "live_events": len(events),
                    "workspace_tokens_est": workspace.estimated_tokens,
                    "texture_from_seq": workspace.texture_from_seq,
                    "summary_upto": (summary or (0, ""))[0],
                    "ledger": {kind: len(items) for kind, items in records.items()},
                    "episodes": len(episodes),
                    "threads": {
                        "total": len(threads.all_keys),
                        "admitted": [
                            (thread.key, round(thread.activation, 2))
                            for thread in threads.admitted
                        ],
                        "cued": [thread.key for thread in threads.cued],
                    }
                    if threads
                    else None,
                }
            ),
            flush=True,
        )
        return PreparedRequest(
            recon.session_id,
            recon.outcome,
            workspace.messages,
            tools=out_tools if tools_scoped else None,
            tools_scoped=tools_scoped,
            recall_offered=recall_offered,
        )

    def resolve_recall(self, session_id: str, arguments_json: str) -> str:
        return resolve_recall(self.store.live_events(session_id), arguments_json)

    def observe_reply(
        self,
        session_id: str,
        message: dict[str, Any],
        complete: bool = True,
        ts: float | None = None,
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
            session_id, keep, source="mind", complete=complete, confirmed=False, ts=ts
        )

    def _attention(
        self,
        session_id: str,
        events: list,
        records: dict[str, list[dict[str, Any]]],
    ) -> ThreadsView | None:
        """CRS tick (v2): decay/boost thread activations for any user turns
        not yet applied, persist, and compute workspace admission + cues.

        Returns None when the steward has proposed no threads yet — the
        assembler then renders every fact (exactly v1 behavior).
        """
        structure = self.store.live_threads(session_id)
        if not structure:
            return None
        facts_by_thread: dict[str, list[dict[str, Any]]] = {}
        for fact in records.get("fact", []):
            facts_by_thread.setdefault(str(fact.get("thread")), []).append(fact)
        dynamics = self.store.get_dynamics(session_id)
        states: list[ThreadState] = []
        for thread in structure:
            key = thread["key"]
            state = ThreadState(
                key=key,
                kind=str(thread.get("kind", "topic")),
                summary=str(thread.get("summary", "")),
                anchors=[str(anchor) for anchor in thread.get("anchors") or []],
                open_questions=[str(q) for q in thread.get("open_questions") or []],
                facts=facts_by_thread.get(key, []),
            )
            if key in dynamics:
                state.activation, state.importance = dynamics[key][0], dynamics[key][1]
            states.append(state)

        applied_upto = max((row[2] for row in dynamics.values()), default=0)
        last_user_text = ""
        ticked_upto = applied_upto
        for event in events:
            if event.role != "user":
                continue
            text = content_text(event.message)
            if not text:
                continue
            if event.seq > applied_upto:
                update_dynamics(states, text)
                ticked_upto = event.seq
            last_user_text = text
        if ticked_upto > applied_upto:
            for state in states:
                self.store.set_dynamics(
                    session_id, state.key, state.activation, state.importance, ticked_upto
                )

        admitted = admitted_threads(states)
        cued = cued_threads(states, last_user_text, admitted)
        return ThreadsView(
            admitted=admitted, cued=cued, all_keys={state.key for state in states}
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
