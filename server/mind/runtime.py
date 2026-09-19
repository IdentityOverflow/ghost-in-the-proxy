"""Mind runtime: per-request orchestration, fail-mode policy, reply observation.

Flow per request (docs/architecture.md, docs/memory-v6.md):
  reconcile (session + new events) -> wait for in-flight maintenance ->
  build the scene (ledger replay, attention, bounded memory, texture) ->
  [only if truth would otherwise leave view uncovered: fold now] -> forward.
After the provider responds, the reply is recorded as a provisional event and
a background maintenance pass folds and consolidates — the idle loop. The
next request's reconciliation confirms what the client retained.

MIND_FAIL_MODE=open   -> any mind error falls back to passthrough (production)
MIND_FAIL_MODE=strict -> mind errors raise; eval runs MUST use strict, or a
                         crashed mind gets silently graded as the passthrough.
"""

import asyncio
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

from .assembler import DEFAULT_TOKEN_SCALE, Workspace, assemble, estimate_tokens, token_budgets
from .config import MindConfig, mind_config
from .consolidate import consolidate_once
from .dynamics import ThreadState, admitted_threads, cued_threads, update_dynamics
from .ledger import LedgerState, replay
from .mem import MemQuery, create_mem_backend
from .memory_view import ThreadsView, memory_budget, render_memory
from .metrics import emit
from .perception import reconcile
from .recall import RECALL_TOOL, resolve_recall
from .router import scope_tools
from .steward import FoldOutcome, run_steward
from .store import Event, MindStore, content_text


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
    # What the mind estimated for the whole request (workspace + tool
    # schemas) — paired with the backend's usage to calibrate the estimate.
    estimated_tokens: int = 0


@dataclass
class _Scene:
    workspace: Workspace
    state: LedgerState
    threads: ThreadsView | None
    autocued: int


@dataclass
class _SessionContext:
    """What the background pass needs from the request that preceded it."""

    provider: Any
    model: str
    clock: float | None
    tools_tokens: int
    memory_text: str


class MindRuntime:
    def __init__(self, config: MindConfig):
        self.config = config
        db_path = Path(config.db_dir) / "minds.sqlite3"
        self.store = MindStore(db_path)
        self.mem = create_mem_backend(config.mem_backend, config=config, store=self.store)
        # Per-session observe watermark: events enter Mem at the same moment
        # reconciliation confirms them as conversation truth. After a fork,
        # the rewritten tail gets fresh seqs above the watermark, so it is
        # re-observed.
        self._mem_seen: dict[str, int] = {}
        # One lock per session serializes scene-building with maintenance:
        # a request waits for an in-flight fold instead of racing it.
        self._locks: dict[str, asyncio.Lock] = {}
        self._context: dict[str, _SessionContext] = {}
        self._background: set[asyncio.Task] = set()
        # Real tokens per estimated token, learned from usage.prompt_tokens.
        self._scale: dict[str, float] = {}

    def _lock(self, session_id: str) -> asyncio.Lock:
        return self._locks.setdefault(session_id, asyncio.Lock())

    # -- request path -----------------------------------------------------------

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
        session_id = recon.session_id
        # Chronos (v4): the time the mind renders and reasons with. None keeps
        # every time-aware surface inert.
        clock = (now if now is not None else time.time()) if self.config.time_enabled else None
        waited = time.monotonic()
        async with self._lock(session_id):
            waited = time.monotonic() - waited
            events = self.store.live_events(session_id)
            seen = self._mem_seen.get(session_id, 0)
            for event in events:
                if event.seq > seen:
                    await self.mem.observe(session_id, event)
            if events:
                self._mem_seen[session_id] = events[-1].seq

            # v3 routing first: tool schemas ride in the same request, so
            # their cost has to be known before the workspace is budgeted.
            last_user_text = _last_user_text(events)
            scoped = tools
            tools_scoped = False
            if self.config.tool_router_enabled and tools:
                scoped = scope_tools(tools, events, last_user_text)
                tools_scoped = scoped is not tools
            out_tools = list(scoped) if scoped else []

            scene = await self._scene(session_id, events, clock, out_tools)
            folded = False
            if self._must_fold_now(events, scene):
                # Truth is about to leave view with nothing covering it (first
                # contact with a long transcript, a restart, folding fell
                # behind): bounded synchronous catch-up, then rebuild.
                upto = self._fold_boundary(events, scene.workspace.desired_from_seq - 1)
                await self._fold(session_id, events, provider, model, upto, clock, "request")
                scene = await self._scene(session_id, events, clock, out_tools)
                folded = True

            # Offer recall once anything has folded out of verbatim view.
            recall_offered = self.config.recall_enabled and scene.state.covered_upto > 0
            if recall_offered:
                out_tools = out_tools + [RECALL_TOOL]
                tools_scoped = True
            tools_tokens = estimate_tokens(out_tools) if out_tools else 0
            self._context[session_id] = _SessionContext(
                provider, model, clock, tools_tokens, _memory_text(scene.workspace)
            )

        workspace, state, threads = scene.workspace, scene.state, scene.threads
        ledger_counts = {kind: len(items) for kind, items in state.grouped().items()}
        print(
            "[mind]",
            json.dumps(
                {
                    "session": session_id,
                    "outcome": recon.outcome,
                    "tools": {
                        "client": len(tools or []),
                        "forwarded": len(scoped or []),
                        "recall": recall_offered,
                    },
                    "live_events": len(events),
                    "workspace_tokens_est": workspace.estimated_tokens,
                    "memory_tokens_est": workspace.memory_tokens,
                    "texture_from_seq": workspace.texture_from_seq,
                    "covered_upto": state.covered_upto,
                    "ledger": ledger_counts,
                    "episodes": len(state.episodes),
                    "threads": {
                        "total": len(threads.all_keys),
                        "admitted": [
                            (thread.name, round(thread.activation, 2))
                            for thread in threads.admitted
                        ],
                        "cued": [thread.name for thread in threads.cued],
                    }
                    if threads
                    else None,
                }
            ),
            flush=True,
        )
        emit(
            "request",
            session=session_id,
            outcome=recon.outcome,
            live_events=len(events),
            workspace_tokens=workspace.estimated_tokens,
            memory_tokens=workspace.memory_tokens,
            system_tokens=workspace.system_tokens,
            budget_tokens=workspace.budget_tokens,
            tools_tokens=tools_tokens,
            token_scale=round(self._scale.get(session_id, DEFAULT_TOKEN_SCALE), 3),
            texture_messages=len(workspace.messages) - (1 if workspace.system_tokens else 0),
            texture_from_seq=workspace.texture_from_seq,
            summary_upto=state.covered_upto,
            evicted_uncovered=workspace.evicted_uncovered,
            ledger=ledger_counts,
            episodes=len(state.episodes),
            consolidations=len(self.store.live_consolidations(session_id)),
            autocue=scene.autocued,
            folded_on_request=folded,
            waited_for_maintenance_s=round(waited, 2),
        )
        return PreparedRequest(
            session_id,
            recon.outcome,
            workspace.messages,
            tools=out_tools if tools_scoped else None,
            tools_scoped=tools_scoped,
            recall_offered=recall_offered,
            estimated_tokens=workspace.estimated_tokens + tools_tokens,
        )

    async def _scene(
        self,
        session_id: str,
        events: list[Event],
        clock: float | None,
        out_tools: list[dict[str, Any]],
    ) -> _Scene:
        """Everything the model will see, built from current store state."""
        state = replay(self.store.live_folds(session_id))
        consolidations = self.store.live_consolidations(session_id)
        threads = self._attention(session_id, events, state)
        recalled = await self._autocue(session_id, events, state.covered_upto)
        # +1 tool: recall joins the belt once anything is folded.
        tools_tokens = estimate_tokens(out_tools + [RECALL_TOOL]) if (out_tools or state.covered_upto) else 0
        scale = self._scale.get(session_id, DEFAULT_TOKEN_SCALE)
        budget, _hard = token_budgets(self.config, tools_tokens, scale)
        memory_text = await render_memory(
            self.config,
            state,
            consolidations,
            threads,
            cue=_last_user_text(events),
            mem=self.mem,
            budget_tokens=memory_budget(self.config, budget),
            now=clock,
            recalled_spans=recalled,
            seq_ts={event.seq: event.ts for event in events if event.ts},
        )
        workspace = assemble(
            self.config,
            self.store.get_client_system(session_id),
            events,
            covered_upto=state.covered_upto,
            memory_text=memory_text,
            now=clock,
            tools_tokens=tools_tokens,
            scale=scale,
        )
        return _Scene(workspace, state, threads, len(recalled or []))

    def _must_fold_now(self, events: list[Event], scene: _Scene) -> bool:
        if scene.workspace.evicted_uncovered > 0:
            return True
        if self.config.background_fold:
            return False
        return self._uncovered_tokens(events, scene) > self.config.summary_trigger_tokens

    # -- maintenance (the idle loop) ----------------------------------------------

    def schedule_maintenance(self, session_id: str) -> None:
        """Fold and consolidate after the reply went out. Fire-and-forget: the
        session lock makes the next request wait for it, nothing else does."""
        if not self.config.background_fold or session_id not in self._context:
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return  # sync caller (unit tests): maintenance runs on the request path
        task = loop.create_task(self._maintain(session_id))
        self._background.add(task)
        task.add_done_callback(self._background.discard)

    async def _maintain(self, session_id: str) -> None:
        context = self._context[session_id]
        try:
            async with self._lock(session_id):
                events = self.store.live_events(session_id)
                state = replay(self.store.live_folds(session_id))
                workspace = assemble(
                    self.config,
                    self.store.get_client_system(session_id),
                    events,
                    covered_upto=state.covered_upto,
                    memory_text=context.memory_text,
                    now=context.clock,
                    tools_tokens=context.tools_tokens,
                    scale=self._scale.get(session_id, DEFAULT_TOKEN_SCALE),
                )
                scene = _Scene(workspace, state, None, 0)
                if self._uncovered_tokens(events, scene) > self.config.summary_trigger_tokens:
                    upto = self._fold_boundary(events, workspace.desired_from_seq - 1)
                    await self._fold(
                        session_id, events, context.provider, context.model, upto,
                        context.clock, "background",
                    )
                started = time.monotonic()
                level = await consolidate_once(
                    self.config, self.store, session_id, context.provider, context.model
                )
                if level:
                    emit(
                        "consolidation", session=session_id, level=level,
                        seconds=round(time.monotonic() - started, 2),
                    )
        except Exception as error:
            # Maintenance is best-effort by design: the request path catches
            # up synchronously if it ever has to.
            print(f"[mind] maintenance failed ({error!r})", flush=True)
            emit("maintenance_error", session=session_id, error=repr(error)[:300])

    async def drain(self) -> None:
        """Wait for background maintenance (tests, graceful shutdown)."""
        while self._background:
            await asyncio.gather(*list(self._background), return_exceptions=True)

    async def _fold(
        self,
        session_id: str,
        events: list[Event],
        provider: Any,
        model: str,
        upto: int,
        clock: float | None,
        path: str,
    ) -> None:
        started = time.monotonic()
        before = replay(self.store.live_folds(session_id))
        outcome = FoldOutcome()
        error_text = ""
        try:
            outcome = await run_steward(
                self.config, self.store, session_id, events, provider, model, upto,
                now=clock, mem=self.mem,
            )
        except Exception as error:
            error_text = repr(error)[:300]
            print(f"[mind] fold failed ({error!r}); span stays uncovered", flush=True)
        after = replay(self.store.live_folds(session_id))
        if after.covered_upto > before.covered_upto:
            # A fold is an episode boundary: everything up to the watermark
            # left verbatim view.
            await self.mem.boundary(session_id, "fold", after.covered_upto)
        emit(
            "fold",
            session=session_id,
            path=path,
            upto_seq=upto,
            covered_upto=after.covered_upto,
            ok=not error_text and not outcome.prose_fallbacks and not outcome.stale,
            error=error_text or "; ".join(outcome.errors)[:300],
            folds=outcome.folds,
            prose_fallbacks=outcome.prose_fallbacks,
            stale=outcome.stale,
            ops_applied=outcome.ops_applied,
            ops_dropped=outcome.ops_dropped[:20],
            deduped=outcome.deduped,
            seconds=round(time.monotonic() - started, 2),
            ledger_before=len(before.records),
            ledger_after=len(after.records),
        )

    # -- reply observation -----------------------------------------------------------

    def observe_reply(
        self,
        session_id: str,
        message: dict[str, Any],
        complete: bool = True,
        ts: float | None = None,
    ) -> None:
        """Record our own reply as a provisional event (confirmed next request),
        then let the idle loop run."""
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
        self.schedule_maintenance(session_id)

    def note_usage(self, session_id: str, estimated_tokens: int, prompt_tokens: int | None) -> None:
        """Calibrate the token estimate from what the backend actually counted.
        chars/4 is wrong by a model- and language-dependent factor; the
        backend knows the truth, so learn it (EMA, clamped to sane bounds)."""
        if not prompt_tokens or estimated_tokens < 200:
            return
        observed = max(1.0, min(prompt_tokens / estimated_tokens, 3.0))
        previous = self._scale.get(session_id, DEFAULT_TOKEN_SCALE)
        # Err high: undercounting overflows the window, overcounting only
        # wastes a little of it.
        blended = 0.7 * previous + 0.3 * observed
        self._scale[session_id] = max(blended, observed * 0.97)

    async def resolve_recall(self, session_id: str, arguments_json: str) -> str:
        events = self.store.live_events(session_id)
        # The trajectory stub: recent verbatim context as the episodic cue
        # (single moments are ambiguous — holographic experiment #5).
        return await resolve_recall(
            events,
            arguments_json,
            backend=self.mem,
            session_id=session_id,
            trajectory=events[-8:],
            char_budget=self.recall_char_budget(),
        )

    def recall_char_budget(self) -> int:
        """Per-hop recall payload: ~8% of the window, within [1200, 12000] chars."""
        return max(1200, min(int(self.config.window * 0.08) * 4, 12000))

    # -- organs ------------------------------------------------------------------------

    async def _autocue(self, session_id: str, events: list[Event], folded_upto: int) -> list | None:
        """Per-turn semantic rescue (s13: models do not reliably CALL recall).

        Only for backends that opt in, only over FOLDED material (verbatim
        texture needs no rescuing), and only spans with a real semantic
        component — lexical reach already has the cue and recall channels.
        """
        if not getattr(self.mem, "autocue", False) or folded_upto <= 0:
            return None
        last_user = _last_user_text(events)
        if not last_user:
            return None
        # Rank over the whole history, THEN filter to folded: a top-k taken
        # first is won by the recent verbatim turns (which resemble the
        # question most) and the folded spans this exists for never surface.
        spans = await self.mem.query(
            session_id,
            MemQuery(text=last_user, trajectory=events[-8:], k=max(len(events), 4)),
            events,
        )
        rescued = [
            span
            for span in spans
            if span.seq <= folded_upto and span.sim >= self.config.embed_min_sim
        ]
        return rescued[:2] or None

    def _attention(
        self, session_id: str, events: list[Event], state: LedgerState
    ) -> ThreadsView | None:
        """CRS tick (v2): decay/boost thread activations for any user turns
        not yet applied, persist, and compute workspace admission + cues.

        Returns None when the steward has proposed no threads yet — the
        renderer then treats every fact as loose (v1 behavior, under budget).
        """
        if not state.threads:
            return None
        facts_by_thread: dict[str, list[dict[str, Any]]] = {}
        for record in state.by_kind("fact"):
            facts_by_thread.setdefault(str(record.data.get("thread")), []).append(record.data)
        dynamics = self.store.get_dynamics(session_id)
        states: list[ThreadState] = []
        for thread in state.threads.values():
            thread_state = ThreadState(
                key=thread.id,
                name=thread.name,
                kind=str(thread.data.get("kind", "topic")),
                summary=str(thread.data.get("summary", "")),
                anchors=[str(anchor) for anchor in thread.data.get("anchors") or []],
                open_questions=[str(q) for q in thread.data.get("open_questions") or []],
                facts=facts_by_thread.get(thread.id, []),
            )
            if thread.id in dynamics:
                thread_state.activation, thread_state.importance = dynamics[thread.id][:2]
            states.append(thread_state)

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
            for thread_state in states:
                self.store.set_dynamics(
                    session_id, thread_state.key, thread_state.activation,
                    thread_state.importance, ticked_upto,
                )

        admitted = admitted_threads(states)
        cued = cued_threads(states, last_user_text, admitted)
        return ThreadsView(
            admitted=admitted, cued=cued, all_keys={thread_state.key for thread_state in states}
        )

    def _fold_boundary(self, events: list[Event], base_upto: int) -> int:
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

    def _uncovered_tokens(self, events: list[Event], scene: _Scene) -> int:
        """Eviction pressure: tokens the BUDGET wants out but no fold covers."""
        return sum(
            estimate_tokens(event.message)
            for event in events
            if scene.state.covered_upto < event.seq < scene.workspace.desired_from_seq
        )


def _last_user_text(events: list[Event]) -> str:
    return next(
        (
            text
            for event in reversed(events)
            if event.role == "user" and (text := content_text(event.message))
        ),
        "",
    )


def _memory_text(workspace: Workspace) -> str:
    """Stand-in for the memory section when maintenance re-measures pressure:
    same size as what the request just rendered, without re-rendering it."""
    return "x" * (workspace.memory_tokens * 4)


_runtime: MindRuntime | None = None


def get_mind_runtime() -> MindRuntime | None:
    """Singleton accessor; None when the mind is disabled (pure passthrough)."""
    global _runtime
    if not mind_config.enabled:
        return None
    if _runtime is None:
        _runtime = MindRuntime(mind_config)
    return _runtime
