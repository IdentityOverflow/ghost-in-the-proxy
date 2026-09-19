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
from .memory_view import ThreadsView, memory_budget, render_memory, render_memory_parts
from .metrics import emit
from .perception import reconcile, resolve_session
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
    # Characters of recall payload the request can still absorb, across all
    # hops, before the follow-up would overflow the window.
    recall_budget_chars: int = 0


@dataclass
class _Scene:
    session_id: str
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
        # One lock per session serializes REQUESTS (reconcile + scene). The
        # background pass does not hold it: a request waits for an in-flight
        # pass only up to maintenance_wait_s, then proceeds on committed
        # state — folds commit atomically and refuse stale or overlapping
        # spans, so racing a pass is safe; hanging behind one is not.
        self._locks: dict[str, asyncio.Lock] = {}
        self._context: dict[str, _SessionContext] = {}
        self._background: set[asyncio.Task] = set()
        self._maintaining: dict[str, asyncio.Task] = {}
        # Previous outgoing request per session, serialized: the shared prefix
        # with the next one is what a backend's KV cache could reuse.
        self._last_request: dict[str, str] = {}
        # Real tokens per estimated token, learned from usage.prompt_tokens.
        # Keyed by MODEL: it is a property of the tokenizer, not the session.
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
        # Chronos (v4): the time the mind renders and reasons with. None keeps
        # every time-aware surface inert.
        clock = (now if now is not None else time.time()) if self.config.time_enabled else None
        # Lock FIRST, reconcile under it: reconciliation mutates the store,
        # and a request that mutates and then queues behind maintenance can
        # wake up to another request's events.
        waited = time.monotonic()
        held = self._lock(resolve_session(self.store, messages) or "__new__")
        await held.acquire()
        try:
            waited = time.monotonic() - waited
            recon = reconcile(self.store, messages, now=now)
            session_id = recon.session_id
            own = self._lock(session_id)
            if own is not held:
                # New session (or the match changed while we queued). A fresh
                # lock acquires without suspending, so nothing can slip in.
                if own.locked():
                    held.release()
                    held = None
                    await own.acquire()
                    held = own
                else:
                    await own.acquire()
                    held.release()
                    held = own
            return await self._prepare_locked(recon, provider, model, tools, clock, waited)
        finally:
            if held is not None:
                held.release()

    async def _prepare_locked(
        self,
        recon: Any,
        provider: Any,
        model: str,
        tools: list[dict[str, Any]] | None,
        clock: float | None,
        waited: float,
    ) -> PreparedRequest:
        session_id = recon.session_id
        waited += await self._await_maintenance(session_id)
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

        scene = await self._scene(session_id, events, clock, out_tools, model)
        folded = False
        if self._must_fold_now(events, scene):
            # Truth is about to leave view with nothing covering it (first
            # contact with a long transcript, a restart, folding fell
            # behind): bounded synchronous catch-up, then rebuild.
            upto = self._fold_boundary(events, scene.workspace.desired_from_seq - 1)
            await self._fold(session_id, events, provider, model, upto, clock, "request")
            scene = await self._scene(session_id, events, clock, out_tools, model)
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
        serialized = json.dumps(workspace.messages, ensure_ascii=False)
        previous = self._last_request.get(session_id, "")
        shared = 0
        for left, right in zip(previous, serialized):
            if left != right:
                break
            shared += 1
        self._last_request[session_id] = serialized
        emit(
            "request",
            session=session_id,
            outcome=recon.outcome,
            request_chars=len(serialized),
            # Share of THIS request already seen as a prefix of the last one —
            # an upper bound on KV-cache reuse (0 on the first turn).
            prefix_reuse=round(shared / len(serialized), 3) if serialized else 0.0,
            live_events=len(events),
            workspace_tokens=workspace.estimated_tokens,
            memory_tokens=workspace.memory_tokens,
            system_tokens=workspace.system_tokens,
            budget_tokens=workspace.budget_tokens,
            tools_tokens=tools_tokens,
            token_scale=round(self._scale.get(model, DEFAULT_TOKEN_SCALE), 3),
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
            recall_budget_chars=self._recall_headroom(
                workspace.estimated_tokens + tools_tokens, model
            ),
        )

    async def _scene(
        self,
        session_id: str,
        events: list[Event],
        clock: float | None,
        out_tools: list[dict[str, Any]],
        model: str = "",
    ) -> _Scene:
        """Everything the model will see, built from current store state."""
        state = replay(self.store.live_folds(session_id))
        consolidations = self.store.live_consolidations(session_id)
        threads = self._attention(session_id, events, state)
        recalled = await self._autocue(session_id, events, state.covered_upto)
        # +1 tool: recall joins the belt once anything is folded.
        tools_tokens = estimate_tokens(out_tools + [RECALL_TOOL]) if (out_tools or state.covered_upto) else 0
        scale = self._scale.get(model, DEFAULT_TOKEN_SCALE)
        budget, _hard = token_budgets(self.config, tools_tokens, scale)
        render_args = dict(
            cue=_last_user_text(events),
            mem=self.mem,
            budget_tokens=memory_budget(self.config, budget),
            now=clock,
            recalled_spans=recalled,
            seq_ts={event.seq: event.ts for event in events if event.ts},
        )
        volatile_text = ""
        if self.config.memory_placement == "split":
            parts = await render_memory_parts(
                self.config, state, consolidations, threads, **render_args
            )
            memory_text, volatile_text = parts.stable, parts.volatile
        else:
            memory_text = await render_memory(
                self.config, state, consolidations, threads, **render_args
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
            volatile_text=volatile_text,
            memory_budget_tokens=render_args["budget_tokens"],
            fold_boundaries=(
                [fold["span_to"] for fold in self.store.live_folds(session_id)]
                if self.config.memory_placement == "split"
                else None
            ),
        )
        return _Scene(session_id, workspace, state, threads, len(recalled or []))

    async def _await_maintenance(self, session_id: str) -> float:
        task = self._maintaining.get(session_id)
        if task is None or task.done():
            return 0.0
        started = time.monotonic()
        await asyncio.wait({task}, timeout=self.config.maintenance_wait_s)
        waited = time.monotonic() - started
        if not task.done():
            print(
                f"[mind] maintenance still running after {waited:.0f}s; proceeding without it",
                flush=True,
            )
        return waited

    def _must_fold_now(self, events: list[Event], scene: _Scene) -> bool:
        running = self._maintaining.get(scene.session_id)
        if running is not None and not running.done():
            return False  # a pass is already folding; never run two stewards
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
        running = self._maintaining.get(session_id)
        if running is not None and not running.done():
            return  # one pass per session at a time; the next reply reschedules
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return  # sync caller (unit tests): maintenance runs on the request path
        task = loop.create_task(self._maintain(session_id))
        self._maintaining[session_id] = task
        self._background.add(task)
        task.add_done_callback(self._background.discard)

    async def _maintain(self, session_id: str) -> None:
        context = self._context[session_id]
        try:
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
                scale=self._scale.get(context.model, DEFAULT_TOKEN_SCALE),
            )
            scene = _Scene(session_id, workspace, state, None, 0)
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

    def _extraction_target(self, provider: Any, model: str) -> tuple[Any, str]:
        """Who runs steward/consolidation calls. Default: the conversation's
        own provider and model. MIND_EXTRACTION_MODEL may name another model
        on the same provider, a MODEL_MAP alias, or "provider:model" — a
        DIFFERENT backend matters on a single-slot local server (LM Studio),
        where a background fold otherwise queues ahead of the user's next
        message and evicts the conversation's KV cache (measured live: 45-85 s
        stalls)."""
        target = self.config.extraction_model
        if not target:
            return provider, model
        from ..routing.router import PROVIDERS, resolve_provider_and_model
        from ..config import settings

        name, _, rest = target.partition(":")
        if rest and name in PROVIDERS:
            return PROVIDERS[name], rest
        if target in settings.model_map:
            mapped_provider, mapped_model = resolve_provider_and_model(target)
            if mapped_provider is not None:
                return mapped_provider, mapped_model
        return provider, target

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
        provider, model = self._extraction_target(provider, model)
        try:
            outcome = await run_steward(
                self.config, self.store, session_id, events, provider, model, upto,
                now=clock, mem=self.mem,
                scale=self._scale.get(model, DEFAULT_TOKEN_SCALE),
                on_usage=lambda estimated, actual: self.note_usage(model, estimated, actual),
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

    def note_usage(self, model: str, estimated_tokens: int, prompt_tokens: int | None) -> None:
        """Calibrate the token estimate from what the backend actually counted.
        chars/4 is wrong by a model- and language-dependent factor; the
        backend knows the truth, so learn it per model (EMA)."""
        if not prompt_tokens or estimated_tokens < 200:
            return
        # No upper clamp worth the name: a measured density is a fact, and
        # clamping below it guarantees overflow (code or CJK can run 4-5x).
        observed = max(0.8, min(prompt_tokens / estimated_tokens, 8.0))
        previous = self._scale.get(model, DEFAULT_TOKEN_SCALE)
        # Err high: undercounting overflows the window, overcounting only
        # wastes a little of it.
        blended = 0.7 * previous + 0.3 * observed
        self._scale[model] = max(blended, observed * 0.97)

    def _recall_headroom(self, estimated_tokens: int, model: str) -> int:
        """Characters of recall payload this request can absorb in total."""
        scale = self._scale.get(model, DEFAULT_TOKEN_SCALE)
        reserve = max(int(self.config.window * self.config.reserve_fraction), 1024)
        # Keep at least a third of the reply reserve for the actual answer.
        headroom_real = self.config.window - reserve // 3 - int(estimated_tokens * scale)
        return max(0, int(headroom_real / scale) * 4)

    async def resolve_recall(
        self, session_id: str, arguments_json: str, char_budget: int | None = None
    ) -> str:
        events = self.store.live_events(session_id)
        # The trajectory stub: recent verbatim context as the episodic cue
        # (single moments are ambiguous — holographic experiment #5).
        return await resolve_recall(
            events,
            arguments_json,
            backend=self.mem,
            session_id=session_id,
            trajectory=events[-8:],
            char_budget=min(self.recall_char_budget(), char_budget)
            if char_budget is not None
            else self.recall_char_budget(),
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
        folded = [span for span in spans if span.seq <= folded_upto and span.sim > 0]
        if not folded:
            return None
        # The fixed cosine floor was calibrated on a 15-turn scenario; in a
        # long conversation SOMETHING always clears it (the soak auto-cued on
        # 133 of 160 turns). A span must also stand out from this query's own
        # background: mean + 1.5 sd over the folded history.
        sims = [span.sim for span in folded]
        mean = sum(sims) / len(sims)
        spread = (sum((sim - mean) ** 2 for sim in sims) / len(sims)) ** 0.5
        floor = max(self.config.embed_min_sim, mean + 1.5 * spread)
        # Rank by the SEMANTIC component (the hybrid score's lexical half
        # favours long assistant replies sharing common words), and prefer
        # what the user said: that is where one-off details live.
        rescued = sorted(
            (span for span in folded if span.sim >= floor),
            key=lambda span: span.sim + (0.03 if span.role == "user" else 0.0),
            reverse=True,
        )
        return rescued[:3] or None

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
