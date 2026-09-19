"""Adversarial v6 regressions: exercise the original failure paths, not mocks of fixes."""

import asyncio
import json
from types import SimpleNamespace

import httpx
import pytest

from server.endpoints import chat
from server.mind import assembler, consolidate, dynamics, steward
from server.mind.config import MindConfig
from server.mind.ledger import LedgerState, Record, apply_ops
from server.mind.memory_view import render_memory
from server.mind.perception import _anchor, reconcile
from server.mind.runtime import MindRuntime, PreparedRequest
from server.schemas import ChatCompletionRequest


def message(role, content):
    return {"role": role, "content": content}


def config(**overrides):
    return MindConfig(**{
        "enabled": True, "window": 4096, "mem_backend": "lexical",
        "time_enabled": False, "background_fold": True,
        "extraction_model": None, "steward_json_schema": True,
        "steward_input_tokens": 2600, "extraction_max_tokens": 4000,
        **overrides,
    })


@pytest.fixture
def runtime(tmp_path):
    value = MindRuntime(config(db_dir=str(tmp_path)))
    yield value
    value.store._conn.close()


class Provider:
    name = "v6-regression"

    def __init__(self):
        self.calls = []

    async def chat_completions(self, payload):
        self.calls.append(payload)
        estimated = sum(assembler.estimate_tokens(m["content"]) for m in payload["messages"])
        return {
            "choices": [{"message": message("assistant", '{"ops": [], "episode": "A note."}')}],
            "usage": {"prompt_tokens": estimated * 5},
        }


def endpoint(monkeypatch, mind, provider):
    monkeypatch.setattr(chat, "get_mind_runtime", lambda: mind)
    monkeypatch.setattr(chat, "resolve_provider_and_model", lambda _: (provider, None))
    monkeypatch.setattr(chat, "agent_graph", SimpleNamespace(invoke=lambda state: state))
    monkeypatch.setattr(chat.mind_config, "fake_clock", False)


class Request:
    headers = {}

    async def is_disconnected(self):
        return False


def test_queued_fork_does_not_replace_first_question(runtime):
    async def scenario():
        prefix = [message("user", "opening"), message("assistant", "acknowledged")]
        sid = reconcile(runtime.store, prefix).session_id
        lock = runtime._lock(sid)
        await lock.acquire()  # maintenance owns the session
        tasks = []
        try:
            tasks.append(asyncio.create_task(runtime.prepare(prefix + [message("user", "QUESTION A")], Provider(), "m")))
            await asyncio.sleep(0)
            tasks.append(asyncio.create_task(runtime.prepare(prefix + [message("user", "QUESTION B")], Provider(), "m")))
            await asyncio.sleep(0)
            assert all(not task.done() for task in tasks)
        finally:
            lock.release()
        a, b = await asyncio.wait_for(asyncio.gather(*tasks), 2)
        assert a.session_id == b.session_id == sid
        assert a.messages[-1] == message("user", "QUESTION A")
        assert b.messages[-1] == message("user", "QUESTION B")
        assert b.outcome == "fork"
    asyncio.run(scenario())


def test_new_sessions_and_cancelled_new_waiter_release_locks(runtime):
    async def scenario():
        lock = runtime._lock("__new__")
        await lock.acquire()
        cancelled = asyncio.create_task(runtime.prepare([message("user", "cancelled")], Provider(), "m"))
        await asyncio.sleep(0)
        cancelled.cancel()
        with pytest.raises(asyncio.CancelledError):
            await cancelled
        assert lock.locked()  # cancellation must not release somebody else's lock
        tasks = [asyncio.create_task(runtime.prepare([message("user", text)], Provider(), "m"))
                 for text in ("brand new A", "brand new B")]
        await asyncio.sleep(0)
        lock.release()
        a, b = await asyncio.wait_for(asyncio.gather(*tasks), 2)
        assert a.session_id != b.session_id
        assert not any(lock.locked() for lock in runtime._locks.values())
    asyncio.run(scenario())


def test_cancelled_existing_session_waiter_does_not_hold_lock(runtime):
    async def scenario():
        transcript = [message("user", "existing")]
        sid = reconcile(runtime.store, transcript).session_id
        lock = runtime._lock(sid)
        await lock.acquire()
        task = asyncio.create_task(runtime.prepare(transcript, Provider(), "m"))
        await asyncio.sleep(0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert lock.locked()
        lock.release()
        await asyncio.wait_for(runtime.prepare(transcript, Provider(), "m"), 2)
        assert not lock.locked()
    asyncio.run(scenario())


@pytest.mark.parametrize("level", [1, 2])
def test_fork_during_condense_rejects_stale_commit(runtime, monkeypatch, level):
    async def scenario():
        store = runtime.store
        sid = reconcile(store, [message("user" if i % 2 == 0 else "assistant", f"turn {i}")
                                for i in range(8)]).session_id
        for seq in range(1, 9):
            assert store.append_fold(sid, seq, seq, [], f"leaf {seq}")
        if level == 2:
            for first in (1, 3, 5, 7):
                assert store.append_consolidation(sid, 1, first, first + 1, first, first + 1, "era", 2)
        entered, resume = asyncio.Event(), asyncio.Event()

        async def condense(*args):
            entered.set()
            await resume.wait()
            return "STALE BRANCH"

        results = []
        original = store.append_consolidation

        def append(*args, **kwargs):
            result = original(*args, **kwargs)
            results.append(result)
            return result

        monkeypatch.setattr(consolidate, "_condense", condense)
        monkeypatch.setattr(store, "append_consolidation", append)
        task = asyncio.create_task(consolidate.consolidate_once(config(era_size=2), store, sid, Provider(), "m"))
        await asyncio.wait_for(entered.wait(), 2)
        # Fork strictly inside the summary's event span, after its first child.
        replacement = store.append_event(sid, message("assistant", "replacement"), "client")
        store.supersede_from(sid, 2, replacement)
        before = store.live_consolidations(sid)
        resume.set()
        assert await asyncio.wait_for(task, 2) is None
        assert results == [False]
        assert store.live_consolidations(sid) == before
        assert all(row["content"] != "STALE BRANCH" for row in before)
    asyncio.run(scenario())


def test_unicode_and_empty_keys_do_not_merge():
    state = LedgerState()
    report = apply_ops(state, [
        {"op": "add", "kind": "fact", "subject": subject, "claim": claim}
        for subject, claim in [("姓名", "李雷"), ("过敏", "花生"), ("!!!", "one"), ("???", "two")]
    ], 1, 2)
    assert not report.dropped
    assert report.deduped == 0
    assert [(r.data["subject"], r.data["claim"]) for r in state.records.values()] == [
        ("姓名", "李雷"), ("过敏", "花生"), ("!!!", "one"), ("???", "two")]


@pytest.mark.parametrize("text", [
    "姓名: 李雷 过敏: 花生",
    "花生过敏", "аллергия арахис",
])
def test_non_english_tokenization(text):
    assert dynamics.tokenize(text)


def test_giant_system_prompt_raises(runtime):
    sid = reconcile(runtime.store, [message("user", "hi")]).session_id
    with pytest.raises(assembler.WorkspaceOverflow):
        assembler.assemble(config(), "system " * 10000, runtime.store.live_events(sid))


@pytest.mark.parametrize("fail_mode", ["open", "strict"])
def test_endpoint_overflow_is_400_in_both_fail_modes(runtime, monkeypatch, fail_mode):
    provider = Provider()
    endpoint(monkeypatch, runtime, provider)
    monkeypatch.setattr(chat.mind_config, "fail_mode", fail_mode)
    response = asyncio.run(chat.chat_completions(ChatCompletionRequest(
        model="m", messages=[message("system", "system " * 10000), message("user", "hi")],
    ), Request()))
    assert response.status_code == 400
    assert json.loads(response.body)["error"]["code"] == "context_length_exceeded"
    assert provider.calls == []


def test_giant_tool_arguments_raise(runtime):
    transcript = [message("user", "run this"), {
        "role": "assistant", "tool_calls": [{"id": "c", "type": "function", "function": {
            "name": "execute", "arguments": json.dumps({"code": "x" * 60000})}}]},
        {"role": "tool", "tool_call_id": "c", "content": "done"}]
    sid = reconcile(runtime.store, transcript).session_id
    with pytest.raises(assembler.WorkspaceOverflow):
        assembler.assemble(config(), None, runtime.store.live_events(sid))


def test_dense_estimate_and_model_specific_calibration(runtime):
    assert assembler.estimate_tokens("汉" * 8000) >= 6000
    runtime.note_usage("dense", 1000, 5000)
    assert runtime._scale["dense"] >= 4.8
    dense_scale = runtime._scale["dense"]
    runtime.note_usage("sparse", 1000, 1000)
    assert runtime._scale["dense"] == dense_scale
    assert runtime._scale["sparse"] < 1.5


def test_steward_usage_calibrates_runtime_via_callback(runtime):
    sid = reconcile(runtime.store, [message("user", "remember this")]).session_id
    provider = Provider()
    asyncio.run(runtime._fold(sid, runtime.store.live_events(sid), provider, "steward-model", 1, None, "request"))
    assert runtime.store.live_folds(sid)
    assert runtime._scale["steward-model"] >= 4.8


def test_prepared_recall_headroom_shrinks_with_workspace(runtime):
    async def scenario():
        small = await runtime.prepare([message("user", "small")], Provider(), "m")
        full = await runtime.prepare([message("system", "x" * 6000), message("user", "full")], Provider(), "m")
        assert full.estimated_tokens > small.estimated_tokens
        assert 0 < full.recall_budget_chars < small.recall_budget_chars
    asyncio.run(scenario())


def test_recall_low_headroom_returns_notice():
    result = asyncio.run(chat._recall(None, "s", "{}", 200))
    assert "no room left" in result
    assert len(result) <= 200


@pytest.mark.parametrize("chars_left", [0, 1, 20])
def test_recall_notice_is_small_and_constant_when_nothing_remains(chars_left):
    # A tool call must get SOME result; the notice is a fixed ~20 tokens, paid
    # from the third of the reply reserve _recall_headroom never hands out.
    result = asyncio.run(chat._recall(None, "s", "{}", chars_left))
    assert "no room left" in result and len(result) <= 120


def test_recall_payload_fits_remaining_headroom(runtime):
    sid = reconcile(runtime.store, [message("user", "needle " * 2000)]).session_id
    result = asyncio.run(chat._recall(runtime, sid, '{"query":"needle"}', 900))
    assert "needle" in result
    assert len(result) + chat.RECALL_HOP_OVERHEAD_CHARS <= 900


@pytest.mark.parametrize("stream", [False, True])
def test_recall_cumulative_accounting_includes_exchange_overhead(monkeypatch, stream):
    budgets = []
    call = {"id": "r", "type": "function", "function": {"name": "recall", "arguments": "{}"}}

    class Mind:
        async def prepare(self, *args, **kwargs):
            return PreparedRequest("s", "continue", [message("user", "question")],
                                   recall_offered=True, recall_budget_chars=2000)

        def note_usage(self, *args):
            pass

        def observe_reply(self, *args, **kwargs):
            pass

    class RecallProvider:
        count = 0

        def reply(self):
            self.count += 1
            return {"role": "assistant", "tool_calls": [call]} if self.count <= 2 else message("assistant", "answer")

        async def chat_completions(self, payload):
            return {"choices": [{"message": self.reply()}]}

        async def chat_completions_stream(self, payload):
            delta = self.reply()
            if "tool_calls" in delta:
                delta["tool_calls"] = [{"index": 0, **call}]
            yield ("data: " + json.dumps({"choices": [{"delta": delta}]}) + "\n\n").encode()
            yield b"data: [DONE]\n\n"

    async def recall(mind, sid, arguments, chars_left):
        budgets.append(chars_left)
        return "x" * 100

    endpoint(monkeypatch, Mind(), RecallProvider())
    monkeypatch.setattr(chat, "_recall", recall)
    monkeypatch.setattr(chat.mind_config, "recall_enabled", True)
    monkeypatch.setattr(chat.mind_config, "recall_max_hops", 3)

    async def scenario():
        response = await chat.chat_completions(ChatCompletionRequest(
            model="m", messages=[message("user", "question")], stream=stream), Request())
        if stream:
            async for _ in response.body_iterator:
                pass
    asyncio.run(scenario())
    assert budgets == [2000, 2000 - 100 - chat.RECALL_HOP_OVERHEAD_CHARS]


@pytest.mark.parametrize("single_large_turn", [
    False,
    True,
])
def test_steward_full_prompt_fits_4096(runtime, single_large_turn):
    scale = assembler.DEFAULT_TOKEN_SCALE
    assert steward._transcript_cap(runtime.config, scale) < 2600
    turns = [message("user", "x" * 10000)] if single_large_turn else [
        message("user" if i % 2 == 0 else "assistant", "x" * 800) for i in range(12)]
    sid = reconcile(runtime.store, turns).session_id
    provider = Provider()
    asyncio.run(steward.run_steward(runtime.config, runtime.store, sid,
                                   runtime.store.live_events(sid), provider, "m", len(turns), scale=scale))
    assert provider.calls
    for payload in provider.calls:
        full_estimate = assembler.estimate_tokens(payload["messages"])
        assert full_estimate * scale + steward._output_reserve(runtime.config) <= 4096


@pytest.mark.parametrize("body,refused", [
    ("context length exceeded maximum tokens", False),
    ("unsupported response_format json_schema", True),
])
def test_only_schema_400_poisons_schema_cache(monkeypatch, body, refused):
    monkeypatch.setattr(steward, "_SCHEMA_REFUSED", set())
    request = httpx.Request("POST", "https://backend.invalid/chat")
    error = httpx.HTTPStatusError(body, request=request, response=httpx.Response(400, text=body, request=request))

    class RejectOnce(Provider):
        async def chat_completions(self, payload):
            if not self.calls:
                self.calls.append(payload)
                raise error
            return await super().chat_completions(payload)

    provider = RejectOnce()
    coro = steward._extract(config(), provider, "m", [message("user", "extract")], True)
    if refused:
        asyncio.run(coro)
        assert "response_format" not in provider.calls[-1]
    else:
        with pytest.raises(httpx.HTTPStatusError):
            asyncio.run(coro)
        assert len(provider.calls) == 1
    assert ("v6-regression:m" in steward._SCHEMA_REFUSED) is refused


@pytest.mark.parametrize("short_count", [0, 8])
def test_oversized_first_commitment_does_not_block_short_items(short_count):
    state = LedgerState()
    # A long standing request outranks the later short entries. All fields
    # remain within the ledger's 400-character cap.
    state.records["r1"] = Record("r1", "commitment", {
        "statement": "界" * 400, "trigger": "界" * 400, "actor": "assistant", "status": "open",
    }, updated_seq=100)
    for i in range(short_count):
        rid = f"r{i + 2}"
        state.records[rid] = Record(rid, "commitment", {"statement": f"short task {i}", "status": "open"})
    text = asyncio.run(render_memory(config(), state, [], None, "", None, 700))
    assert "界" * 400 not in text
    if short_count:
        for i in range(short_count):
            assert f"short task {i}" in text
        assert "say more exist" in text
    else:
        assert "1 tracked items exist but are not shown" in text


def test_redeclared_thread_local_handle_resolves_to_existing_thread():
    state = LedgerState()
    apply_ops(state, [{"op": "thread", "id": "old", "name": "garden"}], 1, 2)
    existing = next(iter(state.threads))
    report = apply_ops(state, [
        {"op": "add", "kind": "fact", "subject": "liner", "claim": "4m", "thread": "n1"},
        {"op": "thread", "id": "n1", "name": "garden", "summary": "updated"},
    ], 2, 4)
    assert not report.dropped
    assert list(state.threads) == [existing]
    assert state.records["r1"].data["thread"] == existing
    assert state.threads[existing].data["summary"] == "updated"


def test_explicit_null_clears_due_but_omissions_preserve_fields():
    state = LedgerState()
    apply_ops(state, [{"op": "add", "kind": "commitment", "statement": "remind me",
                       "due": "2030-01-01T12:00", "trigger": "tomorrow", "actor": "user"}], 1, 2)
    apply_ops(state, [{"op": "update", "id": "r1", "trigger": "only when asked"}], 2, 4)
    before = dict(state.records["r1"].data)
    assert before["due"] == "2030-01-01T12:00"
    report = apply_ops(state, [{"op": "update", "id": "r1", "due": None}], 3, 6)
    assert report.applied == 1
    assert not report.dropped
    assert state.records["r1"].data == {key: value for key, value in before.items() if key != "due"}


def test_assistant_opening_truncation_reuses_session(runtime):
    sid = reconcile(runtime.store, [message("assistant", "Welcome to the garden, how can I help?")]).session_id
    result = reconcile(runtime.store, [message("assistant", "Welcome to the garden"), message("user", "hello")])
    assert result.session_id == sid
    assert result.outcome == "truncation"
    assert runtime.store.list_session_ids() == [sid]


def test_user_opening_anchor_excludes_unrelated_session(runtime):
    transcript = [message("user", "my garden")]
    sid = reconcile(runtime.store, transcript).session_id
    unrelated = reconcile(runtime.store, [message("user", "my boat")]).session_id
    anchor = _anchor(transcript)
    assert anchor is not None
    assert sid in runtime.store.list_session_ids(anchor)
    assert unrelated not in runtime.store.list_session_ids(anchor)
    assert reconcile(runtime.store, transcript + [message("assistant", "hello")]).session_id == sid


def test_request_does_not_hang_behind_stuck_maintenance(tmp_path, monkeypatch):
    # Live: a background fold stuck in upstream 429 retries held the session
    # lock and the next request waited >300 s (harness timeout, soak aborted).
    # A request now waits at most maintenance_wait_s, proceeds on committed
    # state, and never starts a second steward for the same span.
    from server.mind.runtime import MindRuntime
    from server.mind.config import MindConfig

    cfg = MindConfig(
        enabled=True, db_dir=str(tmp_path), mem_backend="lexical", window=8192,
        workspace_cap_tokens=2400, summary_trigger_tokens=10, min_keep_turns=1,
        maintenance_wait_s=0.05,
    )
    runtime = MindRuntime(cfg)
    calls = []

    async def go():
        entered, release = asyncio.Event(), asyncio.Event()

        async def stuck(*args, **kwargs):
            calls.append(args[6])
            entered.set()
            await release.wait()
            from server.mind.steward import FoldOutcome
            return FoldOutcome()

        monkeypatch.setattr("server.mind.runtime.run_steward", stuck)
        transcript = [message("user", "first " * 330), message("assistant", "answer " * 510),
                      message("user", "second " * 220)]
        prepared = await runtime.prepare(transcript, None, "m")
        sid = prepared.session_id
        reply = message("assistant", "reply " * 800)
        runtime.observe_reply(sid, reply)
        await asyncio.wait_for(entered.wait(), 2)
        following = await asyncio.wait_for(
            runtime.prepare(transcript + [reply, message("user", "next")], None, "m"), 2
        )
        assert following.session_id == sid
        assert following.messages[-1]["content"].endswith("next")
        assert len(calls) == 1  # the request did not start a rival steward
        release.set()
        await runtime.drain()

    asyncio.run(go())


def test_add_without_kind_is_recovered_not_dropped():
    # Live shape from a loosely-constrained fold: the fact text sat in
    # "trigger", the key in "topic", no kind anywhere — a decision was lost.
    from server.mind.ledger import LedgerState, apply_ops

    state = LedgerState()
    report = apply_ops(state, [
        {"op": "add", "topic": "heater", "trigger": "Mara ordered the Autoterm 2D diesel heater. (core: true)"},
        {"op": "add", "statement": "remind Mara to book the ferry", "trigger": "before June"},
        {"op": "add", "topic": "paint", "status": "decided", "choice": "Drift Sage"},
        {"op": "add", "thread": "t9"},
    ], fold=1, span_to=4)
    kinds = sorted(record.kind for record in state.records.values())
    assert kinds == ["commitment", "decision", "fact"]
    fact = next(r for r in state.records.values() if r.kind == "fact")
    assert fact.data["subject"] == "heater" and fact.data["core"] is True
    assert fact.data["claim"] == "Mara ordered the Autoterm 2D diesel heater."
    assert len(report.dropped) == 1  # the op with nothing recoverable


def test_steward_schema_is_a_discriminated_union():
    from server.mind.steward import OP_SCHEMA

    shapes = OP_SCHEMA["schema"]["properties"]["ops"]["items"]["anyOf"]
    assert all(shape["additionalProperties"] is False for shape in shapes)
    fact = next(s for s in shapes if s["properties"].get("kind") == {"const": "fact"})
    assert {"subject", "claim"} <= set(fact["required"]) and "trigger" not in fact["properties"]


def test_multi_part_cue_scores_each_clause():
    from server.mind.relevance import _clauses, score_texts

    class ClauseMem:
        async def text_sims(self, query, texts):
            # The blurred whole question matches nothing; one clause does.
            return [0.9 if query.strip() == "the battery size" and "280" in t else 0.1 for t in texts]

    cue = "Quick-fire round: the van's name, the dog's name, and the battery size?"
    assert "the battery size" in _clauses(cue)
    assert _clauses("What size was the fresh water tank I got?") == []
    scored = asyncio.run(score_texts(cue, ["electrical system: 280Ah LiFePO4", "paint: green"], ClauseMem()))
    assert scored[0].matched and not scored[1].matched


def test_split_memory_keeps_the_system_message_stable_across_turns(tmp_path):
    # The cache contract: between folds the system message must be byte-
    # identical whatever the user says and whatever the clock reads; all
    # per-turn memory sits in a marked block on the LATEST user message only.
    from server.mind.config import MindConfig
    from server.mind.runtime import MindRuntime
    from server.mind.memory_view import NOTES_OPEN

    runtime = MindRuntime(MindConfig(enabled=True, db_dir=str(tmp_path), mem_backend="lexical",
                                     memory_placement="split"))

    async def go():
        transcript = [message("system", "You are Sable."), message("user", "my dog is called Biscuit"),
                      message("assistant", "noted"), message("user", "and the van is Juniper")]
        first = await runtime.prepare(transcript, None, "m", now=1_800_000_000.0)
        sid = first.session_id
        runtime.store.append_fold(sid, 1, 2, [
            {"op": "add", "kind": "fact", "subject": "dog name", "claim": "Biscuit", "core": True},
            {"op": "add", "kind": "fact", "subject": "favourite biscuit", "claim": "ginger nuts"},
            {"op": "add", "kind": "commitment", "statement": "remind about the vet", "trigger": "in May"},
        ], "They met the dog.")
        systems, lasts = [], []
        for turn, text in enumerate(["what biscuit do I like?", "tell me about sails", "the dog again?"]):
            transcript += [message("assistant", f"reply {turn}"), message("user", text)]
            prepared = await runtime.prepare(transcript, None, "m", now=1_800_000_000.0 + 3600 * (turn + 1))
            systems.append(prepared.messages[0]["content"])
            lasts.append(prepared.messages[-1]["content"])
            # older user messages never carry a block
            assert all(NOTES_OPEN not in str(m["content"]) for m in prepared.messages[1:-1])
        assert systems[0] == systems[1] == systems[2]
        assert "Biscuit" in systems[0] and "remind about the vet" in systems[0]
        assert "Current time" not in systems[0]
        assert all(NOTES_OPEN in last and "Current time" in last for last in lasts)
        # non-core facts are per-turn material: on the user message, never in the system
        assert "ginger nuts" in lasts[0] and "ginger nuts" not in systems[0]

    asyncio.run(go())


def test_split_parts_respect_the_shared_budget():
    from server.mind.ledger import LedgerState, apply_ops
    from server.mind.memory_view import estimate_tokens, render_memory_parts
    from server.mind.config import MindConfig

    state = LedgerState()
    apply_ops(state, [{"op": "add", "kind": "fact", "subject": f"thing {i}", "claim": "detail " * 30,
                       "core": i % 2 == 0} for i in range(120)]
              + [{"op": "add", "kind": "commitment", "statement": f"promise {i} " + "x " * 20} for i in range(40)],
              fold=1, span_to=2)
    for budget in (900, 1600, 4000):
        parts = asyncio.run(render_memory_parts(MindConfig(), state, [], None, "thing 7 detail", None, budget, now=1_800_000_000.0))
        assert estimate_tokens(parts.stable) + estimate_tokens(parts.volatile) <= budget
        assert "Open commitments" in parts.stable


def test_triggered_commitment_is_nudged_only_on_real_trigger_match():
    # Live finding: "payday is this friday" did not surface "order copper
    # rivets — on payday"; the item was merely listed. The per-turn notes now
    # say it plainly — but one shared common word must not fire a reminder.
    from server.mind.ledger import LedgerState, apply_ops
    from server.mind.memory_view import render_memory_parts
    from server.mind.config import MindConfig

    state = LedgerState()
    apply_ops(state, [
        {"op": "add", "kind": "commitment", "statement": "remind Noor to order copper rivets", "trigger": "on payday"},
        {"op": "add", "kind": "commitment", "statement": "reseal the roof vent", "trigger": "before the first rain test"},
    ], fold=1, span_to=2)

    def notes(cue):
        return asyncio.run(render_memory_parts(MindConfig(), state, [], None, cue, None, 1600)).volatile

    fired = notes("ugh long week. payday is this friday at least")
    assert "bring it up now" in fired and "copper rivets" in fired and "roof vent" not in fired.split("bring it up now")[1]
    assert "bring it up now" not in notes("I failed my driving test today")
    assert "roof vent" in notes("doing the first rain test on saturday").split("bring it up now")[1]


def test_extraction_target_routes_to_another_provider(tmp_path, monkeypatch):
    from server.mind.config import MindConfig
    from server.mind.runtime import MindRuntime
    from server.routing import router

    remote, local = object(), object()
    monkeypatch.setitem(router.PROVIDERS, "openrouter", remote)
    runtime = MindRuntime(MindConfig(enabled=True, db_dir=str(tmp_path), mem_backend="lexical",
                                     extraction_model="openrouter:google/gemma-4-26b-a4b-it"))
    assert runtime._extraction_target(local, "gemma-local") == (remote, "google/gemma-4-26b-a4b-it")
    runtime.config.extraction_model = "smaller-local-model"
    assert runtime._extraction_target(local, "gemma-local") == (local, "smaller-local-model")
    runtime.config.extraction_model = None
    assert runtime._extraction_target(local, "gemma-local") == (local, "gemma-local")


def test_first_fold_lands_into_reserved_space(tmp_path):
    # Live: 32 turns of pure texture filled the whole workspace; when the first
    # fold landed, the new ~1.3k-token memory section overflowed the hard
    # limit, uncovered events were evicted and a second fold ran ON the
    # request path (a 169 s turn). Pressure is now measured with the memory
    # budget reserved from turn one, so a fold never forces another.
    from server.mind.assembler import assemble, token_budgets
    from server.mind.config import MindConfig
    from server.mind.memory_view import memory_budget
    from server.mind.store import MindStore

    cfg = MindConfig(window=8192)
    store = MindStore(tmp_path / "m.sqlite3")
    sid = store.create_session(None)
    for turn in range(40):
        store.append_event(sid, message("user", f"turn {turn} " + "word " * 120), source="client")
        store.append_event(sid, message("assistant", f"reply {turn} " + "word " * 120), source="mind")
    events = store.live_events(sid)
    budget, hard = token_budgets(cfg)
    reserve = memory_budget(cfg, budget)

    empty = assemble(cfg, "sys", events, covered_upto=0, memory_budget_tokens=reserve)
    # What the budget WANTS in view leaves room for a full memory section...
    wanted = sum(
        len(json.dumps(e.message)) // 4 for e in events if e.seq >= empty.desired_from_seq
    )
    assert wanted + reserve <= budget
    # ...so once a fold covers the rest and memory appears, nothing uncovered is evicted.
    landed = assemble(
        cfg, "sys", events, covered_upto=empty.desired_from_seq - 1,
        memory_text="m" * (reserve * 4 * 6 // 10), volatile_text="v" * (reserve * 4 * 4 // 10),
        memory_budget_tokens=reserve, fold_boundaries=[empty.desired_from_seq - 1],
    )
    assert landed.evicted_uncovered == 0
    assert landed.estimated_tokens <= hard


def test_stable_memory_survives_budget_jitter_and_nudges_fire_once():
    from server.mind.ledger import LedgerState, apply_ops
    from server.mind.memory_view import render_memory_parts
    from server.mind.config import MindConfig

    state = LedgerState()
    apply_ops(state, [{"op": "add", "kind": "fact", "subject": f"thing {i}", "claim": "detail " * 12,
                       "core": True} for i in range(40)]
              + [{"op": "add", "kind": "commitment", "statement": "order copper rivets", "trigger": "on payday"}],
              fold=1, span_to=2)
    cache, nudged = {}, set()

    def render(budget, cue):
        return asyncio.run(render_memory_parts(
            MindConfig(), state, [], None, cue, None, budget,
            stable_cache=cache, revision=(1, 1, 0, 0), already_nudged=nudged,
        ))

    first = render(1600, "payday is this friday")
    assert "bring it up now" in first.volatile
    # Calibration wobble moves the budget a little: the system text must not move.
    assert render(1540, "anything").stable == first.stable == render(1700, "else").stable
    # ...but a real squeeze, or a new ledger revision, re-renders it.
    assert render(900, "anything").stable != first.stable
    # The nudge fired once; the same trigger next turn does not nag.
    assert "bring it up now" not in render(1600, "yes it's payday, I know").volatile


def test_runtime_never_folds_on_the_request_path_in_a_steady_conversation(tmp_path, monkeypatch):
    # Live (twice): the first fold came late, its memory section overflowed the
    # window guard, and a second fold ran synchronously — 160-170 s turns. The
    # assembler-level reserve was not enough: fold pressure is measured in the
    # BACKGROUND pass, which has to reserve the memory budget too.
    from server.mind.config import MindConfig
    from server.mind.runtime import MindRuntime
    from server.mind import runtime as runtime_module

    class Provider:
        name = "fake"

        async def chat_completions(self, payload):
            facts = [{"op": "add", "kind": "fact", "subject": f"s{i} {len(payload['messages'][-1]['content'])}",
                      "claim": "detail " * 25, "core": i % 2 == 0} for i in range(6)]
            return {"choices": [{"message": {"content": json.dumps({"ops": facts, "episode": "things happened. " * 6})}}]}

    paths = []
    original = runtime_module.MindRuntime._fold

    async def spy(self, session_id, events, provider, model, upto, clock, path):
        paths.append(path)
        return await original(self, session_id, events, provider, model, upto, clock, path)

    monkeypatch.setattr(runtime_module.MindRuntime, "_fold", spy)
    runtime = MindRuntime(MindConfig(enabled=True, db_dir=str(tmp_path), mem_backend="lexical", window=8192))

    async def go():
        transcript = [message("system", "You are Sable.")]
        worst = 0
        for turn in range(45):
            transcript.append(message("user", f"turn {turn}: " + "lorem ipsum dolor " * 25))
            prepared = await runtime.prepare(transcript, Provider(), "m", now=1_800_000_000.0 + 60 * turn)
            worst = max(worst, prepared.estimated_tokens)
            reply = message("assistant", f"reply {turn}: " + "consectetur adipiscing " * 25)
            runtime.observe_reply(prepared.session_id, reply)
            transcript.append(reply)
            await runtime.drain()
        return worst

    worst = asyncio.run(go())
    assert paths and set(paths) == {"background"}
    from server.mind.assembler import token_budgets
    # Over the soft budget is allowed while a fold catches up; over the hard
    # limit never (that is what evicts uncovered truth and forces a sync fold).
    assert worst <= token_budgets(runtime.config)[1]


def test_style_note_names_the_rut_and_stays_quiet_otherwise():
    from server.mind.memory_view import style_note

    rut = [
        "Nice work. I'll be here whenever you need me. How does it feel?",
        "Good plan. What comes next?",
        "Solid. I'll be here whenever you need me. Ready for it?",
    ]
    note = style_note(rut)
    assert "3 of your last 3 replies ended with a question" in note
    assert "i'll be here whenever" in note
    assert style_note(["All done.", "Fine then, that settles it.", "Ok!"]) == ""
    assert style_note(rut[:2]) == ""  # too little evidence to call it a habit
    # a statement breaks the streak
    assert "ended with a question" not in style_note(rut + ["That is that."])


def test_observe_reads_the_surface_and_stays_quiet_when_nothing_is_off():
    from server.mind.thoughts import observe

    long_reply = "That sounds like a lot to carry. " + "word " * 95
    users = ["I had a long day and the dog ate my shoe and then work exploded " * 3, "ok", "lol", "fair", "k thanks"]
    lines = observe(users, [long_reply] * 5)
    joined = " ".join(lines)
    assert "one short line" in joined and "one or two sentences" in joined
    assert "opening by validating" in joined
    # Bold text in a chat gets called out, unless the user writes that way too.
    assert any("plain sentences" in line for line in observe(users, ["**Plan:** do it. " + "w " * 70] * 4))
    # A varied, plain conversation produces nothing to say.
    varied = ["Ha. No.", "Honestly I'd sell it. " + "word " * 30, "Sure.", "Tuesday works, bring the dog. " + "w " * 12]
    assert observe(["so what do you think about the boat then, given everything"] * 4, varied) == []


def test_quick_thoughts_join_the_notes_block_and_fail_open(tmp_path):
    from server.mind.config import MindConfig
    from server.mind.runtime import MindRuntime

    class Provider:
        name = "fake"
        calls = 0

        async def chat_completions(self, payload):
            Provider.calls += 1
            assert payload["messages"][-1]["role"] == "user"  # asked over the same prefix
            if payload.get("logprobs"):
                import math
                return {"choices": [{"message": {"content": "A"}, "logprobs": {"content": [{"token": "A", "top_logprobs": [
                    {"token": "A", "logprob": math.log(0.9)}, {"token": " B", "logprob": math.log(0.1)}]}]}}]}
            raise RuntimeError("sketch backend down")

    runtime = MindRuntime(MindConfig(enabled=True, db_dir=str(tmp_path), mem_backend="lexical",
                                     thoughts="observe,typed,sketch"))

    async def go():
        transcript = [message("user", "ugh. my boss moved the deadline again, i am so done")]
        prepared = await runtime.prepare(transcript, Provider(), "m")
        return prepared.messages[-1]["content"]

    last = asyncio.run(go())
    assert "They want to be heard" in last and "never mention it" in last
    assert last.endswith("i am so done")  # the user's words still close the message
    assert Provider.calls == 2  # typed answered; sketch failed open without breaking the turn
