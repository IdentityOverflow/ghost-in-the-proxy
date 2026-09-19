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
