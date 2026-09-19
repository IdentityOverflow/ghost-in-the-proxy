"""Memory v6 contracts: replayable truth, bounded views, serialized maintenance."""

import asyncio
import copy
import json

import httpx
import pytest

from server.endpoints.chat import _DeltaCollector
from server.mind import steward
from server.mind.assembler import DEFAULT_TOKEN_SCALE, assemble, estimate_tokens, token_budgets
from server.mind.config import MindConfig
from server.mind.ledger import Episode, LedgerState, apply_ops, replay
from server.mind.mem import LexicalMem, MemSpan
from server.mind.memory_view import ThreadsView, memory_budget, render_memory
from server.mind.recall import resolve_recall
from server.mind.runtime import MindRuntime
from server.mind.store import MindStore


def config(**overrides):
    return MindConfig(**{
        "enabled": True, "window": 8192, "mem_backend": "lexical",
        "time_enabled": False, "background_fold": True,
        "steward_json_schema": True, "extraction_model": None, **overrides,
    })


def message(role, text):
    return {"role": role, "content": text}


def fact(subject="battery", claim="280Ah", **fields):
    return {"op": "add", "kind": "fact", "subject": subject, "claim": claim, **fields}


def commitment(statement="rotate token", **fields):
    return {"op": "add", "kind": "commitment", "statement": statement, **fields}


def decision(topic="heater", **fields):
    return {"op": "add", "kind": "decision", "topic": topic, **fields}


def state_with(*ops):
    state = LedgerState()
    report = apply_ops(state, list(ops), fold=1, span_to=2)
    assert not report.dropped
    return state


def view(state, budget=4000, **kwargs):
    return asyncio.run(render_memory(
        config(), state, kwargs.pop("consolidations", []), kwargs.pop("threads", None),
        kwargs.pop("cue", ""), None, budget, **kwargs,
    ))


@pytest.fixture
def store(tmp_path):
    return MindStore(tmp_path / "memory.sqlite3")


def session_with_events(store, count=8, chars=40):
    sid = store.create_session(None)
    for i in range(count):
        store.append_event(
            sid, message("user" if i % 2 == 0 else "assistant", f"turn {i}: " + "x" * chars),
            source="client", confirmed=True,
        )
    return sid


class CannedProvider:
    name = "memory-v6-test"

    def __init__(self, *responses):
        self.responses = responses or ('{"ops": [], "episode": "Things happened."}',)
        self.calls = []

    async def chat_completions(self, payload):
        self.calls.append(copy.deepcopy(payload))
        response = self.responses[min(len(self.calls) - 1, len(self.responses) - 1)]
        if isinstance(response, Exception):
            raise response
        return {"choices": [{"message": {"content": response}}]}


def test_ops_add_update_close_and_omission():
    state = state_with(fact(), commitment(), decision(status="leaning", choice="diesel"))
    report = apply_ops(state, [
        {"op": "update", "id": "r1", "claim": "300Ah (was 280Ah)", "src": 5},
        {"op": "close", "id": "r2"},
    ], fold=2, span_to=6)
    assert report.applied == 2 and report.dropped == []
    assert state.records["r1"].data["claim"] == "300Ah (was 280Ah)"
    assert state.records["r1"].src == 5
    assert state.records["r1"].created_fold == 1
    assert state.records["r1"].updated_fold == 2
    assert state.records["r1"].updated_seq == 6
    assert not state.records["r2"].is_open
    assert state.records["r3"].data["status"] == "leaning"  # omission is unchanged
    apply_ops(state, [{"op": "close", "id": "r3", "choice": "electric"}], 3, 8)
    assert state.records["r3"].data == {"topic": "heater", "status": "decided", "choice": "electric"}


def test_unknown_id_drops_only_that_op():
    state = state_with(fact())
    report = apply_ops(state, [
        {"op": "update", "id": "r999", "claim": "invented"},
        {"op": "update", "id": "r1", "claim": "300Ah"}, commitment(),
    ], 2, 4)
    assert len(report.dropped) == 1 and "unknown id" in report.dropped[0]
    assert report.applied == 2
    assert state.records["r1"].data["claim"] == "300Ah"
    assert state.records["r2"].kind == "commitment"


@pytest.mark.parametrize("thread_first", [True, False])
def test_proposal_local_thread_handles(thread_first):
    thread = {"op": "thread", "id": "n1", "name": "heating", "summary": "Keep warm"}
    record = fact("heater", "diesel", thread="n1")
    state = state_with(*([thread, record] if thread_first else [record, thread]))
    assert list(state.threads) == ["t1"]
    assert state.records["r1"].data["thread"] == "t1"
    assert state.threads["t1"].name == "heating"


def test_add_normalized_live_key_updates_and_keeps_id():
    state = state_with(fact("Battery-capacity", "200Ah"))
    report = apply_ops(state, [fact(" BATTERY capacity! ", "280Ah")], 2, 4)
    assert report.deduped == 1 and report.applied == 1
    assert list(state.records) == ["r1"] and state.next_record == 2
    assert state.records["r1"].data["claim"] == "280Ah"
    assert state.records["r1"].created_fold == 1
    assert state.records["r1"].updated_fold == 2


@pytest.mark.parametrize("status", ["done", "dropped"])
def test_closed_commitment_can_be_readded_with_new_id(status):
    state = state_with(commitment())
    apply_ops(state, [{"op": "close", "id": "r1", "status": status}], 2, 4)
    report = apply_ops(state, [commitment()], 3, 6)
    assert report.deduped == 0
    assert list(state.records) == ["r1", "r2"]
    assert state.records["r1"].data["status"] == status
    assert state.records["r2"].is_open


@pytest.mark.parametrize("op", [commitment(status="nonsense"), decision(status="nonsense")])
def test_invalid_add_and_update_statuses_normalize_to_open(op):
    state = state_with(op)
    assert state.records["r1"].data["status"] == "open"
    apply_ops(state, [{"op": "update", "id": "r1", "status": "BOGUS"}], 2, 4)
    assert state.records["r1"].data["status"] == "open"


def test_close_fact_is_dropped_without_mutation():
    state = state_with(fact())
    before = copy.deepcopy(state)
    report = apply_ops(state, [{"op": "close", "id": "r1"}], 2, 4)
    assert report.applied == 0 and len(report.dropped) == 1
    assert "close on a fact" in report.dropped[0]
    assert state == before


def test_replay_deterministic_ids_and_prefix_snapshot():
    folds = [
        {"seq": 1, "span_from": 1, "span_to": 2, "ops": [
            fact(thread="n1"), {"op": "thread", "id": "n1", "name": "van"}, commitment(),
        ], "episode": "First turn"},
        {"seq": 2, "span_from": 3, "span_to": 4, "ops": [
            {"op": "update", "id": "r1", "claim": "300Ah"},
            {"op": "close", "id": "r2"}, commitment(),
        ], "episode": "Second turn"},
    ]
    before = replay(folds[:1])
    snapshot = copy.deepcopy(before)
    after = replay(folds)
    assert after == replay(copy.deepcopy(folds))
    assert list(after.records) == ["r1", "r2", "r3"]
    assert list(after.threads) == ["t1"]
    assert replay(folds[:1]) == snapshot == before
    assert before.records["r1"].data["claim"] == "280Ah"
    assert before.records["r2"].is_open
    assert before.covered_upto == 2 and after.covered_upto == 4
    assert [e.text for e in after.episodes] == ["First turn", "Second turn"]


def test_store_rejects_fold_finishing_after_fork(store):
    sid = session_with_events(store, 4)
    replacement = store.append_event(sid, message("user", "edited"), source="client")
    store.supersede_from(sid, 3, replacement)
    assert store.append_fold(sid, 1, 4, [fact()], "stale") is None
    assert store.live_folds(sid) == []
    assert replay(store.live_folds(sid)).covered_upto == 0


@pytest.mark.parametrize("fork_seq", [3, 5, 8])
def test_deep_fork_preserves_first_fold_and_invalidates_touching_derivations(store, fork_seq):
    sid = session_with_events(store)
    assert store.append_fold(sid, 1, 2, [fact()], "original") == 1
    prefix = replay(store.live_folds(sid))
    assert store.append_fold(sid, 3, 8, [
        {"op": "update", "id": "r1", "claim": "changed"}, commitment(),
    ], "later") == 2
    store.append_consolidation(sid, 1, 1, 1, 1, 2, "safe era")
    store.append_consolidation(sid, 1, 1, 2, 1, 8, "touches fork")
    store.set_dynamics(sid, "t1", 0.8, 0.4, 8)
    replacement = store.append_event(sid, message("user", "edited"), source="client")
    store.supersede_from(sid, fork_seq, replacement)
    assert [f["seq"] for f in store.live_folds(sid)] == [1]
    assert replay(store.live_folds(sid)) == prefix
    assert [c["content"] for c in store.live_consolidations(sid)] == ["safe era"]
    assert store.get_dynamics(sid) == {}
    assert [e.seq for e in store.live_events(sid)] == list(range(1, fork_seq)) + [replacement]


@pytest.mark.parametrize("empty", ["{}", '{"ops": [], "episode": ""}'])
def test_empty_steward_proposal_preserves_ledger_and_commits_prose(store, empty):
    sid = session_with_events(store, 4)
    store.append_fold(sid, 1, 2, [fact(), commitment()], "before")
    before = replay(store.live_folds(sid))
    # proposal, one unconstrained retry, then the episode-only fallback
    provider = CannedProvider(empty, empty, "<think>private reasoning</think>They discussed travel.")
    outcome = asyncio.run(steward.run_steward(
        config(), store, sid, store.live_events(sid), provider, "m", 4,
    ))
    folds = store.live_folds(sid)
    assert outcome.folds == outcome.prose_fallbacks == 1
    assert len(provider.calls) == 3
    assert folds[-1]["kind"] == "prose" and folds[-1]["ops"] == []
    assert folds[-1]["episode"] == "They discussed travel."
    after = replay(folds)
    assert after.records == before.records and after.threads == before.threads
    assert after.covered_upto == 4 and len(after.episodes) == 2


def test_steward_strips_think_blocks_before_json_parse(store):
    sid = session_with_events(store, 2)
    proposal = json.dumps({"ops": [fact()], "episode": "Battery recorded."})
    provider = CannedProvider('<think>bad {braces} and secret reasoning</think>\n' + proposal)
    outcome = asyncio.run(steward.run_steward(
        config(), store, sid, store.live_events(sid), provider, "m", 2,
    ))
    assert outcome.ops_applied == 1 and outcome.prose_fallbacks == 0
    assert len(provider.calls) == 1
    state = replay(store.live_folds(sid))
    assert state.records["r1"].data["claim"] == "280Ah"
    assert state.episodes[0].text == "Battery recorded."


def test_schema_400_retried_without_format_and_remembered(store, monkeypatch):
    monkeypatch.setattr(steward, "_SCHEMA_REFUSED", set())
    request = httpx.Request("POST", "https://test.invalid/chat/completions")
    refusal = httpx.HTTPStatusError(
        "response_format unsupported", request=request,
        response=httpx.Response(400, request=request),
    )
    provider = CannedProvider(refusal, '{"ops": [], "episode": "plain JSON works"}')
    sid = session_with_events(store, 4)

    async def go():
        first = await steward.run_steward(config(), store, sid, store.live_events(sid), provider, "m", 2)
        second = await steward.run_steward(config(), store, sid, store.live_events(sid), provider, "m", 4)
        assert first.prose_fallbacks == second.prose_fallbacks == 0

    asyncio.run(go())
    assert len(provider.calls) == 3
    assert "response_format" in provider.calls[0]
    assert all("response_format" not in p for p in provider.calls[1:])
    assert provider.calls[0]["messages"] == provider.calls[1]["messages"]
    assert f"{provider.name}:m" in steward._SCHEMA_REFUSED
    assert [f["span_to"] for f in store.live_folds(sid)] == [2, 4]


def test_oversized_steward_span_is_contiguous_bounded_folds(store):
    sid = session_with_events(store, 12, chars=1200)
    provider = CannedProvider(json.dumps({"ops": [fact()], "episode": "encoded"}))
    outcome = asyncio.run(steward.run_steward(
        config(steward_input_tokens=900), store, sid, store.live_events(sid), provider, "m", 12,
    ))
    folds = store.live_folds(sid)
    assert len(folds) == len(provider.calls) == outcome.folds > 1
    assert folds[0]["span_from"] == 1 and folds[-1]["span_to"] == 12
    assert all(a["span_to"] + 1 == b["span_from"] for a, b in zip(folds, folds[1:]))
    for payload in provider.calls:
        transcript = payload["messages"][1]["content"].split("NEW TURNS:\n")[1].split("\n\nChanges")[0]
        assert estimate_tokens(transcript) <= 900
    assert len(replay(folds).records) == 1  # later adds dedupe, not ledger replacement


def test_steward_slice_pins_open_items_before_relevant_facts():
    state = state_with(
        commitment("renew passport"), decision("route", status="open"),
        decision("heater", status="leaning"), commitment("finished task", status="done"),
        *[fact(f"battery {i}", "battery capacity " * 15) for i in range(100)],
    )
    cfg = config(steward_slice_tokens=300)
    text = asyncio.run(steward.build_slice(cfg, state, "battery capacity", None))
    assert "renew passport" in text and "route: OPEN" in text and "heater: LEANING" in text
    assert "finished task" not in text
    assert "battery" in text
    assert estimate_tokens(text) <= cfg.steward_slice_tokens


def test_steward_slice_many_pinned_items_respects_budget():
    state = state_with(*[decision(f"choice {i}", reason="detail " * 50) for i in range(30)])
    cfg = config(steward_slice_tokens=300)
    text = asyncio.run(steward.build_slice(cfg, state, "", None))
    assert estimate_tokens(text) <= cfg.steward_slice_tokens


def test_steward_slice_includes_all_open_commitments_when_they_fit():
    state = state_with(*[commitment(f"task {i:02}") for i in range(20)])
    text = asyncio.run(steward.build_slice(config(steward_slice_tokens=2000), state, "", None))
    assert estimate_tokens(text) <= 2000
    for i in range(20):
        assert f"task {i:02}" in text


def crowded_state():
    state = state_with(*[
        commitment(f"promise {i:03} " + "important " * 20) if i < 100 else
        fact(f"profile {i:03}", "detail " * 40, core=True) if i < 200 else
        decision(f"choice {i:03}", choice="option " * 30)
        for i in range(300)
    ])
    state.episodes = [Episode(i, i * 2 - 1, i * 2, f"episode {i:03} " + "travel " * 30) for i in range(1, 61)]
    return state


@pytest.mark.parametrize("budget", [
    900,
    1800,
    4000,
])
def test_memory_render_never_exceeds_budget_with_300_records_60_episodes(budget):
    state = crowded_state()
    assert len(state.records) == 300 and len(state.episodes) == 60
    text = view(state, budget)
    assert "Open commitments" in text
    assert estimate_tokens(text) <= budget


def test_open_commitments_take_priority_and_complete_list_is_truthful():
    state = crowded_state()
    # A single short promise competes with hundreds of facts/decisions/episodes.
    state.records = {key: record for key, record in state.records.items() if record.kind != "commitment"}
    apply_ops(state, [commitment("RAISE THIS PROMISE")], 2, 999)
    text = view(state, 900)
    assert "RAISE THIS PROMISE" in text
    assert "Open commitments (complete list of tracked items)" in text
    for heading in ("### Always true", "### Decisions", "### Earlier events"):
        if heading in text:
            assert text.index("### Open commitments") < text.index(heading)


def test_partial_commitment_list_is_not_labelled_complete():
    text = view(crowded_state(), 900)
    assert "most pressing of 100 tracked" in text and "say more exist" in text
    assert "complete list" not in text
    assert "promise 099" not in text  # not all promises fit


def test_leaning_and_core_profile_survive_rendering():
    state = state_with(fact("allergy", "peanuts", core=True), decision(status="leaning", choice="diesel"))
    text = view(state)
    profile = text.split("### Always true (who and what matters)\n")[1].split("###")[0]
    assert "allergy: peanuts" in profile
    assert "heater: LEANING (not yet decided) — diesel" in text


def test_nonadmitted_thread_fact_requires_matching_cue():
    state = state_with(
        {"op": "thread", "id": "n1", "name": "recipe"},
        fact("dumpling recipe", "semolina replaces flour", thread="n1"),
    )
    threads = ThreadsView(admitted=[], cued=[], all_keys={"t1"})
    assert "semolina replaces flour" not in view(state, threads=threads, cue="solar panel")
    text = view(state, threads=threads, cue="dumpling semolina")
    assert "semolina replaces flour" in text.split("### Recalled")[1]


def test_eras_hide_leaves_except_recent_three_and_cue_matches_in_time_order():
    state = LedgerState(episodes=[
        Episode(i, i * 2 - 1, i * 2, f"leaf-{i:02}" + (" violet telescope" if i == 2 else ""))
        for i in range(1, 11)
    ])
    eras = [{"seq": 1, "level": 1, "child_from": 1, "child_to": 10,
             "span_from": 1, "span_to": 20, "content": "ERA-GIST"}]
    text = view(state, consolidations=eras)
    assert "ERA-GIST" in text
    assert all(f"leaf-{i:02}" not in text for i in range(1, 8))
    assert all(f"leaf-{i:02}" in text for i in range(8, 11))
    cued = view(state, consolidations=eras, cue="violet telescope")
    assert "(recalled detail) leaf-02" in cued
    assert all(f"leaf-{i:02}" not in cued for i in (1, 3, 4, 5, 6, 7))
    assert cued.index("leaf-02") < cued.index("leaf-08") < cued.index("leaf-09") < cued.index("leaf-10")


def test_unconsolidated_episodes_render_oldest_first():
    state = LedgerState(episodes=[Episode(i, i, i, f"event-{i:02}") for i in range(1, 10)])
    text = view(state)
    positions = [text.index(f"event-{i:02}") for i in range(1, 10)]
    assert positions == sorted(positions)


def test_workspace_cap_at_128k_and_tools_subtracted():
    cfg = config(window=131072, workspace_cap_tokens=16000)
    assert token_budgets(cfg) == (16000, 20000)
    # When window rather than cap binds, every schema token comes off both limits.
    small = config(window=4096)
    budget, hard = token_budgets(small)
    assert token_budgets(small, tools_tokens=200) == (budget - 200, hard - 200)
    # Even with tools, a giant window must never reintroduce transcript stuffing.
    assert token_budgets(cfg, tools_tokens=200) == (16000, 20000)


@pytest.mark.parametrize("ends_on_assistant", [False, True])
def test_hard_guard_evicts_uncovered_history_keeps_newest_and_user_opening(store, ends_on_assistant):
    sid = session_with_events(store, 40 if ends_on_assistant else 41, chars=1800)
    events = store.live_events(sid)
    workspace = assemble(config(window=4096), "client", events, covered_upto=0)
    assert workspace.evicted_uncovered > 0
    assert workspace.messages[-1] == events[-1].message
    assert workspace.messages[1]["role"] == "user"
    assert workspace.estimated_tokens <= workspace.hard_limit_tokens


def test_giant_newest_user_middle_truncated_with_recall_marker(store):
    sid = store.create_session(None)
    original = "HEAD-ANCHOR " + "huge paste " * 5000 + " TAIL-ANCHOR"
    store.append_event(sid, message("user", original), source="client")
    workspace = assemble(config(window=4096), None, store.live_events(sid))
    rendered = workspace.messages[-1]["content"]
    assert rendered.startswith("HEAD-ANCHOR") and rendered.endswith("TAIL-ANCHOR")
    assert "characters omitted here to fit the context window" in rendered and "recall(...)" in rendered
    assert len(rendered) < len(original)
    assert workspace.estimated_tokens <= workspace.hard_limit_tokens
    assert store.live_events(sid)[0].message["content"] == original


def test_runtime_60_turns_system_stays_within_memory_budget(tmp_path):
    cfg = config(db_dir=str(tmp_path), workspace_cap_tokens=2400, summary_trigger_tokens=100)
    runtime = MindRuntime(cfg)
    provider = CannedProvider(json.dumps({
        "ops": [fact("name", "Ada", core=True), commitment("renew passport")],
        "episode": "They discussed travel and the next steps in their plans.",
    }))
    client_system = "You are a helpful travel planner."

    async def go():
        transcript = [message("system", client_system)]
        sid = None
        for turn in range(60):
            transcript.append(message("user", f"Question {turn}: " + "travel details " * 100))
            prepared = await runtime.prepare(transcript, provider, "m")
            sid = sid or prepared.session_id
            assert prepared.session_id == sid
            system = prepared.messages[0]
            assert system["role"] == "system" and system["content"].startswith(client_system)
            tools_cost = estimate_tokens(prepared.tools) if prepared.tools else 0
            budget, _ = token_budgets(cfg, tools_cost)
            assert estimate_tokens(system["content"]) <= memory_budget(cfg, budget) + estimate_tokens(client_system) + 1
            reply = message("assistant", f"Answer {turn}: " + "useful advice " * 60)
            runtime.observe_reply(sid, reply)
            transcript.append(reply)
            await runtime.drain()
        assert len(runtime.store.live_folds(sid)) >= 10
        assert replay(runtime.store.live_folds(sid)).records
        assert runtime.store.live_consolidations(sid)
        assert len(runtime.store.live_events(sid)) == 120

    asyncio.run(go())


def test_background_folds_after_reply_and_next_request_waits_without_double_fold(tmp_path, monkeypatch):
    cfg = config(db_dir=str(tmp_path), workspace_cap_tokens=2400, summary_trigger_tokens=10, min_keep_turns=1)
    runtime = MindRuntime(cfg)
    provider = CannedProvider()
    calls = []
    original = steward.run_steward

    async def go():
        entered, release, attempted = asyncio.Event(), asyncio.Event(), asyncio.Event()

        async def blocked(*args, **kwargs):
            calls.append(args[6])
            entered.set()
            await release.wait()
            return await original(*args, **kwargs)

        monkeypatch.setattr("server.mind.runtime.run_steward", blocked)
        # Fits before the reply; afterwards the newest user/reply pair fits,
        # but adding the preceding assistant block exceeds the soft cap.
        transcript = [message("user", "first " * 330), message("assistant", "answer " * 510),
                      message("user", "second " * 220)]
        prepared = await runtime.prepare(transcript, provider, "m")
        sid = prepared.session_id
        assert not calls and runtime.store.live_folds(sid) == []
        reply = message("assistant", "reply " * 800)
        runtime.observe_reply(sid, reply)
        await asyncio.wait_for(entered.wait(), 2)
        assert not runtime._maintaining[sid].done()  # the pass runs OFF the request lock
        assert runtime.store.live_folds(sid) == []  # provider is still thinking

        async def next_request():
            attempted.set()
            return await runtime.prepare(transcript + [reply, message("user", "next")], provider, "m")

        task = asyncio.create_task(next_request())
        try:
            await attempted.wait()
            await asyncio.sleep(0)  # task reached prepare's lock acquisition
            assert not task.done()
        finally:
            release.set()
        await runtime.drain()
        following = await asyncio.wait_for(task, 2)
        assert following.session_id == sid
        assert len(calls) == 1  # background fold only, no racing request fold
        assert len(provider.calls) == 1
        assert replay(runtime.store.live_folds(sid)).covered_upto == 2

    asyncio.run(go())


def test_note_usage_raises_scale_and_reduces_available_workspace(tmp_path):
    runtime = MindRuntime(config(db_dir=str(tmp_path)))
    sid = runtime.store.create_session(None)
    runtime.note_usage(sid, estimated_tokens=1000, prompt_tokens=2200)
    assert runtime._scale[sid] > DEFAULT_TOKEN_SCALE
    assert token_budgets(runtime.config, scale=runtime._scale[sid])[0] < token_budgets(runtime.config)[0]
    before = runtime._scale[sid]
    runtime.note_usage(sid, estimated_tokens=1000, prompt_tokens=None)
    assert runtime._scale[sid] == before


@pytest.mark.parametrize("budget", [200, 1200, 4500])
def test_recall_respects_total_char_budget(store, budget):
    sid = store.create_session(None)
    for i in range(3):
        store.append_event(sid, message("user", f"needle passage {i}: " + "z" * 5000), source="client")
    out = asyncio.run(resolve_recall(store.live_events(sid), '{"query": "needle passage"}', char_budget=budget))
    assert "[seq 1, user, verbatim]" in out and "needle passage 0" in out
    assert len(out) <= budget


def test_runtime_recall_char_budget_bounds_output(tmp_path):
    runtime = MindRuntime(config(db_dir=str(tmp_path), window=4096))
    sid = runtime.store.create_session(None)
    runtime.store.append_event(sid, message("user", "needle " * 3000), source="client")
    budget = runtime.recall_char_budget()
    assert budget == max(1200, min(int(4096 * 0.08) * 4, 12000))
    out = asyncio.run(runtime.resolve_recall(sid, '{"query": "needle"}'))
    assert "needle" in out and "[seq 1, user, verbatim]" in out
    assert len(out) <= budget


def test_autocue_ranks_all_history_before_filtering_folded(tmp_path):
    runtime = MindRuntime(config(db_dir=str(tmp_path)))
    sid = session_with_events(runtime.store, 20)

    class RankedMem(LexicalMem):
        autocue = True

        async def query(self, session_id, query, events):
            # Four newer hits rank above the old detail; top-4 before filtering loses it.
            spans = [MemSpan(seq, "user", "recent", 1.0, "fake", sim=0.9) for seq in (19, 17, 15, 13)]
            spans += [MemSpan(1, "user", "old detail", 0.8, "fake", sim=0.8)]
            return spans[:query.k]

    runtime.mem = RankedMem()
    spans = asyncio.run(runtime._autocue(sid, runtime.store.live_events(sid), folded_upto=2))
    assert [span.seq for span in spans] == [1]


def test_delta_collector_usage_only_chunk_with_empty_choices():
    collector = _DeltaCollector()
    collector.feed(b'data: {"choices":[{"delta":{"content":"hello"}}]}\n\n')
    usage = b'data: {"choices":[],"usage":{"prompt_tokens":1234,"completion_tokens":9}}\n\n'
    collector.feed(usage[:37])
    assert collector.prompt_tokens is None
    collector.feed(usage[37:] + b'data: [DONE]\n\n')
    assert collector.prompt_tokens == 1234
    assert collector.message() == {"role": "assistant", "content": "hello"}


def test_steward_salvages_complete_ops_from_truncated_json(store):
    # Live failure shape (gemma-4-26b, schema-constrained): derails mid-object
    # with a mangled key and whitespace to the token cap. The complete ops
    # before the break must still commit, as a normal steward fold.
    sid = session_with_events(store, 2)
    broken = (
        '{"ops":[{"op":"add","kind":"fact","subject":"van {name}","claim":"Juniper \\"the van\\""},'
        '{"op":"thread","id":"n2","name":"safety","kind:":"topic","summary_":"Mara is researching\n\n\n   \n\n'
    )
    provider = CannedProvider(broken)
    outcome = asyncio.run(steward.run_steward(
        config(), store, sid, store.live_events(sid), provider, "m", 2,
    ))
    assert outcome.folds == 1 and outcome.prose_fallbacks == 0
    assert len(provider.calls) == 1
    state = replay(store.live_folds(sid))
    assert [r.data["claim"] for r in state.records.values()] == ['Juniper "the van"']


def test_steward_mangled_keys_are_normalized(store):
    sid = session_with_events(store, 2)
    proposal = '{"ops":[{"op":"thread","id":"n1","name":"heating","kind:":"aside","summary ":"Heater talk."}],"episode":"x"}'
    asyncio.run(steward.run_steward(
        config(), store, sid, store.live_events(sid), CannedProvider(proposal), "m", 2,
    ))
    thread = next(iter(replay(store.live_folds(sid)).threads.values()))
    assert thread.data["kind"] == "aside" and thread.data["summary"] == "Heater talk."
