"""Mind v0: perception reconciliation, event store invariants, assembler budget."""

import pytest

from server.mind.assembler import assemble, estimate_tokens
from server.mind.config import MindConfig
from server.mind.perception import reconcile
from server.mind.store import MindStore


@pytest.fixture
def store(tmp_path):
    return MindStore(tmp_path / "test.sqlite3")


def config(**overrides) -> MindConfig:
    base = dict(enabled=True, db_dir="unused", window=4096)
    base.update(overrides)
    return MindConfig(**base)


def user(text):
    return {"role": "user", "content": text}


def assistant(text):
    return {"role": "assistant", "content": text}


SYSTEM = {"role": "system", "content": "client system prompt"}


def play_turn(store, transcript, reply):
    """Simulate one request/response cycle: reconcile, then record our reply."""
    recon = reconcile(store, transcript)
    seq = store.append_event(recon.session_id, assistant(reply), source="mind", confirmed=False)
    return recon, seq


def test_new_session_then_continue(store):
    recon1, _ = play_turn(store, [SYSTEM, user("hello")], "hi there")
    assert recon1.outcome == "new"

    transcript = [SYSTEM, user("hello"), assistant("hi there"), user("next question")]
    recon2 = reconcile(store, transcript)
    assert recon2.outcome == "continue"
    assert recon2.session_id == recon1.session_id
    assert recon2.new_events == [user("next question")]
    # Our provisional reply got confirmed by the client's transcript.
    events = store.live_events(recon2.session_id)
    assert [e.confirmed for e in events] == [True, True, True]


def test_different_system_prompt_rejects_short_coincidence(store):
    # Two clients both open with "hi": a one-message overlap under a
    # different system prompt is coincidence, not the same conversation.
    system_a = {"role": "system", "content": "You are client A."}
    system_b = {"role": "system", "content": "You are client B."}
    recon_a, _ = play_turn(store, [system_a, user("hi")], "hello!")
    recon_b = reconcile(store, [system_b, user("hi")])
    assert recon_b.outcome == "new"
    assert recon_b.session_id != recon_a.session_id


def test_system_prompt_breaks_tie_between_identical_openings(store):
    # Both sessions hold the identical two-message opening; only the client
    # system prompt distinguishes them. The matching prompt must win.
    system_a = {"role": "system", "content": "You are client A."}
    system_b = {"role": "system", "content": "You are client B."}
    opening = [user("hi"), assistant("Acknowledged.")]
    recon_a = reconcile(store, [system_a] + opening)
    recon_b = reconcile(store, [system_b] + opening)
    assert recon_a.session_id != recon_b.session_id

    recon = reconcile(store, [system_b] + opening + [user("what did I configure?")])
    assert recon.outcome == "continue"
    assert recon.session_id == recon_b.session_id


def test_same_client_system_change_still_continues(store):
    # Dynamic system prompts: the same client varying its prompt must not
    # lose its session — two matched user turns outrank the prompt change.
    system_v1 = {"role": "system", "content": "Prompt version 1."}
    system_v2 = {"role": "system", "content": "Prompt version 2."}
    history = [user("plan the trip to Kyoto"), assistant("planned"), user("add a temple day")]
    recon1 = reconcile(store, [system_v1] + history)
    transcript = [system_v2] + history + [assistant("added"), user("book it")]
    recon2 = reconcile(store, transcript)
    assert recon2.outcome == "continue"
    assert recon2.session_id == recon1.session_id


def test_s12_zero_overlap_is_certified_by_the_real_tokenizer():
    # The s12 discriminator's premise: probe A shares NO content words with
    # the aside, so lexical cue and recall search have nothing to grip. If
    # scenario wording drifts, this fails before a live run wastes a gate.
    from evals.scenarios.s12_semantic_callback import ASIDE, PROBE_LEXICAL, PROBE_SEMANTIC
    from server.mind.dynamics import tokenize

    assert tokenize(ASIDE) & tokenize(PROBE_SEMANTIC) == set()
    # The control probe must stay lexically reachable (cue needs >=2 hits).
    assert len(tokenize(ASIDE) & tokenize(PROBE_LEXICAL)) >= 2


def test_s13_probe_overlaps_certified_by_the_real_tokenizer():
    # Probe A must share nothing with the courier turn, probe B nothing with
    # the cat turn (its answer), and the control must keep >=2 hits on the
    # radiator turn (cue reinjection needs 2). Full turn texts, not asides:
    # lexical search sees whole events.
    from evals.scenarios.s13_sequence_recall import (
        PROBE_CONTROL,
        PROBE_DROPPED,
        PROBE_ORDER,
        TURN_CAT,
        TURN_COURIER,
        TURN_RADIATOR,
    )
    from server.mind.dynamics import tokenize

    assert tokenize(TURN_COURIER) & tokenize(PROBE_DROPPED) == set()
    assert tokenize(TURN_CAT) & tokenize(PROBE_ORDER) == set()
    assert len(tokenize(TURN_RADIATOR) & tokenize(PROBE_CONTROL)) >= 2


class _RecordingMem:
    """Fake backend: records hook calls, answers every content query."""

    name = "recording"
    autocue = False

    def __init__(self):
        self.observed: list[tuple[str, int]] = []
        self.boundaries: list[tuple[str, str, int]] = []
        self.queries: list = []

    async def observe(self, session_id, event):
        self.observed.append((session_id, event.seq))

    async def boundary(self, session_id, reason, upto_seq):
        self.boundaries.append((session_id, reason, upto_seq))

    async def query(self, session_id, query, events):
        from server.mind.mem import MemSpan

        self.queries.append(query)
        return [MemSpan(seq=1, role="user", text="canned span", score=1.0, backend=self.name)]


def _fake_embedder(mem):
    """Deterministic 2-d 'semantics': reptile-ish text points one way,
    everything else the other. Patches mem._embed."""

    async def embed(texts):
        import numpy as np

        out = []
        for text in texts:
            low = text.lower()
            if any(w in low for w in ("tortoise", "reptile", "muriel")):
                vec = [1.0, 0.0, 0.0]
            elif "garden" in low:
                vec = [0.0, 1.0, 0.0]
            else:
                vec = [0.0, 0.0, 1.0]
            out.append(np.asarray(vec, dtype=np.float32))
        return out

    mem._embed = embed
    return mem


def test_mem_backend_swap_and_trajectory(store, tmp_path):
    import asyncio

    from server.mind.runtime import MindRuntime

    runtime = MindRuntime(config(db_dir=str(tmp_path / "minds")))
    runtime.mem = _RecordingMem()

    async def go():
        transcript = [user("first question")]
        prepared = await runtime.prepare(transcript, provider=None, model="m")
        # observe fired once per confirmed event, no duplicates on re-prepare
        assert runtime.mem.observed == [(prepared.session_id, 1)]
        transcript += [assistant("an answer"), user("second question")]
        await runtime.prepare(transcript, provider=None, model="m")
        assert [seq for _, seq in runtime.mem.observed] == [1, 2, 3]

        out = await runtime.resolve_recall(prepared.session_id, '{"query": "anything at all"}')
        assert "canned span" in out and "[seq 1, user, verbatim]" in out
        assert runtime.mem.queries[-1].trajectory  # trajectory stub supplied
        assert runtime.mem.queries[-1].kind == "content"

    asyncio.run(go())


def test_mem_boundary_fires_on_fold(store, tmp_path, monkeypatch):
    import asyncio

    from server.mind import runtime as runtime_module
    from server.mind.runtime import MindRuntime

    # Tiny window: the texture must genuinely overflow — folds are a
    # pressure response, never a default.
    cfg = config(
        db_dir=str(tmp_path / "minds"),
        window=256,
        summary_trigger_tokens=10,
        min_keep_turns=1,
    )
    runtime = MindRuntime(cfg)
    runtime.mem = _RecordingMem()

    async def fake_steward(config, store_, session_id, events, provider, model, upto, now=None):
        store_.append_summary(session_id, upto, "condensed")

    monkeypatch.setattr(runtime_module, "run_steward", fake_steward)

    async def go():
        transcript = [user("a rather long opening message about many topics " * 30)]
        prepared = await runtime.prepare(transcript, provider=None, model="m")
        transcript += [assistant("reply " * 120), user("next")]
        await runtime.prepare(transcript, provider=None, model="m")
        assert runtime.mem.boundaries, "fold should signal a Mem boundary"
        session_id, reason, upto = runtime.mem.boundaries[0]
        assert session_id == prepared.session_id and reason == "fold" and upto >= 1

    asyncio.run(go())


def test_lexical_mem_matches_v3_ranking(store):
    import asyncio

    from server.mind.mem import LexicalMem, MemQuery

    sid = store.create_session(None)
    store.append_event(sid, user("the tortoise Muriel lived in the stairwell"), source="client")
    store.append_event(sid, user("completely unrelated gardening chatter"), source="client")
    events = store.live_events(sid)
    spans = asyncio.run(LexicalMem().query(sid, MemQuery(text="Muriel stairwell tortoise"), events))
    assert spans and spans[0].seq == 1 and spans[0].backend == "lexical"
    # order queries: honest empty, never a guess
    assert asyncio.run(
        LexicalMem().query(sid, MemQuery(text="x", kind="next", anchor_seq=1), events)
    ) == []


def test_embedding_mem_semantic_reach_and_order(store, tmp_path):
    import asyncio

    from server.mind.mem import EmbeddingMem, MemQuery

    cfg = config(db_dir=str(tmp_path / "minds"))
    mem = _fake_embedder(EmbeddingMem(cfg, store))
    sid = store.create_session(None)
    store.append_event(sid, user("the tortoise Muriel lived in the stairwell"), source="client")
    store.append_event(sid, user("completely unrelated gardening chatter"), source="client")
    events = store.live_events(sid)

    async def go():
        for event in events:
            await mem.observe(sid, event)
        # zero lexical overlap, pure semantic hit — the s13 probe-A shape
        spans = await mem.query(sid, MemQuery(text="who was the slow reptile?"), events)
        assert spans and spans[0].seq == 1
        assert spans[0].sim >= cfg.embed_min_sim
        assert "Muriel" in spans[0].text
        # hybrid: lexical-only material still reachable (verbatim words)
        spans = await mem.query(sid, MemQuery(text="gardening chatter"), events)
        assert spans and spans[0].seq == 2
        # order comes from the log
        nxt = await mem.query(sid, MemQuery(text="", kind="next", anchor_seq=1), events)
        assert [span.seq for span in nxt] == [2]

    asyncio.run(go())


def test_embedding_mem_fails_open_to_lexical(store, tmp_path):
    import asyncio

    from server.mind.mem import EmbeddingMem, MemQuery

    cfg = config(db_dir=str(tmp_path / "minds"))
    mem = EmbeddingMem(cfg, store)

    async def dead_embed(texts):
        return None

    mem._embed = dead_embed
    sid = store.create_session(None)
    store.append_event(sid, user("the tortoise Muriel lived in the stairwell"), source="client")
    events = store.live_events(sid)

    async def go():
        await mem.observe(sid, events[0])  # embed fails; must not raise
        spans = await mem.query(sid, MemQuery(text="Muriel stairwell"), events)
        assert spans and spans[0].seq == 1 and spans[0].sim == 0.0  # lexical carried it

    asyncio.run(go())


def test_autocue_injects_folded_semantic_spans(store, tmp_path):
    import asyncio

    from server.mind.mem import EmbeddingMem
    from server.mind.runtime import MindRuntime

    cfg = config(db_dir=str(tmp_path / "minds"))
    runtime = MindRuntime(cfg)
    runtime.mem = _fake_embedder(EmbeddingMem(cfg, runtime.store))

    async def go():
        transcript = [
            user("the tortoise Muriel lived in the stairwell"),
            assistant("noted"),
            user("tell me about gardening instead"),
        ]
        prepared = await runtime.prepare(transcript, provider=None, model="m")
        # fold the tortoise turn out of view, then probe semantically
        runtime.store.append_summary(prepared.session_id, 2, "earlier chatter")
        transcript += [assistant("gardening is nice"), user("who was the slow reptile?")]
        prepared2 = await runtime.prepare(transcript, provider=None, model="m")
        system_text = prepared2.messages[0]["content"]
        assert "Recalled verbatim" in system_text
        assert "Muriel" in system_text
        # on-topic lexical matches must NOT be auto-injected (sim filter):
        transcript += [assistant("it was Muriel"), user("more about gardening chatter please")]
        prepared3 = await runtime.prepare(transcript, provider=None, model="m")
        assert "Recalled verbatim" not in prepared3.messages[0]["content"]

    asyncio.run(go())


def test_create_mem_backend_fails_open_to_lexical():
    from server.mind.mem import LexicalMem, create_mem_backend

    assert isinstance(create_mem_backend("lexical"), LexicalMem)
    assert isinstance(create_mem_backend("no-such-backend"), LexicalMem)


def test_truncation_stop_button(store):
    recon1, reply_seq = play_turn(store, [user("explain quantum physics")], "It is a long story about particles and waves")

    # Client stopped us mid-reply and kept only a prefix, then asked again.
    truncated = assistant("It is a long story")
    transcript = [user("explain quantum physics"), truncated, user("shorter please")]
    recon2 = reconcile(store, transcript)
    assert recon2.outcome == "truncation"
    assert recon2.session_id == recon1.session_id

    events = store.live_events(recon2.session_id)
    contents = [e.message.get("content") for e in events]
    assert "It is a long story" in contents          # the truncation is truth
    assert "It is a long story about particles and waves" not in contents  # superseded


def test_fork_on_edited_turn(store):
    recon1, _ = play_turn(store, [user("q1")], "a1")
    store_events = store.live_events(recon1.session_id)
    assert len(store_events) == 2

    # Client continued once...
    transcript = [user("q1"), assistant("a1"), user("q2")]
    play_turn(store, transcript, "a2")

    # ...then edited q2 into q2-edited (fork at the shared 2-message prefix).
    forked = [user("q1"), assistant("a1"), user("q2-edited")]
    recon3 = reconcile(store, forked)
    assert recon3.outcome == "fork"
    assert recon3.session_id == recon1.session_id
    live = [e.message.get("content") for e in store.live_events(recon3.session_id)]
    assert live == ["q1", "a1", "q2-edited"]


def test_regenerate_last_reply(store):
    recon1, _ = play_turn(store, [user("q1")], "first draft answer")
    # Client regenerated: same prefix, different assistant reply retained.
    transcript = [user("q1"), assistant("second draft answer"), user("q2")]
    recon2 = reconcile(store, transcript)
    assert recon2.session_id == recon1.session_id
    assert recon2.outcome == "fork"
    live = [e.message.get("content") for e in store.live_events(recon2.session_id)]
    assert live == ["q1", "second draft answer", "q2"]


def test_regenerate_request_drops_tail_not_session(store):
    """The regenerate REQUEST itself: client resends history minus our last
    reply, nothing new after it. Must stay in-session with the tail dropped,
    not spawn a fresh mind (live bug found designing s7)."""
    play_turn(store, [user("q1")], "a1")
    recon1, _ = play_turn(store, [user("q1"), assistant("a1"), user("q2")], "first draft")

    regenerate_request = [user("q1"), assistant("a1"), user("q2")]
    recon2 = reconcile(store, regenerate_request)
    assert recon2.session_id == recon1.session_id
    assert recon2.outcome == "fork"
    assert recon2.new_events == []
    live = [e.message.get("content") for e in store.live_events(recon2.session_id)]
    assert live == ["q1", "a1", "q2"]  # first draft superseded, prefix intact

    # Our second draft gets appended and the client retains it: continue.
    seq = store.append_event(recon2.session_id, assistant("second draft"), source="mind")
    recon3 = reconcile(store, regenerate_request + [assistant("second draft"), user("q3")])
    assert recon3.outcome == "continue"
    assert recon3.session_id == recon1.session_id
    live = [e.message.get("content") for e in store.live_events(recon3.session_id)]
    assert live == ["q1", "a1", "q2", "second draft", "q3"]


def test_unrelated_transcript_is_new_session(store):
    play_turn(store, [user("about cats")], "meow")
    recon = reconcile(store, [user("about dogs")])
    assert recon.outcome == "new"
    assert len(store.list_session_ids()) == 2


def test_supersede_does_not_eat_replacement(store):
    session = store.create_session(None)
    store.append_event(session, user("a"), source="client")
    store.append_event(session, assistant("b"), source="mind")
    replacement = store.append_event(session, assistant("b-truncated"), source="client")
    store.supersede_from(session, 2, replacement)
    live = [e.message.get("content") for e in store.live_events(session)]
    assert live == ["a", "b-truncated"]


def test_assembler_respects_budget_and_blocks(store):
    cfg = config(window=4096)
    session = store.create_session("sys")
    long = "x " * 900  # ~450 tokens each
    for i in range(12):
        store.append_event(session, user(f"question {i} {long}"), source="client")
        store.append_event(session, assistant(f"answer {i} {long}"), source="mind")
    events = store.live_events(session)
    summary = (events[-6].seq, "summary of earlier turns")

    workspace = assemble(cfg, "sys", events, summary)
    budget = cfg.window - max(int(cfg.window * cfg.reserve_fraction), 1024)
    assert workspace.estimated_tokens <= budget
    # Newest event always present; system carries client prompt + summary.
    assert workspace.messages[0]["role"] == "system"
    assert "sys" in workspace.messages[0]["content"]
    assert "summary of earlier turns" in workspace.messages[0]["content"]
    assert workspace.messages[-1]["content"] == events[-1].message["content"]


def test_assembler_never_orphans_tool_results(store):
    cfg = config(window=4096)
    session = store.create_session(None)
    store.append_event(session, user("start"), source="client")
    store.append_event(
        session,
        {"role": "assistant", "tool_calls": [{"id": "c1", "type": "function",
            "function": {"name": "t", "arguments": "{}"}}]},
        source="mind",
    )
    store.append_event(session, {"role": "tool", "tool_call_id": "c1", "content": "big " * 400}, source="client")
    store.append_event(session, user("and now?"), source="client")
    events = store.live_events(session)

    workspace = assemble(cfg, None, events, (0, ""))
    roles = [m["role"] for m in workspace.messages]
    # If the tool result is present, its assistant tool_calls must precede it.
    if "tool" in roles:
        assert roles.index("tool") == roles.index("assistant") + 1


def test_texture_always_opens_on_user_turn(store):
    cfg = config(window=4096)
    session = store.create_session(None)
    long = "y " * 700
    for i in range(10):
        store.append_event(session, user(f"q{i} {long}"), source="client")
        store.append_event(session, assistant(f"a{i} {long}"), source="mind")
    events = store.live_events(session)
    # Summary boundary deliberately placed AFTER a user turn (worst case).
    for upto in (3, 4, 5, 12, 13):
        workspace = assemble(cfg, None, events, (upto, "earlier turns summary"))
        non_system = [m for m in workspace.messages if m["role"] != "system"]
        assert non_system[0]["role"] == "user", f"upto={upto} opened on {non_system[0]['role']}"


def test_ledger_generation_versioning(store):
    session = store.create_session(None)
    store.replace_records(session, {"fact": [{"subject": "port", "claim": "8080"}]}, provenance_seq=2)
    store.replace_records(
        session,
        {"fact": [{"subject": "port", "claim": "9090 (was 8080)"}],
         "commitment": [{"actor": "user", "statement": "rotate token", "status": "open"}]},
        provenance_seq=6,
    )
    live = store.live_records(session)
    assert live["fact"] == [{"subject": "port", "claim": "9090 (was 8080)"}]
    assert len(live["commitment"]) == 1


def test_fork_invalidates_derived_records(store):
    session = store.create_session(None)
    store.append_event(session, user("q1"), source="client")
    store.append_event(session, assistant("a1"), source="mind")
    store.append_event(session, user("q2"), source="client")
    store.replace_records(session, {"fact": [{"subject": "x", "claim": "y"}]}, provenance_seq=3)
    store.append_episode(session, 1, 3, "early events")
    replacement = store.append_event(session, user("q2-edited"), source="client")
    store.supersede_from(session, 3, replacement)
    assert store.live_records(session) == {}
    assert store.live_episodes(session) == []


def test_render_memory_sections():
    from server.mind.assembler import render_memory

    text = render_memory(
        "",
        {
            "decision": [{"topic": "sync", "status": "leaning", "choice": "Turnstile", "reason": "simple"}],
            "commitment": [
                {"actor": "user", "statement": "rotate token", "trigger": "deployment", "status": "open"},
                {"actor": "user", "statement": "old thing", "status": "done"},
            ],
            "fact": [{"subject": "port", "claim": "9090 (was 8080)"}],
        },
        [(1, 6, "planned the app")],
    )
    assert "sync: LEANING (not yet decided) — Turnstile (reason: simple)" in text
    assert "rotate token — trigger: deployment" in text
    assert "old thing" not in text  # done commitments stay out of the open list
    assert "port: 9090 (was 8080)" in text
    assert "planned the app" in text
    assert render_memory("", {}, []) == ""  # empty memory renders nothing


# -- v2 CRS dynamics ----------------------------------------------------------


def make_thread(key, summary, **kw):
    from server.mind.dynamics import ThreadState

    return ThreadState(key=key, kind=kw.pop("kind", "topic"), summary=summary, **kw)


def test_dynamics_decay_and_input():
    from server.mind.dynamics import ACTIVE_THRESHOLD, update_dynamics

    logging = make_thread(
        "logging-pipeline",
        "Designing a vector to Loki logging pipeline for five Raspberry Pis.",
        anchors=["vector", "loki", "journald", "grafana"],
    )
    aside = make_thread(
        "dumpling-aside",
        "Grandmother's plum dumpling trick: semolina instead of flour.",
        kind="aside",
        anchors=["semolina", "plum", "dumplings", "grandmother"],
    )
    threads = [logging, aside]
    for message in [
        "What labels should I put on the loki streams?",
        "Now the retention policy for loki, 30 days hot.",
        "One Pi is on flaky wifi, what happens to logs when the link drops?",
        "Configure that vector disk buffer, 200 MB cap.",
        "Add a grafana alert for any Pi silent 15 minutes.",
    ]:
        update_dynamics(threads, message)
    # The working topic stays admitted; the untouched aside decays out.
    assert logging.activation >= ACTIVE_THRESHOLD
    assert aside.activation < ACTIVE_THRESHOLD
    # Decay is not deletion: importance floor keeps it retrievable, not zero.
    assert aside.importance > 0.0


def test_dynamics_question_attractor():
    from server.mind.dynamics import update_dynamics

    inquiry = make_thread(
        "retention-question",
        "Open question about archival.",
        kind="inquiry",
        open_questions=["Should thread importance decay over time?"],
        activation=0.2,
    )
    control = make_thread("control", "Open question about archival.", activation=0.2)
    update_dynamics([inquiry, control], "I think importance should decay over time, yes.")
    assert inquiry.activation > control.activation + 0.2  # attractor fired


def test_cued_recall_of_dormant_thread():
    from server.mind.dynamics import cued_threads

    aside = make_thread(
        "dumpling-aside",
        "Grandmother's plum dumpling trick: semolina instead of flour.",
        kind="aside",
        anchors=["semolina", "plum dumplings", "grandmother"],
        activation=0.05,
    )
    # Generic on-topic message must NOT cue it...
    assert cued_threads([aside], "anything else worth batching or buffering?", []) == []
    # ...but an explicit reach for it must.
    cue = "what was that trick from my grandmother's recipe I mentioned way back?"
    assert [t.key for t in cued_threads([aside], cue, [])] == ["dumpling-aside"]


def test_cued_recall_survives_chatty_cue_live_repro():
    """Regression: run 20260706-101717 missed the s5 cue by a coverage hair.

    Thread content is the VERBATIM steward output from that run; the cue is
    the VERBATIM s5 turn-15 message. Chatty cues dilute coverage (9 content
    words, 2 hits), and possessives must collapse (grandmother's ->
    grandmother) on both sides.
    """
    from server.mind.dynamics import cued_threads

    aside = make_thread(
        "plum-dumplings",
        "A brief mention of a grandmother's trick for making lighter plum "
        "dumplings using semolina.",
        kind="aside",
        anchors=["plum dumplings", "semolina"],
        activation=0.05,
        facts=[
            {
                "subject": "plum_dumpling_trick",
                "thread": "plum-dumplings",
                "claim": "Use semolina instead of flour in the dough to make them lighter.",
            }
        ],
    )
    cue = (
        "Totally different thing before I go cook — what was that trick "
        "from my grandmother's recipe I mentioned way back?"
    )
    assert [t.key for t in cued_threads([aside], cue, [])] == ["plum-dumplings"]
    # The two on-topic probes around it must still not cue the aside.
    for message in (
        "We're batching sinks now. Open question: anything else in this "
        "pipeline you think is worth batching or buffering that we haven't covered?",
        "Right! Okay, one-paragraph summary of what we designed today.",
    ):
        assert cued_threads([aside], message, []) == []


def test_render_memory_thread_gating():
    from server.mind.assembler import ThreadsView, render_memory

    aside = make_thread(
        "dumpling-aside",
        "Plum dumpling trick.",
        kind="aside",
        facts=[{"subject": "dumpling_trick", "thread": "dumpling-aside", "claim": "semolina not flour"}],
    )
    records = {
        "fact": [
            {"subject": "dumpling_trick", "thread": "dumpling-aside", "claim": "semolina not flour"},
            {"subject": "log_volume", "thread": "logging-pipeline", "claim": "50 MB/day"},
            {"subject": "threadless", "claim": "always renders"},
        ],
        "commitment": [{"actor": "user", "statement": "rotate token", "status": "open"}],
    }
    logging = make_thread("logging-pipeline", "Vector to Loki pipeline.", activation=0.9)
    view = ThreadsView(admitted=[logging], cued=[], all_keys={"logging-pipeline", "dumpling-aside"})
    text = render_memory("", records, [], view)
    assert "log_volume" in text                       # admitted thread's fact renders
    assert "threadless: always renders" in text       # no thread -> safe fallback
    assert "semolina" not in text                     # dormant thread's fact gated out
    assert "Active threads" in text and "logging-pipeline" in text
    assert "rotate token" in text                     # commitments always render

    # Cue the aside back in: fact returns under the Recalled section.
    view = ThreadsView(admitted=[logging], cued=[aside], all_keys={"logging-pipeline", "dumpling-aside"})
    text = render_memory("", records, [], view)
    assert "Recalled" in text and "semolina not flour" in text


def test_store_threads_versioning_and_fork(store):
    sid = store.create_session(None)
    store.append_event(sid, user("q1"), source="client")
    store.replace_threads(sid, [{"key": "alpha", "summary": "one"}], provenance_seq=1)
    store.replace_threads(
        sid,
        [{"key": "alpha", "summary": "updated"}, {"key": "beta", "summary": "two"}],
        provenance_seq=1,
    )
    live = store.live_threads(sid)
    assert {t["key"] for t in live} == {"alpha", "beta"}
    assert [t["summary"] for t in live if t["key"] == "alpha"] == ["updated"]

    store.set_dynamics(sid, "alpha", 0.7, 0.4, updated_seq=1)
    store.set_dynamics(sid, "alpha", 0.6, 0.41, updated_seq=2)  # upsert, not append
    assert store.get_dynamics(sid)["alpha"] == (0.6, 0.41, 2)

    # Fork before the provenance seq invalidates the thread structure.
    for i in range(2, 6):
        store.append_event(sid, user(f"q{i}"), source="client")
    store.supersede_from(sid, from_seq=1, by_seq=5)
    assert store.live_threads(sid) == []


def test_steward_chunks_oversized_folds(store):
    """A huge fold span (post-fork re-fold) must become several capped
    steward passes, each advancing the watermark, never one giant call."""
    import asyncio

    from server.mind.steward import run_steward

    sid = store.create_session(None)
    events = []
    for i in range(1, 13):
        role = "user" if i % 2 else "assistant"
        seq = store.append_event(sid, {"role": role, "content": f"msg{i} " + "x" * 1200}, source="client")
    events = store.live_events(sid)

    calls = []

    class Provider:
        async def chat_completions(self, payload):
            calls.append(payload["messages"][1]["content"])
            return {
                "choices": [
                    {
                        "message": {
                            "content": (
                                '{"threads": [], "facts": [{"subject": "s", "claim": "c"}],'
                                ' "decisions": [], "commitments": [],'
                                ' "episode": "things happened"}'
                            )
                        }
                    }
                ]
            }

    cfg = config(steward_input_tokens=900)
    asyncio.run(run_steward(cfg, store, sid, events, Provider(), "m", upto_seq=12))
    assert len(calls) > 1  # ~12*300 est tokens vs 900 cap -> several passes
    for content in calls:
        # No single pass may carry more transcript than roughly the cap.
        transcript = content.split("New turns:\n", 1)[1]
        assert len(transcript) // 4 < 900 + 400  # cap + one-event slack
    episodes = store.live_episodes(sid)
    assert len(episodes) == len(calls)
    assert episodes[-1][1] == 12  # watermark reached the requested boundary


# -- v3: recall + tool router -------------------------------------------------


TOOLBELT = [
    {"type": "function", "function": {"name": f"tool_{i}", "description": f"does thing {i}"}}
    for i in range(8)
]


def test_router_prunes_chat_turns_and_keeps_tool_turns(store):
    from server.mind.router import scope_tools

    sid = store.create_session(None)
    store.append_event(sid, user("hello"), source="client")
    events = store.live_events(sid)

    # Chat turn: full belt pruned to nothing.
    assert scope_tools(TOOLBELT, events, "what's a good patch-day order?") == []
    # Imperative probe verb: full belt forwarded.
    assert scope_tools(TOOLBELT, events, "Check the containers please") == TOOLBELT
    assert scope_tools(TOOLBELT, events, "Find out what's eating the space") == TOOLBELT
    # Tool talk ("use", "tool") forwards the whole belt: a false negative
    # makes the model play-act calls, a false positive only costs schema tax.
    assert scope_tools(TOOLBELT, events, "use tool 3 on it") == TOOLBELT
    # Small packs are never pruned.
    assert scope_tools(TOOLBELT[:2], events, "chit chat") == TOOLBELT[:2]


def test_router_keeps_belt_while_tools_in_flight(store):
    from server.mind.router import scope_tools

    sid = store.create_session(None)
    store.append_event(sid, user("check it"), source="client")
    store.append_event(
        sid,
        {"role": "assistant", "tool_calls": [{"id": "c1", "type": "function",
            "function": {"name": "tool_1", "arguments": "{}"}}]},
        source="mind",
    )
    store.append_event(sid, {"role": "tool", "tool_call_id": "c1", "content": "out"}, source="client")
    events = store.live_events(sid)
    # Follow-up turn right after tool traffic: model may need the belt again.
    assert scope_tools(TOOLBELT, events, "hmm, and what does that mean?") == TOOLBELT


def test_recall_returns_verbatim_span(store):
    import asyncio
    import json as _json

    from server.mind.recall import resolve_recall as _async_resolve

    def resolve_recall(events, args):
        return asyncio.run(_async_resolve(events, args))

    sid = store.create_session(None)
    traceback = (
        "Traceback (most recent call last):\n  File \"ingest.py\", line 214\n"
        "sqlite3.OperationalError: no such column: photos.review_flag_v2 [ref: ingest-9f4c2e71]"
    )
    store.append_event(sid, user("my app broke:\n" + traceback), source="client")
    store.append_event(sid, assistant("the migration didn't apply"), source="mind")
    for i in range(6):
        store.append_event(sid, user(f"unrelated question {i} about the garden"), source="client")
        store.append_event(sid, assistant(f"answer {i}"), source="mind")
    events = store.live_events(sid)

    out = resolve_recall(events, _json.dumps({"query": "traceback review_flag error line"}))
    assert "photos.review_flag_v2 [ref: ingest-9f4c2e71]" in out  # verbatim
    assert "[seq 1, user, verbatim]" in out                       # provenance
    assert "recall error" in resolve_recall(events, "{}")
    assert "nothing found" in resolve_recall(events, '{"query": "xylophone zeppelin"}')


def test_prepare_scopes_tools_and_offers_recall(store, tmp_path):
    import asyncio

    from server.mind.runtime import MindRuntime

    cfg = config(db_dir=str(tmp_path / "minds"))
    runtime = MindRuntime(cfg)

    async def go():
        transcript = [user("let's plan the garden layout")]
        prepared = await runtime.prepare(transcript, provider=None, model="m", tools=TOOLBELT)
        # Chat turn: tools pruned; no fold yet -> no recall offered.
        assert prepared.tools_scoped and prepared.tools == []

        # Simulate folded material: summary watermark present.
        runtime.store.append_event(prepared.session_id, assistant("plan drafted"), source="mind")
        runtime.store.append_summary(prepared.session_id, 1, "earlier stuff")
        transcript += [assistant("plan drafted"), user("now check the beds status")]
        prepared2 = await runtime.prepare(transcript, provider=None, model="m", tools=TOOLBELT)
        names = [t["function"]["name"] for t in prepared2.tools]
        assert "recall" in names            # offered once memory is folded
        assert "tool_0" in names            # probe verb "check" keeps the belt
        assert prepared2.session_id == prepared.session_id

    asyncio.run(go())


def test_stale_tool_payloads_render_as_digests(store):
    # Small window so the texture is genuinely over budget: digestion is a
    # pressure response, never a default.
    cfg = config(window=2048, tool_digest_chars=300)
    session = store.create_session(None)
    payload = "LOG " * 800  # ~3200 chars of tool output
    store.append_event(session, user("check the service"), source="client")
    store.append_event(
        session,
        {"role": "assistant", "tool_calls": [{"id": "c1", "type": "function",
            "function": {"name": "read", "arguments": "{}"}}]},
        source="mind",
    )
    store.append_event(session, {"role": "tool", "tool_call_id": "c1", "content": payload}, source="client")
    store.append_event(session, assistant("service looks fine"), source="mind")
    store.append_event(session, user("and now a new question"), source="client")
    events = store.live_events(session)

    workspace = assemble(cfg, None, events, (0, ""))
    tool_messages = [m for m in workspace.messages if m.get("role") == "tool"]
    assert tool_messages, "stale tool message still present as a digest"
    assert len(tool_messages[0]["content"]) < 600
    assert "folded away" in tool_messages[0]["content"]
    # The event store keeps the full payload (recall's ground truth).
    stored = [e for e in store.live_events(session) if e.role == "tool"][0]
    assert len(stored.message["content"]) > 3000


def test_current_turn_tool_payload_stays_verbatim(store):
    cfg = config(window=2048, tool_digest_chars=300)
    session = store.create_session(None)
    payload = "DATA " * 700
    store.append_event(session, user("read the big file"), source="client")
    store.append_event(
        session,
        {"role": "assistant", "tool_calls": [{"id": "c1", "type": "function",
            "function": {"name": "read", "arguments": "{}"}}]},
        source="mind",
    )
    store.append_event(session, {"role": "tool", "tool_call_id": "c1", "content": payload}, source="client")
    events = store.live_events(session)

    workspace = assemble(cfg, None, events, (0, ""))
    tool_messages = [m for m in workspace.messages if m.get("role") == "tool"]
    # In-flight exchange: the model needs the full payload to answer NOW.
    assert tool_messages and tool_messages[0]["content"] == payload


def test_no_digestion_without_budget_pressure(store):
    cfg = config(window=8192, tool_digest_chars=300)
    session = store.create_session(None)
    payload = "LOG " * 800
    store.append_event(session, user("check the service"), source="client")
    store.append_event(
        session,
        {"role": "assistant", "tool_calls": [{"id": "c1", "type": "function",
            "function": {"name": "read", "arguments": "{}"}}]},
        source="mind",
    )
    store.append_event(session, {"role": "tool", "tool_call_id": "c1", "content": payload}, source="client")
    store.append_event(session, assistant("looks fine"), source="mind")
    store.append_event(session, user("next question"), source="client")
    events = store.live_events(session)

    workspace = assemble(cfg, None, events, (0, ""))
    tool_messages = [m for m in workspace.messages if m.get("role") == "tool"]
    # Everything fits at 8k: the stale payload must stay verbatim (s2-t5's
    # evidence must not be digested when the budget could carry it).
    assert tool_messages and tool_messages[0]["content"] == payload


# -- chronos (v4) -----------------------------------------------------------


def test_event_ts_roundtrip(store):
    from datetime import datetime

    session = store.create_session(None)
    stamp = datetime(2026, 3, 14, 10, 3).timestamp()
    store.append_event(session, user("fake-clock turn"), source="client", ts=stamp)
    store.append_event(session, assistant("real-clock reply"), source="mind")
    events = store.live_events(session)
    assert events[0].ts == stamp
    # No explicit ts -> the store's real clock, not zero.
    assert events[1].ts > stamp


def test_reconcile_stamps_fake_clock(store):
    stamp = 1_700_000_000.0
    recon = reconcile(store, [user("hello")], now=stamp)
    events = store.live_events(recon.session_id)
    assert [event.ts for event in events] == [stamp]


def test_format_gap_units():
    from server.mind.assembler import format_gap

    assert format_gap(120) == "2 minutes"
    assert format_gap(9000) == "2.5 hours"
    assert format_gap(14400) == "4 hours"
    assert format_gap(3 * 86400) == "3 days"


def test_gap_marker_rendered_on_user_turn(store):
    from datetime import datetime

    cfg = config(window=8192)
    session = store.create_session(None)
    t0 = datetime(2026, 3, 14, 10, 26).timestamp()
    store.append_event(session, user("heading to the garden"), source="client", ts=t0)
    store.append_event(session, assistant("enjoy!"), source="mind", ts=t0)
    store.append_event(session, user("ok I'm back"), source="client", ts=t0 + 14400)
    events = store.live_events(session)

    workspace = assemble(cfg, None, events, None, now=t0 + 14400)
    back = [m for m in workspace.messages if m.get("role") == "user"][-1]
    assert back["content"].startswith(
        "[4 hours pass — it is now Saturday 2026-03-14 14:26]\n\n"
    )
    # The event store itself is untouched — markers live only in the render.
    assert store.live_events(session)[-1].message["content"] == "ok I'm back"


def test_no_gap_marker_below_threshold(store):
    cfg = config(window=8192, gap_mark_minutes=30)
    session = store.create_session(None)
    store.append_event(session, user("first"), source="client", ts=1000.0)
    store.append_event(session, assistant("ok"), source="mind", ts=1001.0)
    store.append_event(session, user("second, five minutes later"), source="client", ts=1300.0)
    events = store.live_events(session)

    workspace = assemble(cfg, None, events, None, now=1300.0)
    contents = [m["content"] for m in workspace.messages if m.get("role") == "user"]
    assert not any(content.startswith("[") for content in contents)


def test_now_section_and_due_status():
    from datetime import datetime

    from server.mind.assembler import render_memory

    now = datetime(2026, 3, 14, 14, 26).timestamp()
    records = {
        "commitment": [
            {
                "actor": "user",
                "statement": "punch down the dough",
                "trigger": "two hours after 10:03",
                "due": "2026-03-14T12:03",
                "status": "open",
            },
            {
                "actor": "user",
                "statement": "water the rhubarb",
                "trigger": "this evening",
                "due": "2026-03-14T18:00",
                "status": "open",
            },
        ]
    }
    text = render_memory("", records, [], now=now)
    assert "### Now" in text and "Saturday 2026-03-14 14:26" in text
    assert "OVERDUE by 2.4 hours" in text and "raise this NOW" in text
    assert "due Saturday 2026-03-14 18:00 (in 3.6 hours)" in text
    # Garbage due values degrade to the plain trigger, never crash.
    records["commitment"][0]["due"] = "when the cows come home"
    assert "punch down the dough" in render_memory("", records, [], now=now)


def test_no_clock_renders_no_time_surfaces():
    from server.mind.assembler import render_memory

    records = {
        "commitment": [
            {"actor": "user", "statement": "x", "due": "2026-03-14T12:03", "status": "open"}
        ]
    }
    text = render_memory("", records, [], now=None)
    assert "### Now" not in text and "OVERDUE" not in text


def test_fresh_session_gets_bare_time_line_not_memory_theater():
    from datetime import datetime

    from server.mind.assembler import MIND_HEADER, render_memory

    now = datetime(2026, 3, 14, 14, 26).timestamp()
    text = render_memory("", {}, [], now=now)
    assert text == "Current time: Saturday 2026-03-14 14:26."
    assert MIND_HEADER not in text


def test_router_forwards_belt_on_tool_talk_live_repro():
    """Live failure (PI + xwiki MCP adapter, 2026-07-12): 'use the ... tools
    to tell me ...' matched no probe verb, the belt was stripped, and gemma
    play-acted the call as text then confabulated the page content."""
    from server.mind.router import scope_tools

    belt = [
        {"type": "function", "function": {"name": name, "description": "xwiki", "parameters": {}}}
        for name in (
            "xwiki_get_document",
            "xwiki_search",
            "xwiki_create_document",
            "xwiki_list_spaces",
            "xwiki_update_document",
        )
    ]
    text = (
        "please use the xwiki mpc tools available to tell me the content "
        "from the page with ref Sandbox.TestPage3"
    )
    assert scope_tools(belt, [], text) == belt
    # Even without the words 'use'/'tools', naming the domain is enough.
    assert scope_tools(belt, [], "what's on the xwiki sandbox page?") == belt


def test_router_still_prunes_pure_chat_with_fat_belt():
    from server.mind.router import scope_tools

    belt = [
        {"type": "function", "function": {"name": name, "description": "d", "parameters": {}}}
        for name in ("disk_usage", "docker_ps", "query_metrics", "dns_lookup")
    ]
    assert scope_tools(belt, [], "talk me through a sane monthly checklist for my homelab") == []
    assert scope_tools(belt, [], "should I move my reverse proxy from nginx to caddy?") == []


def test_content_parts_arrays_are_first_class():
    """Live failure #2 (PI client, 2026-07-19): PI sends OpenAI content-parts
    arrays, every organ read content as str-or-nothing, so the router saw an
    EMPTY user text and stripped the belt regardless of its trigger rules."""
    from server.mind.router import scope_tools
    from server.mind.store import content_text

    parts_message = {
        "role": "user",
        "content": [{"type": "text", "text": "use the xwiki mpc tools available "
                     "to tell me the content from the page with ref Sandbox.TestPage3"}],
    }
    assert content_text(parts_message).startswith("use the xwiki")
    assert content_text({"role": "user", "content": "plain"}) == "plain"
    assert content_text({"role": "assistant", "content": None}) is None
    assert content_text({"role": "user", "content": [{"type": "image_url", "image_url": {}}]}) is None


def test_runtime_router_sees_parts_array_text(store, tmp_path):
    import asyncio

    from server.mind.config import MindConfig
    from server.mind.runtime import MindRuntime

    runtime = MindRuntime(MindConfig(enabled=True, db_dir=str(tmp_path), window=8192))
    belt = [
        {"type": "function", "function": {"name": f"xwiki_op_{i}", "description": "d", "parameters": {}}}
        for i in range(5)
    ]
    messages = [
        {"role": "user", "content": [{"type": "text", "text": "use the xwiki tools to "
          "tell me the content of the page Sandbox.TestPage3"}]},
    ]

    class NoProvider:
        async def chat_completions(self, payload):  # steward never fires here
            raise AssertionError("no fold expected")

    prepared = asyncio.run(
        runtime.prepare(messages, NoProvider(), "m", tools=belt)
    )
    # Full belt forwarded: either untouched (tools_scoped False) or scoped
    # with every xwiki tool still present. Before the fix, the router saw an
    # empty user text and scoped the belt down to [].
    if prepared.tools_scoped:
        forwarded = [t["function"]["name"] for t in prepared.tools or []]
        assert [n for n in forwarded if n.startswith("xwiki_op_")] == [
            f"xwiki_op_{i}" for i in range(5)
        ]
