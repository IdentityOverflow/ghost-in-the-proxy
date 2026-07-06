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
