"""Steward (memory v6): delta extraction into the fold log.

Runs when eviction pressure says a span must leave verbatim view. One LLM
call per chunk proposes small OPERATIONS against runtime-issued ids plus one
episode line; the runtime validates per-op and commits the fold atomically
(docs/memory-v6.md §2). The v1 steward re-emitted the complete ledger every
fold — a telephone game whose input and output both grew with the
conversation until the call truncated. Here both are bounded: the model
sees a relevance-ranked SLICE of the ledger and writes only what changed.

The LLM proposes; the runtime disposes (CRS §19). If the proposal is
unusable the span still folds — as a plain-text episode with no ops — so
coverage advances and the ledger is simply unchanged, never rewritten.
"""

import json
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from .assembler import estimate_tokens
from .config import MindConfig
from .ledger import ApplyReport, LedgerState, Record, apply_ops, normalize_key, replay
from .mem import MemBackend
from .relevance import score_texts
from .store import Event, MindStore, content_text

STEWARD_SYSTEM = """You maintain the long-term memory of an ongoing conversation. You are given \
the relevant part of the CURRENT MEMORY (every entry has an id) and the NEW TURNS that are about \
to leave view. Output ONLY the changes, as one JSON object: {"ops": [...], "episode": "..."}.

The ops you can use:
- {"op":"thread","id":"n1","name":"...","kind":"topic|aside|inquiry","summary":"...","anchors":[...],"open_questions":[...]} — declare a new line of conversation (ids n1, n2, ...), or pass an existing thread id (t1, t2, ...) to update its summary.
- {"op":"add","kind":"fact","subject":"...","claim":"...","thread":"...","core":true|false,"src":<seq>}
- {"op":"add","kind":"decision","topic":"...","status":"decided|leaning|open","choice":"...","reason":"...","thread":"...","src":<seq>}
- {"op":"add","kind":"commitment","actor":"user|assistant","statement":"...","trigger":"...","due":"YYYY-MM-DDTHH:MM or null","src":<seq>}
- {"op":"update","id":"<existing r-id>", ...only the fields that change..., "src":<seq>}
- {"op":"close","id":"<existing commitment r-id>","status":"done|dropped"}
"episode" is a 2-3 sentence narrative of what happened in the new turns, keeping distinctive one-off details and asides.

Example. CURRENT MEMORY:
t1 thread (topic) garden-redesign: Priya is redesigning her back garden; raised beds are built.
r1 fact [t1] (core) user name: Priya
r4 fact [t1] pond liner size: 3m x 4m
r5 decision [t1] fence material: LEANING — cedar (reason: looks warm)
r6 commitment (assistant) remind Priya to order gravel — trigger: before the patio is laid — open
NEW TURNS:
[seq 41 user] The liner was wrong, it's actually 4m x 5m. Oh and my brother Anil is visiting in May, he's vegetarian. Cedar it is, final answer. Gravel is ordered! Funny thing, a robin is nesting in the shed door hinge, I'm leaving it be. Also — don't let me forget to book the skip before demolition day.
[seq 42 assistant] Noted all of that...
Output:
{"ops":[{"op":"update","id":"r4","claim":"4m x 5m (was 3m x 4m)","src":41},{"op":"thread","id":"n1","name":"brother-visit","kind":"aside","summary":"Priya's brother Anil visits in May.","anchors":["Anil","May","vegetarian"]},{"op":"add","kind":"fact","subject":"brother","claim":"Anil, visiting in May, vegetarian","thread":"n1","core":true,"src":41},{"op":"update","id":"r5","status":"decided","choice":"cedar","src":41},{"op":"thread","id":"n2","name":"shed-robin","kind":"aside","summary":"A robin is nesting in the shed door hinge; Priya is leaving it alone.","anchors":["robin","shed","nest"]},{"op":"add","kind":"fact","subject":"robin in the shed","claim":"nesting in the shed door hinge; Priya is leaving it be","thread":"n2","core":false,"src":41},{"op":"close","id":"r6","status":"done"},{"op":"add","kind":"commitment","actor":"assistant","statement":"remind Priya to book the skip","trigger":"before demolition day","due":null,"src":41}],"episode":"Priya corrected the pond liner size to 4m x 5m and settled on a cedar fence. She mentioned her vegetarian brother Anil visits in May, confirmed the gravel is ordered, noted a robin nesting in the shed door hinge that she is leaving alone, and asked to be reminded to book the skip before demolition day."}

Rules:
- The USER's statements are the evidence. Record what the user tells you about their world, plans and preferences. NEVER record the assistant's advice, how-to steps, checklists or explanations as facts (they belong in the episode narrative at most), and never turn an assistant's guess into a fact.
- Entries you do not mention stay exactly as they are. Never re-add something already in memory; to change it, "update" it by id. An empty ops list is fine when nothing durable was said.
- A correction is an "update" of the existing entry: put the new value in the claim and note the old one ("4m x 5m (was 3m x 4m)").
- Subjects are specific: "battery capacity", "battery location" — not just "battery". One fact per claim.
- Decisions belong to the USER. Record what the user settled or is leaning toward — NEVER the assistant's recommendations. "decided" only when the user explicitly settled it ("decided", "final", "ordered it"); a question like "should I do X or Y?" is an OPEN decision even if the assistant recommended X; if they say they are still thinking, it is "leaning" or "open". When a leaning becomes final, "update" its status to "decided". Do not record vague goals as decisions.
- Commitments are promises to act LATER or standing requests to track something ("remind me to X", "don't let me forget X", "before we leave, X"). Record them with their trigger. A request fulfilled in the same turn is NOT a commitment, and neither is the user's own to-do for today. When a commitment has been carried out or cancelled, "close" it.
- When a trigger is a time ("in two hours", "tomorrow morning"), compute the absolute datetime from the timestamp of the message that set it and put it in "due"; event triggers get due: null.
- "core": true for what the assistant must ALWAYS have at hand: the names of people, animals and named things, hard constraints, allergies and health needs, key dates and deadlines, budgets. Ordinary details are core: false.
- Record distinctive one-off details and personal asides as facts, even if they seem irrelevant. Keep numbers, names, dates and file paths exact. Never invent entries.
- Threads are lines of conversation that can go quiet and come back. Reuse an existing thread id whenever the subject fits; declare a new thread only for a genuinely new subject, and give a personal aside its own small "aside" thread. Thread names are short real kebab-case slugs like "van-electrics". Every fact's "thread" must be an existing id or one you declared."""

EPISODE_SYSTEM = (
    "Narrate the conversation turns you are given in 2-4 sentences, third person. "
    "Keep every name, number, date, decision (and whether it was final or only a "
    "leaning) and promise exact. Plain text only."
)

def _op(required: list[str], **properties: Any) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": properties,
        "required": required,
        "additionalProperties": False,
    }


_TEXT = {"type": "string"}
_SRC = {"type": "integer"}
_THREAD_KIND = {"type": "string", "enum": ["topic", "aside", "inquiry"]}
_DECISION_STATUS = {"type": "string", "enum": ["decided", "leaning", "open"]}

# A DISCRIMINATED union, not one bag of optional fields. Under the loose
# first version a constrained sampler could wander into any key: observed
# live, a whole fold of {"op":"add","topic":...,"trigger":"<the actual
# fact>"} with no kind — every op dropped, a decision lost. Each shape now
# names its required fields and forbids the others.
OP_SCHEMA = {
    "name": "memory_ops",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "ops": {
                "type": "array",
                "items": {
                    "anyOf": [
                        _op(
                            ["op", "id", "name", "kind", "summary"],
                            op={"const": "thread"}, id=_TEXT, name=_TEXT, kind=_THREAD_KIND,
                            summary=_TEXT,
                            anchors={"type": "array", "items": _TEXT},
                            open_questions={"type": "array", "items": _TEXT},
                        ),
                        _op(
                            ["op", "kind", "subject", "claim", "core"],
                            op={"const": "add"}, kind={"const": "fact"}, subject=_TEXT,
                            claim=_TEXT, thread=_TEXT, core={"type": "boolean"}, src=_SRC,
                        ),
                        _op(
                            ["op", "kind", "topic", "status", "choice"],
                            op={"const": "add"}, kind={"const": "decision"}, topic=_TEXT,
                            status=_DECISION_STATUS, choice=_TEXT, reason=_TEXT, thread=_TEXT,
                            src=_SRC,
                        ),
                        _op(
                            ["op", "kind", "actor", "statement", "trigger"],
                            op={"const": "add"}, kind={"const": "commitment"},
                            actor={"type": "string", "enum": ["user", "assistant"]},
                            statement=_TEXT, trigger=_TEXT, due={"type": ["string", "null"]},
                            src=_SRC,
                        ),
                        _op(
                            ["op", "id"],
                            op={"const": "update"}, id=_TEXT, claim=_TEXT, choice=_TEXT,
                            status=_TEXT, reason=_TEXT, statement=_TEXT, trigger=_TEXT,
                            due={"type": ["string", "null"]}, core={"type": "boolean"}, src=_SRC,
                        ),
                        _op(
                            ["op", "id", "status"],
                            op={"const": "close"}, id=_TEXT,
                            status={"type": "string", "enum": ["done", "dropped"]},
                        ),
                    ]
                },
            },
            "episode": _TEXT,
        },
        "required": ["ops", "episode"],
        "additionalProperties": False,
    },
}

# Backends that rejected response_format once are not asked again.
_SCHEMA_REFUSED: set[str] = set()

MESSAGE_TOKEN_CAP = 1200
SLICE_MAX_THREADS = 12


class StewardParseError(Exception):
    pass


@dataclass
class FoldOutcome:
    """What one run_steward call did — telemetry for the soak instruments."""

    folds: int = 0
    prose_fallbacks: int = 0
    ops_applied: int = 0
    ops_dropped: list[str] = field(default_factory=list)
    deduped: int = 0
    stale: bool = False  # a fork landed inside the span; nothing committed
    errors: list[str] = field(default_factory=list)


async def run_steward(
    config: MindConfig,
    store: MindStore,
    session_id: str,
    events: list[Event],
    provider: Any,
    model: str,
    upto_seq: int,
    now: float | None = None,
    mem: MemBackend | None = None,
    scale: float = 1.33,
    on_usage: Any = None,
) -> FoldOutcome:
    """Fold live events up to upto_seq into the fold log.

    Oversized spans (first contact with a long transcript, the re-fold after
    a deep fork) are chunked into sequential folds so no single extraction
    call can overflow the model's window. Each chunk commits on its own.
    """
    outcome = FoldOutcome()
    state = replay(store.live_folds(session_id))
    fold = [event for event in events if state.covered_upto < event.seq <= upto_seq]
    if not fold:
        return outcome
    for chunk in _chunks(fold, _transcript_cap(config, scale)):
        state = replay(store.live_folds(session_id))
        committed = await _fold_pass(
            config, store, session_id, state, chunk, provider, model, now, mem, outcome,
            scale, on_usage,
        )
        if not committed:
            outcome.stale = True
            break
    return outcome


def _output_reserve(config: MindConfig) -> int:
    """Real tokens kept free for the steward's reply."""
    return min(config.extraction_max_tokens, max(700, config.window // 5))


def _slice_cap(config: MindConfig) -> int:
    return min(config.steward_slice_tokens, max(200, int(config.window * 0.12)))


def _transcript_cap(config: MindConfig, scale: float) -> int:
    """Estimated tokens of transcript one pass may carry: whatever the window
    leaves after the system prompt, the ledger slice and the reply. At 4096
    the configured 2600 simply does not fit — the span is chunked smaller
    instead of sending a call that can only truncate."""
    usable = int((config.window - _output_reserve(config)) / scale)
    room = usable - estimate_tokens(STEWARD_SYSTEM) - _slice_cap(config) - 120
    return max(300, min(config.steward_input_tokens, room))


def _chunks(fold: list[Event], cap_tokens: int) -> list[list[Event]]:
    chunks: list[list[Event]] = [[]]
    spent = 0
    for event in fold:
        cost = estimate_tokens(_flatten(event.message, cap_tokens))
        if chunks[-1] and spent + cost > cap_tokens:
            chunks.append([])
            spent = 0
        chunks[-1].append(event)
        spent += cost
    return [chunk for chunk in chunks if chunk]


def render_record(record: Record) -> str:
    """One compact id'd line — what the steward sees (and cites) per entry."""
    data = record.data
    thread = f" [{data['thread']}]" if data.get("thread") else ""
    if record.kind == "fact":
        core = " (core)" if data.get("core") else ""
        return f"{record.id} fact{thread}{core} {data.get('subject')}: {data.get('claim')}"
    if record.kind == "decision":
        reason = f" (reason: {data['reason']})" if data.get("reason") else ""
        return (
            f"{record.id} decision{thread} {data.get('topic')}: "
            f"{str(data.get('status', 'open')).upper()} — {data.get('choice', '')}{reason}"
        )
    line = f"{record.id} commitment ({data.get('actor', 'assistant')}) {data.get('statement')}"
    if data.get("trigger"):
        line += f" — trigger: {data['trigger']}"
    if data.get("due"):
        line += f" — due: {data['due']}"
    return line + f" — {data.get('status', 'open')}"


async def build_slice(
    config: MindConfig,
    state: LedgerState,
    cue: str,
    mem: MemBackend | None,
) -> str:
    """The part of the ledger this fold may need to touch, under a budget.

    Always: threads, open commitments, undecided decisions (the entries most
    likely to change). Then everything else ranked by relevance to the span —
    a correction can only be an update if the model can see what it corrects.
    """
    lines: list[str] = []
    budget = _slice_cap(config)

    def add(line: str, floor: int = 0) -> bool:
        """Append while the slice stays above `floor` tokens of headroom."""
        nonlocal budget
        cost = estimate_tokens(line) + 1
        if budget - cost < floor:
            return False
        lines.append(line)
        budget -= cost
        return True

    # Threads and pinned entries may use 60% of the slice between them; the
    # rest is kept for whatever the span is actually about.
    floor = int(budget * 0.4)
    threads = sorted(state.threads.values(), key=lambda t: t.updated_seq, reverse=True)
    for thread in threads[:SLICE_MAX_THREADS]:
        summary = str(thread.data.get("summary", ""))[:160]
        if not add(f"{thread.id} thread ({thread.data.get('kind', 'topic')}) {thread.name}: {summary}", floor):
            break
    shown: set[str] = set()
    commitments = [r for r in state.by_kind("commitment") if r.is_open]
    commitments.sort(key=lambda r: r.updated_seq, reverse=True)
    undecided = [r for r in state.by_kind("decision") if r.data.get("status") != "decided"]
    undecided.sort(key=lambda r: r.updated_seq, reverse=True)
    for record in commitments + undecided:
        if add(render_record(record), floor):
            shown.add(record.id)

    rest = [
        record
        for record in state.records.values()
        if record.id not in shown and (record.kind != "commitment" or record.is_open)
    ]
    scored = await score_texts(cue, [record.text() for record in rest], mem, config.embed_min_sim)
    for item in sorted(scored, key=lambda s: s.score, reverse=True):
        if item.score <= 0 or not add(render_record(rest[item.index])):
            break
    return "\n".join(lines) if lines else "(memory is empty)"


async def _fold_pass(
    config: MindConfig,
    store: MindStore,
    session_id: str,
    state: LedgerState,
    fold: list[Event],
    provider: Any,
    model: str,
    now: float | None,
    mem: MemBackend | None,
    outcome: FoldOutcome,
    scale: float = 1.33,
    on_usage: Any = None,
) -> bool:
    """One chunk -> one committed fold row. False = span went stale (fork)."""
    span_from, span_to = fold[0].seq, fold[-1].seq
    transcript = "\n".join(
        f"[seq {event.seq} {event.role}{_stamp(event, now)}] "
        f"{_flatten(event.message, _transcript_cap(config, scale))}"
        for event in fold
    )
    ledger_slice = await build_slice(config, state, transcript, mem)
    clock_line = ""
    if now is not None:
        clock_line = f"Current datetime: {datetime.fromtimestamp(now).isoformat(timespec='minutes')}\n\n"
    messages = [
        {"role": "system", "content": STEWARD_SYSTEM},
        {
            "role": "user",
            "content": (
                f"{clock_line}CURRENT MEMORY (relevant part):\n{ledger_slice}\n\n"
                f"NEW TURNS:\n{transcript}\n\nChanges (JSON only):"
            ),
        },
    ]
    extraction_model = config.extraction_model or model
    ops: list[dict[str, Any]] = []
    episode = ""
    kind = "steward"
    try:
        try:
            content = await _extract(
                config, provider, extraction_model, messages, schema=True, scale=scale, on_usage=on_usage
            )
            ops, episode = _parse_proposal(content)
        except StewardParseError as error:
            # Observed live (gemma-4-26b via OpenRouter, schema-constrained):
            # ~15% of proposals derail mid-object ('"kind:"', then whitespace
            # to the token cap). An episode-only fold would lose the span's
            # facts for good, so try once more unconstrained first.
            outcome.errors.append(f"retry after {error!r}"[:200])
            content = await _extract(
                config, provider, extraction_model, messages, schema=False, scale=scale
            )
            ops, episode = _parse_proposal(content)
    except Exception as error:
        outcome.errors.append(repr(error)[:300])
        print(f"[mind] steward proposal unusable ({error!r}); episode-only fold", flush=True)
        kind = "prose"
        ops = []
        episode = await _extract(
            config,
            provider,
            extraction_model,
            [
                {"role": "system", "content": EPISODE_SYSTEM},
                {"role": "user", "content": transcript},
            ],
            schema=False,
            scale=scale,
        )
        episode = _strip_think(episode).strip()
        if not episode:
            raise StewardParseError("episode fallback returned nothing")

    if ops:
        merged = await _merge_near_duplicates(config, state, ops, mem)
        outcome.deduped += merged
    # Dry-run the ops on a scratch replay for the report; the fold row stores
    # the ops as proposed (after the semantic merge, which needs embeddings
    # and therefore must happen exactly once, here) — replay revalidates them
    # identically every time.
    report = apply_ops(state, ops, fold=len(state.episodes) + 1, span_to=span_to)
    seq = store.append_fold(session_id, span_from, span_to, ops, episode, kind=kind)
    if seq is None:
        return False
    outcome.folds += 1
    outcome.prose_fallbacks += 1 if kind == "prose" else 0
    _merge_report(outcome, report)
    return True


NEAR_DUPLICATE_SIM = 0.88
SAME_SUBJECT_SIM = 0.72  # enough when one key's words contain the other's


async def _merge_near_duplicates(
    config: MindConfig, state: LedgerState, ops: list[dict[str, Any]], mem: MemBackend | None
) -> int:
    """Turn an `add` that restates an existing entry into an `update` of it.

    Key-equality de-dup (ledger.py) misses paraphrased keys and misses KIND
    entirely — both observed live: "Teodor: mechanic who quoted 640 euros"
    then "timing chain quote: ..." as two facts; and a budget recorded as a
    DECISION (12,000) later corrected by adding a FACT (14,500), leaving two
    contradicting entries the model then chose between at random. Facts and
    decisions are compared together (the ledger aliases their value fields);
    commitments only among themselves. Needs a semantic backend; without one
    this is a no-op. Rewrites ops in place.
    """
    if mem is None:
        return 0
    merged = 0
    for op in ops:
        kind = op.get("kind")
        if op.get("op") != "add" or kind not in ("fact", "decision", "commitment"):
            continue
        family = ("commitment",) if kind == "commitment" else ("fact", "decision")
        existing = [r for r in state.records.values() if r.kind in family and r.is_open]
        if not existing:
            continue
        key = str(op.get("subject") or op.get("topic") or op.get("statement") or "")
        text = " ".join(
            str(op.get(name, ""))
            for name in ("subject", "topic", "statement", "claim", "choice", "trigger")
            if op.get(name)
        )
        sims = await mem.text_sims(text, [record.text() for record in existing])
        if not sims:
            return merged  # no semantic signal available (lexical backend, outage)
        key_words = set(normalize_key(key).split())

        def threshold(record: Record) -> float:
            other = set(record.key.split())
            related = bool(key_words and other) and (key_words <= other or other <= key_words)
            return SAME_SUBJECT_SIM if related else NEAR_DUPLICATE_SIM

        best = max(range(len(existing)), key=lambda index: sims[index] - threshold(existing[index]))
        if sims[best] >= threshold(existing[best]):
            target = existing[best]
            op["op"] = "update"
            op["id"] = target.id
            op.pop("kind", None)
            if target.kind != kind:
                # Keep the target's own key (its identity in its kind); the
                # value crosses over through the ledger's field aliases.
                for name in ("subject", "topic"):
                    op.pop(name, None)
            merged += 1
    return merged


def _merge_report(outcome: FoldOutcome, report: ApplyReport) -> None:
    outcome.ops_applied += report.applied
    outcome.ops_dropped.extend(report.dropped)
    outcome.deduped += report.deduped


async def _extract(
    config: MindConfig,
    provider: Any,
    model: str,
    messages: list[dict[str, Any]],
    schema: bool,
    scale: float = 1.33,
    on_usage: Any = None,
) -> str:
    estimated_input = sum(estimate_tokens(m["content"]) for m in messages)
    room = config.window - int(estimated_input * scale) - 64
    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": 0.1,
        "stream": False,
        "max_tokens": max(256, min(config.extraction_max_tokens, room)),
    }

    def finish(response: dict[str, Any]) -> str:
        if on_usage is not None:
            # Extraction calls are non-streaming and hit the same tokenizer as
            # the conversation: free calibration even for stream-only clients.
            on_usage(estimated_input, (response.get("usage") or {}).get("prompt_tokens"))
        return response["choices"][0]["message"].get("content") or ""

    refusal_key = f"{getattr(provider, 'name', '')}:{model}"
    if schema and config.steward_json_schema and refusal_key not in _SCHEMA_REFUSED:
        try:
            return finish(
                await provider.chat_completions(
                    {**payload, "response_format": {"type": "json_schema", "json_schema": OP_SCHEMA}}
                )
            )
        except Exception as error:
            if not _is_schema_refusal(error):
                raise
            _SCHEMA_REFUSED.add(refusal_key)
            print("[mind] backend refused json_schema; plain JSON from now on", flush=True)
    return finish(await provider.chat_completions(payload))


def _is_schema_refusal(error: Exception) -> bool:
    """A 4xx that is ABOUT response_format — not a context-length 400, which
    would otherwise switch constrained decoding off for good."""
    response = getattr(error, "response", None)
    if getattr(response, "status_code", None) not in (400, 404, 415, 422, 501):
        return False
    try:
        body = response.text.lower()
    except Exception:
        return True
    if any(word in body for word in ("context", "too long", "maximum", "exceed")):
        return False
    return not body or any(
        word in body for word in ("response_format", "json_schema", "schema", "unsupported", "not support", "grammar")
    )


def _strip_think(content: str) -> str:
    return re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL)


def _parse_proposal(content: str) -> tuple[list[dict[str, Any]], str]:
    # Reasoning extraction models wrap output in think blocks whose braces
    # would garbage the greedy JSON match; deliberation is not the proposal.
    content = _strip_think(content)
    match = re.search(r"\{.*\}", content, flags=re.DOTALL)
    data: Any = None
    if match:
        try:
            data = json.loads(match.group(0))
        except json.JSONDecodeError:
            data = None
    if data is None:
        # Truncated or locally malformed: salvage every op object that IS
        # complete rather than discard the whole proposal.
        ops, episode = _salvage(content)
        if not ops and not episode:
            raise StewardParseError(f"no usable JSON in steward output: {content[:150]!r}")
    else:
        if not isinstance(data, dict):
            raise StewardParseError("steward output is not an object")
        ops = data.get("ops", [])
        if not isinstance(ops, list):
            raise StewardParseError("steward field ops is not a list")
        episode = data.get("episode") or ""
        if isinstance(episode, dict):
            episode = episode.get("text") or ""
        episode = str(episode).strip()
    ops = [_clean_keys(op) for op in ops if isinstance(op, dict)]
    if not ops and not episode:
        # `{}` parses fine and says nothing: that is a failed extraction,
        # not an instruction (under v1 it erased the whole ledger).
        raise StewardParseError("steward proposal is empty")
    return ops, episode


def _clean_keys(op: dict[str, Any]) -> dict[str, Any]:
    """'kind:' / 'summary_' / 'summary ' -> the key the model meant."""
    return {re.sub(r"[\s:_]+$", "", str(key)).strip(): value for key, value in op.items()}


def _salvage(content: str) -> tuple[list[dict[str, Any]], str]:
    """Pull complete {...} objects out of a broken "ops" array, and the
    episode string if it survived. String-aware brace matching; anything
    that does not parse on its own is skipped."""
    ops: list[dict[str, Any]] = []
    start = content.find("[", max(content.find('"ops"'), 0))
    if start != -1:
        depth, in_string, escaped, begin = 0, False, False, -1
        for position in range(start + 1, len(content)):
            char = content[position]
            if in_string:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == '"':
                    in_string = False
                continue
            if char == '"':
                in_string = True
            elif char == "{":
                if depth == 0:
                    begin = position
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0 and begin != -1:
                    try:
                        parsed = json.loads(content[begin : position + 1])
                        if isinstance(parsed, dict):
                            ops.append(parsed)
                    except json.JSONDecodeError:
                        pass
                    begin = -1
            elif char == "]" and depth == 0:
                break
    episode = ""
    found = re.search(r'"episode"\s*:\s*"((?:[^"\\]|\\.)*)"', content, flags=re.DOTALL)
    if found:
        try:
            episode = json.loads(f'"{found.group(1)}"').strip()
        except json.JSONDecodeError:
            episode = found.group(1).strip()
    return ops, episode


def _stamp(event: Event, now: float | None) -> str:
    """Per-line timestamp so the steward can compute absolute due datetimes
    from relative triggers. Only rendered when the mind has a clock."""
    if now is None or not event.ts:
        return ""
    return " " + datetime.fromtimestamp(event.ts).isoformat(timespec="minutes")


def _flatten(message: dict[str, Any], chunk_cap_tokens: int = MESSAGE_TOKEN_CAP) -> str:
    parts = []
    content = content_text(message)
    if content:
        parts.append(content)
    for call in message.get("tool_calls") or []:
        function = call.get("function", {})
        parts.append(f"(called tool {function.get('name')} with {function.get('arguments')})")
    text = " ".join(parts)
    # One message may never exceed what a whole pass can carry (at a 4k
    # window that is far below the default per-message cap).
    cap = max(120, min(MESSAGE_TOKEN_CAP, chunk_cap_tokens - 40))
    if estimate_tokens(text) < cap:
        return text
    # Head AND tail: conclusions live at the end of long messages, and the
    # v1 head-only cut lost them before extraction ever saw them. Dense
    # scripts cost ~a token per character, so cut by the measured ratio.
    chars = int(len(text) * cap / estimate_tokens(text))
    head, tail = chars * 2 // 3, chars // 3
    return f"{text[:head]} …[{len(text) - head - tail} chars omitted]… {text[-tail:]}"
