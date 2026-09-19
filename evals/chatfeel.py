"""Does it feel like talking to a person? — conversational-quality instruments.

The goal is not a smarter small model but a better conversationalist: less
repetitive, less predictable, less synthetic. That is hard to quantify, so
this tool gives three imperfect views that are useful together:

  replay   play a fixed USER script (the user turns of a recorded livechat
           session) against the proxy, so different designs answer the same
           person. Caveat: later user turns were written in reaction to a
           different assistant, so this measures style, not dialogue flow.
  tells    a dashboard of synthetic tells per session: replies ending on a
           question, validation openers, chat markdown, quoting the user back,
           reply length and how much it varies, whether length tracks the
           user's, phrase recycling.
  judge    blind pairwise preference by a strong model (gpt-6-astra via the PI
           CLI): same user turn, two designs' replies, A/B order randomized,
           ties allowed, told not to reward length or structure.

    python -m evals.chatfeel replay --script live3 --session d-observe --turns 30
    python -m evals.chatfeel tells d-rhythm d-observe d-typed
    python -m evals.chatfeel judge d-rhythm d-observe --pairs 16
"""

import argparse
import json
import random
import re
import statistics
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent / "results" / "livechat"

VALIDATION_OPENERS = re.compile(
    r"^\W*(that sounds|that's (a|an|so|such|really|great|awesome|wonderful|fantastic|amazing|totally|completely)|"
    r"that is (a|an|so|such|really|great)|it sounds like|sounds like|oh,? that|i love (that|how)|what a|"
    r"i('m| am) so (glad|sorry|happy)|i hear you|i (totally|completely) (get|understand)|"
    r"congratulations|congrats|ah,? the)\b",
    flags=re.IGNORECASE,
)
HYPE = re.compile(
    r"\b(huge|massive|amazing|incredible|fantastic|wonderful|awesome|absolutely|victory|milestone|"
    r"so proud|you've earned|you deserve|well[- ]deserved|a (big|major|real) win)\b",
    flags=re.IGNORECASE,
)
MARKDOWN = re.compile(r"\*\*[^*]+\*\*|^\s*([-*•]|\d+\.)\s+\S|^#{1,4}\s", flags=re.MULTILINE)
QUOTED = re.compile(r"[\"“']([^\"”']{6,60})[\"”']")
SCARE_QUOTES = re.compile(r"(?<![\w])[\"“]([^\"”\n]{2,40})[\"”](?![\w])")
SIGN_OFFS = re.compile(
    r"(you('ve| have) got this|go (crush|get) (it|'em|some (sleep|rest))|you('ve| have) earned (it|this)|"
    r"you should be proud|i('m| am) (here|around) (whenever|if)|i'll be here|good ?night,? \w+|"
    r"you deserve (it|this))[.!]*\s*$",
    flags=re.IGNORECASE,
)


def load(session: str) -> dict:
    return json.loads((ROOT / f"{session}.json").read_text())


def turns_of(session: str) -> list[tuple[str, str]]:
    messages = [m for m in load(session)["messages"] if m["role"] in ("user", "assistant")]
    return [
        (messages[i]["content"], messages[i + 1]["content"])
        for i in range(0, len(messages) - 1, 2)
        if messages[i]["role"] == "user" and messages[i + 1]["role"] == "assistant"
    ]


def trigrams(text: str) -> set[str]:
    words = re.findall(r"[a-z']+", text.lower())
    return {" ".join(words[i : i + 3]) for i in range(len(words) - 2)}


def tells(session: str, limit: int | None = None) -> dict:
    pairs = turns_of(session)[:limit]
    users = [user for user, _ in pairs]
    replies = [reply for _, reply in pairs]
    lengths = [len(reply.split()) for reply in replies]
    user_lengths = [len(user.split()) for user in users]
    ends_q = [reply.rstrip().rstrip("*_\"')”’ ").endswith("?") for reply in replies]
    echo = 0
    for user, reply in pairs:
        lowered = user.lower()
        echo += any(match.group(1).lower() in lowered for match in QUOTED.finditer(reply))
    reuse = []
    for i in range(1, len(replies)):
        previous = set().union(*[trigrams(r) for r in replies[max(0, i - 8) : i]])
        mine = trigrams(replies[i])
        reuse.append(len(mine & previous) / max(1, len(mine)))
    short_users = [i for i, n in enumerate(user_lengths) if n <= 12]
    correlation = (
        statistics.correlation(user_lengths, lengths) if len(set(user_lengths)) > 1 and len(set(lengths)) > 1 else 0.0
    )
    timing = load(session).get("turns", [])[: len(pairs)]
    ttft = sorted(t["ttft_s"] for t in timing) or [0.0]
    return {
        "session": session,
        "turns": len(pairs),
        "ends_on_question": sum(ends_q) / len(pairs),
        "validation_opener": sum(bool(VALIDATION_OPENERS.match(r.strip())) for r in replies) / len(pairs),
        "markdown": sum(bool(MARKDOWN.search(r)) for r in replies) / len(pairs),
        "cheerleading": sum(len(HYPE.findall(r)) for r in replies) / len(pairs),
        "quotes_user_back": echo / len(pairs),
        "scare_quotes": sum(bool(SCARE_QUOTES.search(r)) for r in replies) / len(pairs),
        "pep_sign_off": sum(bool(SIGN_OFFS.search(r.strip())) for r in replies) / len(pairs),
        "mean_words": statistics.mean(lengths),
        "length_variation": statistics.pstdev(lengths) / max(1.0, statistics.mean(lengths)),
        "words_after_short_user_turn": (
            statistics.mean(lengths[i] for i in short_users) if short_users else float("nan")
        ),
        "length_tracks_user": correlation,
        "trigram_reuse": statistics.mean(reuse) if reuse else 0.0,
        "ttft_median_s": ttft[len(ttft) // 2],
        "ttft_p90_s": ttft[max(0, int(len(ttft) * 0.9) - 1)],
    }


def print_tells(sessions: list[str], limit: int | None) -> None:
    rows = [tells(session, limit) for session in sessions]
    labels = [
        ("ends_on_question", "replies ending on a question", "{:.0%}"),
        ("validation_opener", "validation / summary openers", "{:.0%}"),
        ("markdown", "bold / lists in chat", "{:.0%}"),
        ("cheerleading", "hype words per reply", "{:.1f}"),
        ("quotes_user_back", "quotes the user back", "{:.0%}"),
        ("scare_quotes", "phrases in scare quotes", "{:.0%}"),
        ("pep_sign_off", "pep-talk sign-offs", "{:.0%}"),
        ("mean_words", "mean reply length (words)", "{:.0f}"),
        ("words_after_short_user_turn", "  …after a <=12-word user turn", "{:.0f}"),
        ("length_variation", "length variation (sd/mean)", "{:.2f}"),
        ("length_tracks_user", "length tracks the user's (r)", "{:+.2f}"),
        ("trigram_reuse", "phrase recycling (trigram reuse)", "{:.1%}"),
        ("ttft_median_s", "time to first token, median (s)", "{:.1f}"),
        ("ttft_p90_s", "time to first token, p90 (s)", "{:.1f}"),
    ]
    width = max(len(label) for _, label, _ in labels)
    print(" " * width, *[f"{row['session']:>14}" for row in rows])
    print(" " * width, *[f"{str(row['turns']) + ' turns':>14}" for row in rows])
    for key, label, fmt in labels:
        print(f"{label:<{width}}", *[f"{fmt.format(row[key]):>14}" for row in rows])


def replay(script: str, session: str, turns: int, base_url: str, model: str, pause: float) -> None:
    source = load(script)
    system = next((m["content"] for m in source["messages"] if m["role"] == "system"), None)
    users = [m["content"] for m in source["messages"] if m["role"] == "user"][:turns]
    for index, text in enumerate(users):
        command = [
            sys.executable, "-m", "evals.livechat", "--session", session,
            "--base-url", base_url, "--model", model,
        ]
        if index == 0 and system:
            command += ["--system", system]
        subprocess.run(command + [text], check=False, stdout=subprocess.DEVNULL)
        print(f"[{session}] turn {index + 1}/{len(users)}", flush=True)
        time.sleep(pause)


JUDGE_PROMPT = """You are judging how HUMAN two chat companions sound. Below are {n} moments from one casual \
text conversation between a person and a companion they talk to like a friend. For each moment you see the \
person's message and two candidate replies (A and B) from different systems. Earlier conversation differs \
slightly between the systems; judge each reply on its own as a response to that message.

Pick the reply that sounds more like a real person texting a friend: natural, specific, unforced — not \
formulaic, not customer-service, not therapist-speak, not engagement bait (tacked-on questions), not padded.
Rules: do NOT reward length, thoroughness, formatting, politeness or helpfulness as such. A short reply that \
lands beats a long one that performs. If a reply ignores or mishandles what the person actually said, it \
loses. Use "tie" when neither is clearly more human.

Answer with ONLY a JSON list, one object per moment, in order:
[{{"moment": 1, "winner": "A" | "B" | "tie", "why": "<max 12 words>"}}, ...]

{moments}"""


def judge(first: str, second: str, pairs: int, seed: int, skip: int) -> None:
    a_turns, b_turns = turns_of(first), turns_of(second)
    shared = [
        i for i in range(skip, min(len(a_turns), len(b_turns))) if a_turns[i][0] == b_turns[i][0]
    ]
    rng = random.Random(seed)
    chosen = sorted(rng.sample(shared, min(pairs, len(shared))))
    flips = [rng.random() < 0.5 for _ in chosen]
    moments = []
    for number, (index, flip) in enumerate(zip(chosen, flips), start=1):
        reply_a, reply_b = a_turns[index][1], b_turns[index][1]
        if flip:
            reply_a, reply_b = reply_b, reply_a
        moments.append(
            f"--- Moment {number}\nPERSON: {a_turns[index][0]}\n\nREPLY A: {reply_a}\n\nREPLY B: {reply_b}\n"
        )
    prompt = JUDGE_PROMPT.format(n=len(moments), moments="\n".join(moments))
    match = None
    for _attempt in range(3):  # the CLI occasionally returns nothing; ask again
        result = subprocess.run(
            ["pi", "-p", "--no-session", "--model", "openai-codex/gpt-6-astra", "--thinking", "low", "-nt", prompt],
            stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=900,
        )
        match = re.search(r"\[.*\]", result.stdout, flags=re.DOTALL)
        try:
            if match and len(json.loads(match.group(0))) == len(chosen):
                break
        except ValueError:
            pass
        match = None
    if not match:
        print("judge returned no JSON:", result.stdout[:400], result.stderr[:400])
        return
    verdicts = json.loads(match.group(0))
    score = {first: 0, second: 0, "tie": 0}
    for verdict, index, flip in zip(verdicts, chosen, flips):
        winner = verdict.get("winner", "tie")
        if winner == "tie":
            name = "tie"
        else:
            name = (second if winner == "A" else first) if flip else (first if winner == "A" else second)
        score[name] += 1
        print(f"turn {index + 1:>2}: {name:<14} {verdict.get('why', '')}")
    print(f"\n{first}: {score[first]}   {second}: {score[second]}   ties: {score['tie']}   (n={len(chosen)})")
    out = ROOT / f"judge-{first}-vs-{second}.json"
    out.write_text(json.dumps({"score": score, "turns": chosen, "verdicts": verdicts}, indent=1))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("replay")
    p.add_argument("--script", required=True)
    p.add_argument("--session", required=True)
    p.add_argument("--turns", type=int, default=30)
    p.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    p.add_argument("--model", default="google/gemma-4-12b")
    p.add_argument("--pause", type=float, default=1.0)
    p = sub.add_parser("tells")
    p.add_argument("sessions", nargs="+")
    p.add_argument("--turns", type=int, default=None)
    p = sub.add_parser("judge")
    p.add_argument("first")
    p.add_argument("second")
    p.add_argument("--pairs", type=int, default=16)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--skip", type=int, default=3, help="ignore the first N turns (no habits yet)")
    args = parser.parse_args()
    if args.command == "replay":
        replay(args.script, args.session, args.turns, args.base_url, args.model, args.pause)
    elif args.command == "tells":
        print_tells(args.sessions, args.turns)
    else:
        judge(args.first, args.second, args.pairs, args.seed, args.skip)


if __name__ == "__main__":
    main()
