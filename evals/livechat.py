"""Live chat through the proxy, one turn per invocation — a real client in a file.

    python -m evals.livechat --session run1 "Hi, I'm planning a garden."
    python -m evals.livechat --session run1 --stats

Keeps the full transcript in evals/results/livechat/<session>.json and resends
it every turn (what real chat clients do), streams the reply, and records what
a USER feels: time to first token and total time. Built so a person — or
another model playing the user — can hold a long, unscripted conversation with
the mind and we can read the latency curve afterwards.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).parent / "results" / "livechat"


def main() -> None:
    parser = argparse.ArgumentParser(description="One live chat turn through the proxy.")
    parser.add_argument("message", nargs="?", help="the user's message")
    parser.add_argument("--session", required=True, help="transcript name")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--model", default="google/gemma-4-12b")
    parser.add_argument("--system", default=None, help="system prompt (first turn only)")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--stats", action="store_true", help="print the latency table and exit")
    args = parser.parse_args()

    ROOT.mkdir(parents=True, exist_ok=True)
    path = ROOT / f"{args.session}.json"
    state = json.loads(path.read_text()) if path.exists() else {"messages": [], "turns": []}

    if args.stats:
        print_stats(state)
        return
    if not args.message:
        parser.error("a message is required unless --stats")
    if not state["messages"] and args.system:
        state["messages"].append({"role": "system", "content": args.system})
    state["messages"].append({"role": "user", "content": args.message})

    started = time.monotonic()
    first_token: float | None = None
    pieces: list[str] = []
    usage: dict = {}
    with httpx.Client(timeout=600) as client:
        with client.stream(
            "POST",
            f"{args.base_url.rstrip('/')}/chat/completions",
            json={
                "model": args.model,
                "messages": state["messages"],
                "temperature": args.temperature,
                "stream": True,
                "stream_options": {"include_usage": True},
            },
        ) as response:
            if response.status_code != 200:
                body = response.read().decode("utf-8", errors="replace")
                print(f"HTTP {response.status_code}: {body[:500]}", file=sys.stderr)
                sys.exit(1)
            for line in response.iter_lines():
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    chunk = json.loads(data)
                except ValueError:
                    continue
                if chunk.get("usage"):
                    usage = chunk["usage"]
                for choice in chunk.get("choices") or []:
                    piece = (choice.get("delta") or {}).get("content")
                    if piece:
                        if first_token is None:
                            first_token = time.monotonic() - started
                        pieces.append(piece)
    total = time.monotonic() - started
    reply = "".join(pieces)
    state["messages"].append({"role": "assistant", "content": reply})
    state["turns"].append(
        {
            "turn": len(state["turns"]) + 1,
            "ttft_s": round(first_token or total, 1),
            "total_s": round(total, 1),
            "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": usage.get("completion_tokens"),
        }
    )
    path.write_text(json.dumps(state, indent=1, ensure_ascii=False))
    turn = state["turns"][-1]
    print(reply)
    print(
        f"\n[turn {turn['turn']}: first token {turn['ttft_s']}s, total {turn['total_s']}s, "
        f"prompt {turn['prompt_tokens']} tok, reply {turn['completion_tokens']} tok]"
    )


def print_stats(state: dict) -> None:
    turns = state["turns"]
    if not turns:
        print("no turns yet")
        return
    print("turn  ttft_s  total_s  prompt_tok  reply_tok")
    for turn in turns:
        print(
            f"{turn['turn']:>4}  {turn['ttft_s']:>6}  {turn['total_s']:>7}  "
            f"{str(turn['prompt_tokens']):>10}  {str(turn['completion_tokens']):>9}"
        )
    ttfts = sorted(turn["ttft_s"] for turn in turns)
    print(
        f"\n{len(turns)} turns — time to first token: median {ttfts[len(ttfts) // 2]}s, "
        f"p90 {ttfts[int(len(ttfts) * 0.9) - 1 if len(ttfts) > 1 else 0]}s, max {ttfts[-1]}s"
    )


if __name__ == "__main__":
    main()
