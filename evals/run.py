"""CLI for the cognitive-architecture eval suite.

Baseline against the bare proxy (or any OpenAI-compatible endpoint):

    PYTHONPATH=. python -m evals.run --base-url http://localhost:8000/v1 \\
        --model gemma4:latest --label baseline

Later, point the same command at the cognitive middleware and diff the two
report.md files. `--mock` runs the whole suite against canned replies to
verify harness plumbing without a model. `--judge-model` (plus optional
`--judge-base-url`) enables the rubric checks.
"""

import argparse
import asyncio
from datetime import datetime
from pathlib import Path

from .harness import ChatClient, CodexJudge, MockClient, run_scenario, write_results
from .scenarios import ALL_SCENARIOS, get_scenarios

RESULTS_ROOT = Path(__file__).parent / "results"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run cognitive-architecture eval scenarios.")
    parser.add_argument("--base-url", default="http://localhost:8000/v1")
    parser.add_argument("--model", default="gemma4:latest")
    parser.add_argument("--label", default="run", help="short tag for the results directory")
    parser.add_argument(
        "--scenario",
        action="append",
        dest="scenarios",
        help="scenario id to run (repeatable); default: all",
    )
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument(
        "--wall",
        type=int,
        default=None,
        help="harness-enforced context wall in tokens: abort a scenario when prompt_tokens "
        "exceeds this or the backend silently truncates (reported load drops)",
    )
    parser.add_argument(
        "--wall-mode",
        choices=["strict", "exceed"],
        default="strict",
        help="strict: also abort when reported load drops (baseline truncation signature); "
        "exceed: only abort past the wall — required for middleware runs, which shrink context",
    )
    parser.add_argument("--judge-model", default=None)
    parser.add_argument(
        "--judge-votes",
        type=int,
        default=1,
        help="majority-of-N judging for judge checks (odd N; stabilizes borderline rubrics)",
    )
    parser.add_argument("--judge-base-url", default=None, help="defaults to --base-url")
    parser.add_argument(
        "--judge-codex",
        action="store_true",
        help="judge via the Codex CLI (ChatGPT subscription); --judge-model then picks the codex model",
    )
    parser.add_argument(
        "--note",
        default=None,
        help="free-text run condition recorded in results (e.g. 'LM Studio, ctx 32k pinned, stop-at-limit')",
    )
    parser.add_argument("--mock", action="store_true", help="run against canned replies (plumbing test)")
    parser.add_argument("--list", action="store_true", help="list scenarios and exit")
    return parser.parse_args()


async def main() -> None:
    args = parse_args()
    if args.list:
        for scenario in ALL_SCENARIOS:
            probes = sum(1 for turn in scenario.turns if turn.checks)
            print(f"{scenario.id:>16}  {len(scenario.turns):>2} turns, {probes} probes — {scenario.description}")
        return

    scenarios = get_scenarios(args.scenarios)
    judge = None
    if args.judge_codex:
        judge = CodexJudge(codex_model=args.judge_model)
    elif args.judge_model:
        judge = ChatClient(
            base_url=args.judge_base_url or args.base_url,
            model=args.judge_model,
            temperature=0.0,
        )

    client = (
        MockClient()
        if args.mock
        else ChatClient(base_url=args.base_url, model=args.model, temperature=args.temperature)
    )

    results = []
    async with client:
        if judge is not None:
            await judge.__aenter__()
        try:
            for scenario in scenarios:
                print(f"\n=== {scenario.id}: {scenario.title} ({len(scenario.turns)} turns) ===", flush=True)
                results.append(
                    await run_scenario(
                        scenario,
                        client,
                        judge=judge,
                        wall=args.wall,
                        wall_mode=args.wall_mode,
                        judge_votes=args.judge_votes,
                    )
                )
        finally:
            if judge is not None:
                await judge.__aexit__(None, None, None)

    run_meta = {
        "base_url": None if args.mock else args.base_url,
        "model": client.model,
        "judge_model": judge.model if judge else None,
        "temperature": args.temperature,
        "wall": args.wall,
        "wall_mode": args.wall_mode,
        "judge_votes": args.judge_votes,
        "note": args.note,
        "date": datetime.now().isoformat(timespec="seconds"),
    }
    run_dir = write_results(results, args.label, RESULTS_ROOT, run_meta)

    print(f"\n{'scenario':<18} {'probes':<9} {'mean ptok':<10} {'final ptok':<10}")
    for result in results:
        summary = result.to_dict()
        probes = f"{summary['probes_passed']}/{summary['probes_total']}"
        if summary["probes_unreached"]:
            probes += f"+{summary['probes_unreached']}?"
        aborted = f"  ABORTED@t{summary['aborted_at_turn']}" if summary["aborted_at_turn"] else ""
        print(
            f"{summary['scenario_id']:<18}"
            f" {probes:<9}"
            f" {summary['prompt_tokens_mean']:<10}"
            f" {summary['prompt_tokens_final']:<10}{aborted}"
        )
    print(f"\nresults written to {run_dir}")


if __name__ == "__main__":
    asyncio.run(main())
