"""Re-grade a stored run against current scenario checks, optionally with a judge.

Checks grade stored replies, so a run can be re-scored without re-sampling
the model: this is how judge rubrics get applied after the fact, and how
probe-pattern fixes propagate to old runs honestly. Model outputs are never
altered — only their grading.

    PYTHONPATH=. python -m evals.regrade evals/results/<run-dir> \\
        --judge-model gemma4:26b --judge-base-url http://localhost:11434/v1

Writes regrade-results.json and regrade-report.md next to the originals.
"""

import argparse
import asyncio
import json
from datetime import datetime
from pathlib import Path

from .harness import ChatClient, CodexJudge, render_report
from .harness.checks import evaluate_check
from .harness.runner import ScenarioResult, TurnRecord
from .scenarios import ALL_SCENARIOS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Re-grade a stored eval run.")
    parser.add_argument("results_dir", help="run directory containing results.json")
    parser.add_argument("--judge-model", default=None)
    parser.add_argument("--judge-base-url", default="http://localhost:11434/v1")
    parser.add_argument(
        "--judge-codex",
        action="store_true",
        help="judge via the Codex CLI (ChatGPT subscription); --judge-model then picks the codex model",
    )
    return parser.parse_args()


def rebuild_turn(stored: dict) -> TurnRecord:
    return TurnRecord(
        index=stored["index"],
        user=stored["user"],
        reply=stored["reply"],
        prompt_tokens=stored["prompt_tokens"],
        completion_tokens=stored["completion_tokens"],
        usage_estimated=stored["usage_estimated"],
        latency_s=stored["latency_s"],
        tool_calls=stored.get("tool_calls", []),
        tool_fallback_used=stored.get("tool_fallback_used", False),
        note=stored.get("note", ""),
    )


async def main() -> None:
    args = parse_args()
    run_dir = Path(args.results_dir)
    stored = json.loads((run_dir / "results.json").read_text(encoding="utf-8"))
    scenarios_by_id = {scenario.id: scenario for scenario in ALL_SCENARIOS}

    judge = None
    if args.judge_codex:
        judge = CodexJudge(codex_model=args.judge_model)
    elif args.judge_model:
        judge = ChatClient(base_url=args.judge_base_url, model=args.judge_model, temperature=0.0)
        await judge.__aenter__()

    results: list[ScenarioResult] = []
    try:
        for stored_scenario in stored["scenarios"]:
            scenario = scenarios_by_id.get(stored_scenario["scenario_id"])
            if scenario is None:
                print(f"skipping unknown scenario {stored_scenario['scenario_id']}")
                continue
            aborted_at = stored_scenario.get("aborted_at_turn")
            records: list[TurnRecord] = []
            for stored_turn in stored_scenario["turns"]:
                record = rebuild_turn(stored_turn)
                # Turn indices are stable; grade the stored reply against the
                # scenario's *current* checks. Turns at/after an abort stay
                # ungraded — in stop-at-limit semantics they never completed.
                if aborted_at is None or record.index < aborted_at:
                    turn_def = scenario.turns[record.index - 1]
                    for check in turn_def.checks:
                        record.checks.append(await evaluate_check(check, record.reply, judge))
                records.append(record)
            result = ScenarioResult(
                scenario_id=stored_scenario["scenario_id"],
                title=stored_scenario["title"],
                model=stored_scenario["model"],
                turns=records,
                aborted_at_turn=stored_scenario.get("aborted_at_turn"),
                abort_reason=stored_scenario.get("abort_reason", ""),
                probes_unreached=stored_scenario.get("probes_unreached", 0),
            )
            results.append(result)
            summary = result.to_dict()
            print(
                f"{summary['scenario_id']:<18} {summary['probes_passed']}/{summary['probes_total']} probes",
                flush=True,
            )
    finally:
        if judge is not None:
            await judge.__aexit__(None, None, None)

    run_meta = dict(stored.get("run", {}))
    run_meta["regraded"] = datetime.now().isoformat(timespec="seconds")
    run_meta["judge_model"] = judge.model if judge else None
    payload = {"run": run_meta, "scenarios": [result.to_dict() for result in results]}
    (run_dir / "regrade-results.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (run_dir / "regrade-report.md").write_text(render_report(results, run_meta), encoding="utf-8")
    print(f"\nregraded results written to {run_dir}/regrade-*.md/json")


if __name__ == "__main__":
    asyncio.run(main())
