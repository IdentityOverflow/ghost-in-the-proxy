"""Result persistence and the human-readable run report."""

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from .runner import ScenarioResult


def write_results(
    results: list[ScenarioResult],
    label: str,
    out_root: Path,
    run_meta: dict[str, Any],
) -> Path:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = out_root / f"{stamp}-{label}"
    run_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "run": run_meta,
        "scenarios": [result.to_dict() for result in results],
    }
    (run_dir / "results.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (run_dir / "report.md").write_text(render_report(results, run_meta), encoding="utf-8")
    return run_dir


def render_report(results: list[ScenarioResult], run_meta: dict[str, Any]) -> str:
    lines = [
        "# Cognitive-architecture eval report",
        "",
        f"- endpoint: `{run_meta.get('base_url', 'mock')}`",
        f"- model: `{run_meta.get('model')}`",
        f"- judge: `{run_meta.get('judge_model') or 'none (judge checks skipped)'}`",
        f"- condition: {run_meta.get('note') or 'unrecorded — pin and note the backend context config!'}",
        f"- date: {run_meta.get('date')}",
        "",
        "| scenario | probes | prompt tokens (mean / max / final) | latency p50 | tokens estimated? |",
        "|---|---|---|---|---|",
    ]
    for result in results:
        summary = result.to_dict()
        probes = f"{summary['probes_passed']}/{summary['probes_total']}"
        if summary["probes_unreached"]:
            probes += f" (+{summary['probes_unreached']} unreached)"
        lines.append(
            f"| {summary['scenario_id']} — {summary['title']}"
            f" | {probes}"
            f" | {summary['prompt_tokens_mean']} / {summary['prompt_tokens_max']}"
            f" / {summary['prompt_tokens_final']}"
            f" | {summary['latency_p50_s']}s"
            f" | {'yes' if summary['usage_estimated_anywhere'] else 'no'} |"
        )
    lines.append("")

    for result in results:
        lines.append(f"## {result.scenario_id} — {result.title}")
        lines.append("")
        if result.aborted_at_turn is not None:
            lines.append(
                f"**ABORTED at turn {result.aborted_at_turn}** — {result.abort_reason}. "
                f"{result.probes_unreached} probe(s) unreached; on a hard context wall "
                "this is the scenario's finding, not an infrastructure error."
            )
            lines.append("")
        lines.append("Token load per turn (prompt_tokens):")
        lines.append("")
        lines.append("```")
        lines.append(_token_sparkline(result))
        lines.append("```")
        lines.append("")
        for turn in result.probe_turns:
            verdict = "PASS" if turn.passed else "FAIL"
            lines.append(f"### turn {turn.index} [{verdict}] {turn.note}".rstrip())
            lines.append("")
            lines.append(f"> user: {turn.user[:200]}")
            lines.append("")
            for check in turn.checks:
                lines.append(f"- [{check.status}] {check.desc}" + (f" — {check.detail}" if check.detail else ""))
            lines.append("")
            lines.append(f"reply excerpt: {turn.reply[:400].strip()}")
            lines.append("")
    return "\n".join(lines)


def _token_sparkline(result: ScenarioResult) -> str:
    loads = [turn.prompt_tokens for turn in result.turns]
    if not loads:
        return "(no completed turns)"
    peak = max(loads)
    width = 46
    rows = []
    for index, load in enumerate(loads, start=1):
        bar = "#" * max(1, round(load / peak * width))
        rows.append(f"t{index:>2} {load:>7} {bar}")
    return "\n".join(rows)
