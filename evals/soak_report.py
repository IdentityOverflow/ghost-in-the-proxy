"""Soak analysis: join an s14 run with the mind's telemetry into curves.

    PYTHONPATH=. python -m evals.soak_report evals/results/<run>/results.json \\
        [--metrics path/to/mind-metrics.jsonl]

Without --metrics (a baseline run has no mind) only the harness-side curves
print: probe accuracy by distance/position and the prompt-token trajectory.
With it, each turn is joined to the mind's request record (s14 has no client
tools, so requests and turns are 1:1 in order) and the fold records.
"""

import argparse
import json
import re
from pathlib import Path
from typing import Any

BUCKET_TURNS = 20


def _probe_meta(note: str) -> dict[str, Any] | None:
    match = re.match(r"probe class=(\S+) planted=(\d+) dist=(\d+)", note or "")
    if not match:
        return None
    return {"cls": match.group(1), "planted": int(match.group(2)), "dist": int(match.group(3))}


def _distance_bucket(dist: int) -> str:
    if dist <= 15:
        return "near (<=15)"
    if dist <= 60:
        return "mid (16-60)"
    return "far (>60)"


def _rate(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "  -  "
    passed = sum(1 for row in rows if row["passed"])
    return f"{passed}/{len(rows)}"


def _load_metrics(path: Path | None) -> tuple[list[dict], list[dict]]:
    if path is None or not path.exists():
        return [], []
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    requests = [record for record in records if record["kind"] == "request"]
    if requests:
        # The soak is one session; stray sessions (warm-up pings) are dropped.
        sessions: dict[str, int] = {}
        for record in requests:
            sessions[record["session"]] = sessions.get(record["session"], 0) + 1
        main = max(sessions, key=sessions.get)
        requests = [record for record in requests if record["session"] == main]
        folds = [r for r in records if r["kind"] == "fold" and r["session"] == main]
        return requests, folds
    return [], []


def render(results_path: Path, metrics_path: Path | None) -> str:
    data = json.loads(results_path.read_text(encoding="utf-8"))
    scenarios = data["scenarios"] if isinstance(data, dict) and "scenarios" in data else data
    soak = next(item for item in scenarios if item["scenario_id"] == "s14-soak")
    turns = soak["turns"]
    requests, folds = _load_metrics(metrics_path)

    lines = [f"# s14 soak — {soak['model']}", ""]
    if soak.get("aborted_at_turn"):
        lines.append(f"**ABORTED at turn {soak['aborted_at_turn']}**: {soak['abort_reason']}")
        lines.append("")

    probes = []
    for turn in turns:
        meta = _probe_meta(turn.get("note", ""))
        if meta and turn.get("passed") is not None:
            probes.append({**meta, "index": turn["index"], "passed": bool(turn["passed"]), "turn": turn})
    lines.append(f"Probes: {_rate(probes)} passed; {soak.get('probes_unreached', 0)} unreached.")
    lines.append("")

    lines.append("## Accuracy by distance x position")
    lines.append("")
    lines.append("| distance | turns 1-60 | turns 61-120 | turns 121-160 |")
    lines.append("|---|---|---|---|")
    for bucket in ("near (<=15)", "mid (16-60)", "far (>60)"):
        row = [p for p in probes if _distance_bucket(p["dist"]) == bucket]
        cells = [
            _rate([p for p in row if low <= p["index"] <= high])
            for low, high in ((1, 60), (61, 120), (121, 160))
        ]
        lines.append(f"| {bucket} | " + " | ".join(cells) + " |")
    lines.append("")

    lines.append("## Probes")
    lines.append("")
    lines.append("| turn | class | planted | dist | result | memory tok | texture msgs | reply (head) |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for probe in probes:
        request = requests[probe["index"] - 1] if len(requests) >= probe["index"] else {}
        failed = "; ".join(
            check["desc"] for check in probe["turn"]["checks"] if check.get("status") == "fail"
        )
        head = " ".join(probe["turn"]["reply"].split())[:90].replace("|", "/")
        lines.append(
            f"| {probe['index']} | {probe['cls']} | {probe['planted']} | {probe['dist']} | "
            f"{'pass' if probe['passed'] else 'FAIL: ' + failed} | "
            f"{request.get('memory_tokens', '-')} | {request.get('texture_messages', '-')} | {head} |"
        )
    lines.append("")

    lines.append("## Trajectory (means per 20-turn bucket)")
    lines.append("")
    lines.append("| turns | prompt tok | latency s | memory tok | system tok | texture msgs | ledger | episodes |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for low in range(1, len(turns) + 1, BUCKET_TURNS):
        chunk = [turn for turn in turns if low <= turn["index"] < low + BUCKET_TURNS]
        if not chunk:
            continue
        reqs = [requests[t["index"] - 1] for t in chunk if len(requests) >= t["index"]]

        def mean(values: list[float]) -> str:
            return f"{sum(values) / len(values):.0f}" if values else "-"

        lines.append(
            f"| {low}-{low + len(chunk) - 1} | {mean([t['prompt_tokens'] for t in chunk])} | "
            f"{mean([t['latency_s'] for t in chunk])} | "
            f"{mean([r['memory_tokens'] for r in reqs])} | {mean([r['system_tokens'] for r in reqs])} | "
            f"{mean([r['texture_messages'] for r in reqs])} | "
            f"{mean([sum(r['ledger'].values()) for r in reqs])} | {mean([r['episodes'] for r in reqs])} |"
        )
    lines.append("")

    if folds:
        failed = [fold for fold in folds if not fold["ok"]]
        lines.append("## Folds")
        lines.append("")
        summary = (
            f"{len(folds)} folds, {len(failed)} degraded "
            f"({100 * len(failed) / len(folds):.0f}%: prose fallback, error or stale), "
            f"mean {sum(f['seconds'] for f in folds) / len(folds):.1f}s"
        )
        v6 = "ops_applied" in folds[0]
        if v6:
            on_request = sum(1 for fold in folds if fold.get("path") == "request")
            summary += (
                f"; {sum(f['ops_applied'] for f in folds)} ops applied, "
                f"{sum(len(f['ops_dropped']) for f in folds)} dropped, "
                f"{sum(f['deduped'] for f in folds)} de-duplicated; "
                f"{on_request} folds ran on the request path (user waited)."
            )
        else:
            lost = sum(len(fold["lost_keys"]) for fold in folds)
            summary += f"; {lost} ledger keys lost/renamed across all folds."
        lines.append(summary)
        lines.append("")
        if v6:
            lines.append("| # | path | covered | ok | s | ledger before -> after | ops ok/dropped/dedup | note |")
            lines.append("|---|---|---|---|---|---|---|---|")
            for number, fold in enumerate(folds, start=1):
                note = (fold.get("error") or "; ".join(fold["ops_dropped"][:2]))[:80].replace("|", "/")
                lines.append(
                    f"| {number} | {fold.get('path', '')} | {fold['covered_upto']} | "
                    f"{'ok' if fold['ok'] else 'DEGRADED'} | {fold['seconds']} | "
                    f"{fold['ledger_before']} -> {fold['ledger_after']} | "
                    f"{fold['ops_applied']}/{len(fold['ops_dropped'])}/{fold['deduped']} | {note} |"
                )
        else:
            lines.append("| # | upto seq | ok | s | ledger before -> after | lost keys |")
            lines.append("|---|---|---|---|---|---|")
            for number, fold in enumerate(folds, start=1):
                lines.append(
                    f"| {number} | {fold['upto_seq']} | {'ok' if fold['ok'] else 'FALLBACK'} | "
                    f"{fold['seconds']} | {fold['ledger_before']} -> {fold['ledger_after']} | "
                    f"{', '.join(fold['lost_keys'][:6])}{' …' if len(fold['lost_keys']) > 6 else ''} |"
                )
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Render the s14 soak curves.")
    parser.add_argument("results", type=Path)
    parser.add_argument("--metrics", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=None, help="default: soak.md next to results.json")
    args = parser.parse_args()
    report = render(args.results, args.metrics)
    out = args.out or args.results.with_name("soak.md")
    out.write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
