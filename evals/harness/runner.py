"""Plays a scenario against an OpenAI-compatible endpoint and records metrics.

The harness is the client: it resends the full accumulated message history on
every turn, the way real clients do. A transcript-stuffing baseline therefore
shows prompt_tokens growing roughly linearly; a working cognitive middleware
should hold it near-flat at equal or better probe accuracy. Those two curves
are the entire point of this module.
"""

from dataclasses import dataclass, field
from typing import Any

import httpx

from .checks import CheckResult, JudgeClient, evaluate_check
from .client import ChatClient, MockClient
from .scenario import Scenario, Turn

MAX_TOOL_HOPS = 6


@dataclass
class TurnRecord:
    index: int
    user: str
    reply: str
    prompt_tokens: int
    completion_tokens: int
    usage_estimated: bool
    latency_s: float
    tool_calls: list[str] = field(default_factory=list)
    tool_fallback_used: bool = False
    checks: list[CheckResult] = field(default_factory=list)
    note: str = ""

    @property
    def evaluated_checks(self) -> list[CheckResult]:
        return [check for check in self.checks if check.status in ("pass", "fail")]

    @property
    def passed(self) -> bool | None:
        evaluated = self.evaluated_checks
        if not evaluated:
            return None
        return all(check.status == "pass" for check in evaluated)

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "note": self.note,
            "user": self.user,
            "reply": self.reply,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "usage_estimated": self.usage_estimated,
            "latency_s": round(self.latency_s, 3),
            "tool_calls": self.tool_calls,
            "tool_fallback_used": self.tool_fallback_used,
            "passed": self.passed,
            "checks": [check.to_dict() for check in self.checks],
        }


@dataclass
class ScenarioResult:
    scenario_id: str
    title: str
    model: str
    turns: list[TurnRecord]
    # Set when the backend refused to continue (e.g. context wall with a
    # stop-at-limit policy). Hitting the wall is a finding, not a crash:
    # unreached probes are reported separately and never count as passed.
    aborted_at_turn: int | None = None
    abort_reason: str = ""
    probes_unreached: int = 0

    @property
    def probe_turns(self) -> list[TurnRecord]:
        return [turn for turn in self.turns if turn.passed is not None]

    @property
    def probes_passed(self) -> int:
        return sum(1 for turn in self.probe_turns if turn.passed)

    def to_dict(self) -> dict[str, Any]:
        latencies = sorted(turn.latency_s for turn in self.turns) or [0.0]
        prompt_loads = [turn.prompt_tokens for turn in self.turns] or [0]
        return {
            "scenario_id": self.scenario_id,
            "title": self.title,
            "model": self.model,
            "probes_passed": self.probes_passed,
            "probes_total": len(self.probe_turns),
            "probes_unreached": self.probes_unreached,
            "aborted_at_turn": self.aborted_at_turn,
            "abort_reason": self.abort_reason,
            "prompt_tokens_mean": round(sum(prompt_loads) / len(prompt_loads)),
            "prompt_tokens_max": max(prompt_loads),
            "prompt_tokens_final": prompt_loads[-1],
            "usage_estimated_anywhere": any(turn.usage_estimated for turn in self.turns),
            "latency_p50_s": round(latencies[len(latencies) // 2], 3),
            "latency_max_s": round(latencies[-1], 3),
            "turns": [turn.to_dict() for turn in self.turns],
        }


async def run_scenario(
    scenario: Scenario,
    client: ChatClient | MockClient,
    judge: JudgeClient | None = None,
    log: bool = True,
    wall: int | None = None,
    wall_mode: str = "strict",
) -> ScenarioResult:
    messages: list[dict[str, Any]] = []
    if scenario.system_prompt:
        messages.append({"role": "system", "content": scenario.system_prompt})
    tools = [tool.to_openai() for tool in scenario.tools] or None

    records: list[TurnRecord] = []
    aborted_at: int | None = None
    abort_reason = ""
    previous_load = 0
    for index, turn in enumerate(scenario.turns, start=1):
        if isinstance(client, MockClient):
            client.next_reply = turn.mock_reply
        messages.append({"role": "user", "content": turn.user})
        try:
            record = await _play_turn(index, turn, scenario, client, messages, tools)
        except Exception as error:
            aborted_at = index
            abort_reason = _describe_abort(error)
            if log:
                print(f"[{scenario.id}] turn {index:>2} ABORTED: {abort_reason}", flush=True)
            break
        # A growing transcript must report a growing prompt load. A drop means
        # the backend silently truncated: the true size exceeded its window,
        # so under honest stop-at-limit semantics this turn would have failed.
        # wall_mode "exceed" disables this heuristic — REQUIRED for middleware
        # runs, where the system under test legitimately shrinks the context.
        truncation_detected = (
            wall_mode == "strict"
            and not isinstance(client, MockClient)
            and record.prompt_tokens < previous_load
        )
        if wall is not None and (record.prompt_tokens > wall or truncation_detected):
            aborted_at = index
            cause = (
                f"silent backend truncation detected (prompt_tokens {record.prompt_tokens} "
                f"< previous {previous_load})"
                if truncation_detected
                else f"prompt_tokens {record.prompt_tokens} exceeded wall {wall}"
            )
            abort_reason = f"harness context wall ({wall} tokens): {cause}"
            record.checks = []
            records.append(record)
            if log:
                print(f"[{scenario.id}] turn {index:>2} ABORTED: {abort_reason}", flush=True)
            break
        previous_load = record.prompt_tokens
        for check in turn.checks:
            record.checks.append(await evaluate_check(check, record.reply, judge))
        records.append(record)
        if log:
            _log_turn(scenario, record)
    unreached = sum(
        1 for turn in scenario.turns[(aborted_at - 1 if aborted_at else len(scenario.turns)):] if turn.checks
    )
    return ScenarioResult(
        scenario_id=scenario.id,
        title=scenario.title,
        model=client.model,
        turns=records,
        aborted_at_turn=aborted_at,
        abort_reason=abort_reason,
        probes_unreached=unreached,
    )


def _describe_abort(error: Exception) -> str:
    if isinstance(error, httpx.HTTPStatusError):
        body = error.response.text[:300]
        return f"HTTP {error.response.status_code} from backend: {body}"
    return f"{type(error).__name__}: {error}"


async def _play_turn(
    index: int,
    turn: Turn,
    scenario: Scenario,
    client: ChatClient | MockClient,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None,
) -> TurnRecord:
    tool_call_names: list[str] = []
    result = await client.complete(messages, tools=tools)

    hops = 0
    while _tool_calls(result.message) and hops < MAX_TOOL_HOPS:
        hops += 1
        messages.append(result.message)
        for call in _tool_calls(result.message):
            name = call["function"]["name"]
            arguments = call["function"].get("arguments") or "{}"
            tool_call_names.append(name)
            try:
                content = scenario.tool(name).resolve(arguments)
            except KeyError:
                content = f"error: unknown tool {name}"
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call.get("id", f"call-{hops}"),
                    "content": content,
                }
            )
        result = await client.complete(messages, tools=tools)

    fallback_used = False
    if turn.expects_tool and not tool_call_names and turn.fallback_tool:
        # The model answered from the armchair; play the user who runs the
        # command themselves so the information still enters the stream.
        fallback_used = True
        messages.append(result.message)
        payload = scenario.tool(turn.fallback_tool).resolve("{}")
        messages.append(
            {
                "role": "user",
                "content": f"I ran `{turn.fallback_tool}` myself. Output:\n{payload}",
            }
        )
        result = await client.complete(messages, tools=tools)

    messages.append(result.message)
    reply = result.message.get("content") or ""
    return TurnRecord(
        index=index,
        user=turn.user,
        reply=reply,
        prompt_tokens=result.prompt_tokens,
        completion_tokens=result.completion_tokens,
        usage_estimated=result.usage_estimated,
        latency_s=result.latency_s,
        tool_calls=tool_call_names,
        tool_fallback_used=fallback_used,
        note=turn.note,
    )


def _tool_calls(message: dict[str, Any]) -> list[dict[str, Any]]:
    calls = message.get("tool_calls") or []
    return [call for call in calls if call.get("type", "function") == "function"]


def _log_turn(scenario: Scenario, record: TurnRecord) -> None:
    status = ""
    if record.passed is True:
        status = " PROBE PASS"
    elif record.passed is False:
        failed = [check.desc for check in record.checks if check.status == "fail"]
        status = f" PROBE FAIL ({'; '.join(failed)})"
    tools_note = f" tools={record.tool_calls}" if record.tool_calls else ""
    print(
        f"[{scenario.id}] turn {record.index:>2}"
        f" prompt_tokens={record.prompt_tokens:>6}"
        f" latency={record.latency_s:5.1f}s{tools_note}{status}",
        flush=True,
    )
