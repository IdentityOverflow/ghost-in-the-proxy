"""Check evaluation: deterministic regex probes plus the optional LLM judge.

Deterministic checks are preferred everywhere they can express the
expectation; the judge exists for genuinely open-ended rubrics (e.g. "is the
list of open items complete?") and is always optional. A judge check without
a configured judge is reported as skipped, never silently passed.
"""

import json
import re
from dataclasses import dataclass
from typing import Any, Protocol

from .client import ChatResult
from .scenario import Check


class JudgeClient(Protocol):
    """Anything with a chat-complete surface can judge: API client or CLI wrapper."""

    model: str

    async def complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> ChatResult: ...

JUDGE_SYSTEM_PROMPT = (
    "You are a strict evaluator. You receive a rubric question and an "
    "assistant reply. Answer with a JSON object only: "
    '{"pass": true|false, "reason": "<one sentence>"}. '
    "Judge only what the rubric asks; do not reward verbosity."
)


@dataclass
class CheckResult:
    kind: str
    desc: str
    status: str  # "pass" | "fail" | "skipped" | "error"
    detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "desc": self.desc,
            "status": self.status,
            "detail": self.detail,
        }


def _match_any(patterns: list[str], reply: str) -> str | None:
    for pattern in patterns:
        if re.search(pattern, reply, flags=re.IGNORECASE | re.DOTALL):
            return pattern
    return None


async def evaluate_check(
    check: Check,
    reply: str,
    judge: JudgeClient | None,
) -> CheckResult:
    if check.kind == "must_mention":
        hit = _match_any(check.patterns, reply)
        if hit is not None:
            return CheckResult(check.kind, check.desc, "pass", f"matched {hit!r}")
        return CheckResult(
            check.kind, check.desc, "fail", f"no pattern of {check.patterns} matched"
        )

    if check.kind == "must_not_mention":
        hit = _match_any(check.patterns, reply)
        if hit is None:
            return CheckResult(check.kind, check.desc, "pass")
        return CheckResult(check.kind, check.desc, "fail", f"forbidden match {hit!r}")

    if check.kind == "judge":
        if judge is None:
            return CheckResult(check.kind, check.desc, "skipped", "no judge configured")
        return await _run_judge(check, reply, judge)

    return CheckResult(check.kind, check.desc, "error", f"unknown check kind {check.kind}")


async def _run_judge(check: Check, reply: str, judge: JudgeClient) -> CheckResult:
    user_prompt = (
        f"Rubric question:\n{check.rubric}\n\n"
        f"Assistant reply to evaluate:\n---\n{reply}\n---"
    )
    try:
        result = await judge.complete(
            [
                {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ]
        )
        content = result.message.get("content") or ""
        verdict = _parse_judge_verdict(content)
        if verdict is None:
            return CheckResult(check.kind, check.desc, "error", f"unparseable verdict: {content[:200]}")
        passed, reason = verdict
        return CheckResult(check.kind, check.desc, "pass" if passed else "fail", reason)
    except Exception as error:  # judge outage should not abort the scenario run
        return CheckResult(check.kind, check.desc, "error", f"judge call failed: {error}")


def _parse_judge_verdict(content: str) -> tuple[bool, str] | None:
    match = re.search(r"\{.*\}", content, flags=re.DOTALL)
    if not match:
        return None
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict) or "pass" not in data:
        return None
    return bool(data["pass"]), str(data.get("reason", ""))
