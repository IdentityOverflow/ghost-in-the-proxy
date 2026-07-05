"""Judge backend that runs rubric checks through the Codex CLI.

For setups with a ChatGPT subscription but no raw API key: each judge call
is one `codex exec --ephemeral` invocation in read-only sandbox, with the
final message captured via --output-last-message. Slower per call than an
API, but rubric checks are short and few.
"""

import asyncio
import os
import tempfile
import time
from dataclasses import dataclass
from typing import Any

from .client import ChatResult, estimate_tokens


@dataclass
class CodexJudge:
    codex_model: str | None = None  # None = the subscription's default model
    timeout_s: float = 360.0

    @property
    def model(self) -> str:
        return f"codex:{self.codex_model or 'default'}"

    async def __aenter__(self) -> "CodexJudge":
        return self

    async def __aexit__(self, *exc: Any) -> None:
        return None

    async def complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> ChatResult:
        prompt = "\n\n".join(str(message.get("content", "")) for message in messages)
        fd, out_path = tempfile.mkstemp(prefix="codex-judge-", suffix=".txt")
        os.close(fd)
        command = [
            "codex",
            "exec",
            "--skip-git-repo-check",
            "--ephemeral",
            "-s",
            "read-only",
            "-o",
            out_path,
        ]
        if self.codex_model:
            command += ["-m", self.codex_model]
        command.append(prompt)
        started = time.monotonic()
        try:
            process = await asyncio.create_subprocess_exec(
                *command,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.PIPE,
                cwd=tempfile.gettempdir(),
            )
            try:
                _, stderr = await asyncio.wait_for(process.communicate(), timeout=self.timeout_s)
            except asyncio.TimeoutError:
                process.kill()
                raise RuntimeError(f"codex exec timed out after {self.timeout_s}s")
            if process.returncode != 0:
                raise RuntimeError(
                    f"codex exec failed ({process.returncode}): {stderr.decode()[:300]}"
                )
            content = open(out_path, encoding="utf-8").read().strip()
        finally:
            try:
                os.unlink(out_path)
            except OSError:
                pass
        return ChatResult(
            message={"role": "assistant", "content": content},
            prompt_tokens=estimate_tokens(prompt),
            completion_tokens=estimate_tokens(content),
            usage_estimated=True,
            latency_s=time.monotonic() - started,
        )
