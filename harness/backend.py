"""Model backend abstraction.

The iteration loop talks to a `Backend` whose only contract is
`generate(prompt) -> GenerateResult`. Model-specific quirks (Qwen's
`/no_think` directive, future HuggingFace base-weight framing, Google AI
Studio's API shape) live here at the adapter layer and never leak upward.

This is load-bearing per the Apr 13 decision: Phase 1.5 (Google AI Studio
31B cloud) and Phase 2 (HuggingFace base weights via llama.cpp/MLX) both
require swapping backends without touching loop.py. The interface is fixed
now so those swaps are trivial later.
"""

from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable
from urllib.request import Request, urlopen


OLLAMA_ENDPOINT = "http://localhost:11434/api/generate"


@dataclass
class GenerateResult:
    """One generation result from a backend."""

    text: str
    latency_s: float
    eval_tokens: int
    eval_rate_tok_s: float
    raw: dict[str, Any]


class Backend:
    """Abstract backend. Subclasses implement `generate`."""

    name: str  # Human-readable model identifier used in results/episodes

    def generate(self, prompt: str) -> GenerateResult:
        raise NotImplementedError


# -- Ollama -------------------------------------------------------------------


class OllamaBackend(Backend):
    """Ollama local backend with optional model-specific prompt transform.

    The transform hook is the primary extension point. Qwen3.5-9B supports
    a `/no_think` directive that skips thinking-mode output — drastically
    cuts wall-clock latency (Session 0.5 data: Qwen thinking mode exceeded
    900s on Bug 3 × `wrap_completed`). Gemma passes through untransformed.

    The transform is ONLY applied at generate() time. Stored prompts in
    episodes remain canonical so cross-model comparison stays honest.
    """

    def __init__(
        self,
        model: str,
        *,
        prompt_transform: Callable[[str], str] | None = None,
        timeout: int = 600,
        num_ctx: int = 16384,
        temperature: float = 0.2,
    ) -> None:
        self.model = model
        self.name = model
        self.prompt_transform = prompt_transform or (lambda p: p)
        self.timeout = timeout
        self.num_ctx = num_ctx
        self.temperature = temperature

    def generate(self, prompt: str) -> GenerateResult:
        """Call Ollama with a HARD wall-clock guardrail.

        urllib's ``timeout`` parameter is a per-socket-op timeout, not a
        total-time ceiling. On slow-drip generations (observed on Bug 3 ×
        Gemma in Session 1: connection hung for 6+ min past the 900s
        urllib timeout, blocked in a C-level socket read that Python
        signals cannot interrupt), urllib will wait indefinitely as long
        as SOMETHING trickles through.

        This method adds a threading.Timer that force-closes the response
        object when the hard deadline passes. On hard timeout we raise
        TimeoutError from the main thread so the caller catches it as a
        normal exception and the loop can move on cleanly. The torn-down
        connection means the ``finally`` block in the loop will still run
        and reset the anneal-memory repo.
        """
        wire_prompt = self.prompt_transform(prompt)
        body = json.dumps(
            {
                "model": self.model,
                "prompt": wire_prompt,
                "stream": False,
                "keep_alive": "30m",
                "options": {
                    "temperature": self.temperature,
                    "num_ctx": self.num_ctx,
                },
            }
        ).encode("utf-8")

        req = Request(
            OLLAMA_ENDPOINT,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        t0 = time.time()

        resp_holder: dict[str, Any] = {"resp": None}
        aborted = threading.Event()

        def _abort() -> None:
            aborted.set()
            r = resp_holder.get("resp")
            if r is not None:
                try:
                    r.close()
                except Exception:
                    pass

        timer = threading.Timer(self.timeout, _abort)
        timer.daemon = True
        timer.start()
        try:
            resp = urlopen(req, timeout=self.timeout)
            resp_holder["resp"] = resp
            try:
                raw = resp.read()
            finally:
                try:
                    resp.close()
                except Exception:
                    pass
        except Exception as e:
            if aborted.is_set():
                raise TimeoutError(
                    f"Ollama call exceeded hard wall-clock timeout of {self.timeout}s"
                ) from e
            raise
        finally:
            timer.cancel()

        if aborted.is_set():
            raise TimeoutError(
                f"Ollama call exceeded hard wall-clock timeout of {self.timeout}s"
            )

        api = json.loads(raw.decode("utf-8"))
        latency = time.time() - t0

        eval_ns = api.get("eval_duration", 0)
        eval_count = api.get("eval_count", 0)
        rate = (eval_count / (eval_ns / 1e9)) if eval_ns else 0.0

        return GenerateResult(
            text=api.get("response", ""),
            latency_s=latency,
            eval_tokens=eval_count,
            eval_rate_tok_s=rate,
            raw=api,
        )


# -- Prompt transforms --------------------------------------------------------


def qwen_no_think(prompt: str) -> str:
    """Prepend Qwen3.5's `/no_think` directive.

    Qwen3 thinking mode emits hidden `<think>...</think>` tokens before the
    visible answer. The Session 0.5 hypothesis was that `/no_think` would
    rescue Bug 3 × Qwen (which timed out at 900s single-shot) by cutting
    latency.

    **Session 1 empirical verdict: /no_think is a BAD default for this
    harness.** Tested directly on Apr 14:
    - Bug 1 × Qwen /no_think: 149.8s, 1563 tokens, MISSED THE BUG ENTIRELY
      (produced a fix that preserved the original `or` operator and added
      a dead-code special case). Session 0.5 baseline without /no_think
      got 112.2s and correctly identified the bug (then drifted on an
      unrelated operator). Quality regression without a latency win.
    - Bug 3 × Qwen /no_think: STILL timed out at 900s. Bottleneck isn't
      thinking mode, it's total generation time on an 18K-char prompt
      through a 6.6GB model.

    Kept here as opt-in because it's the canonical documented Qwen3
    directive and future experiments may want it. NOT applied by default
    in make_backend() — see that function's comment.
    """
    return f"/no_think\n\n{prompt}"


# -- Factory ------------------------------------------------------------------


def make_backend(model: str, *, qwen_no_think_opt_in: bool = False) -> OllamaBackend:
    """Construct the right backend for a model tag.

    Currently everything is Ollama. Model-specific transforms are selected
    from the tag. Future: route `gemini/*` tags to a Google AI Studio
    backend, `hf/*` tags to a llama.cpp/MLX backend, etc. — same contract.

    Qwen defaults to thinking mode (no `/no_think`) after the Session 1
    empirical finding that `/no_think` regresses fix quality on Bug 1 and
    fails to rescue Bug 3 anyway. Pass ``qwen_no_think_opt_in=True`` for
    experiments that explicitly want the directive.
    """
    tag = model.lower()
    if tag.startswith("qwen") and qwen_no_think_opt_in:
        return OllamaBackend(model, prompt_transform=qwen_no_think)
    return OllamaBackend(model)
