"""Backends: ways to get calibrated answers for a circuit's questions.

    SystemOne       any System One HTTP server (TypeSafe's Jev, s1proto); stdlib only
    OpenAILogprobs  OpenAI-compatible chat APIs with logprobs (OpenAI, Fireworks, vLLM); needs `openai`
    Anthropic       Claude via tool use, stated or sampled probabilities; needs `anthropic`

Provider modules import their SDKs lazily, so importing this package
never requires them. Write your own by satisfying
`decision_circuits.types.Backend`: one method, `answer(state, questions,
*, model=None) -> answers`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from decision_circuits.backends.systemone import SystemOne, SystemOneError

if TYPE_CHECKING:
    from decision_circuits.backends.anthropic import Anthropic
    from decision_circuits.backends.openai import OpenAILogprobs

__all__ = ["Anthropic", "OpenAILogprobs", "SystemOne", "SystemOneError"]


def __getattr__(name: str) -> Any:
    if name == "OpenAILogprobs":
        from decision_circuits.backends.openai import OpenAILogprobs

        return OpenAILogprobs
    if name == "Anthropic":
        from decision_circuits.backends.anthropic import Anthropic

        return Anthropic
    raise AttributeError(name)
