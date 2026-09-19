"""The refund desk (example 08), with an OpenAI-compatible model answering
through logprobs.

    pip install "decision-circuits[openai]"
    export OPENAI_API_KEY=...                       # OpenAI
    uv run python examples/10_refund_desk_with_openai_logprobs.py

    # or any OpenAI-compatible server with logprobs (Fireworks, Together, vLLM):
    export FIREWORKS_API_KEY=...
    uv run python examples/10_refund_desk_with_openai_logprobs.py fireworks

The backend letters each question's options, asks for one letter, and
reads the next-token log-probabilities over the letters. Those are
measured probabilities rather than stated ones, one short request per
question, run in parallel. Everything else is example 08.
"""

import importlib.util
import os
import sys
from pathlib import Path

from decision_circuits.backends import OpenAILogprobs

if len(sys.argv) > 1 and sys.argv[1] == "fireworks":
    key = os.environ.get("FIREWORKS_API_KEY") or sys.exit("set FIREWORKS_API_KEY")
    backend = OpenAILogprobs(model="accounts/fireworks/models/qwen3-8b", base_url="https://api.fireworks.ai/inference/v1", api_key=key)
    label = "Qwen3-8B on Fireworks, logprobs"
else:
    os.environ.get("OPENAI_API_KEY") or sys.exit("set OPENAI_API_KEY (or pass `fireworks`)")
    backend = OpenAILogprobs(model="gpt-4.1-mini")
    label = "gpt-4.1-mini, logprobs"

spec = importlib.util.spec_from_file_location("refund_desk", Path(__file__).with_name("08_refund_desk.py"))
assert spec and spec.loader
desk = importlib.util.module_from_spec(spec)
spec.loader.exec_module(desk)

desk.run_desk(backend, label, diagram=False)
