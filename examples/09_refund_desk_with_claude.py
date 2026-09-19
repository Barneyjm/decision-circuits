"""The refund desk (example 08), with Claude answering the model questions.

    pip install "decision-circuits[anthropic]"
    export ANTHROPIC_API_KEY=...
    uv run python examples/09_refund_desk_with_claude.py

Claude's API exposes no token probabilities, so the backend has two
modes and this example runs both so you can compare them on the same
tickets:

  stated    one call; Claude reports a probability per option through a tool
  sampled   k calls at temperature 1, one pick each; the distribution is the vote

The gates, thresholds, and code facts are exactly example 08's. Only
the backend changed. Watch the "checked" and "risky" columns: stated
probabilities tend to be more extreme than sampled ones, which changes
what lands inside an uncertainty band.
"""

import importlib.util
import os
import sys
from pathlib import Path

from decision_circuits.backends import Anthropic

if not os.environ.get("ANTHROPIC_API_KEY"):
    sys.exit("set ANTHROPIC_API_KEY")

spec = importlib.util.spec_from_file_location("refund_desk", Path(__file__).with_name("08_refund_desk.py"))
assert spec and spec.loader
desk = importlib.util.module_from_spec(spec)
spec.loader.exec_module(desk)

desk.run_desk(Anthropic(model="claude-sonnet-5", mode="stated"), "Claude, stated probabilities", diagram=False)
print("\n" + "=" * 70)
desk.run_desk(Anthropic(model="claude-haiku-4-5-20251001", mode="sampled", k=7), "Claude Haiku, 7 samples", diagram=False)
