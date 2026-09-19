"""The same circuits, but the state is a receipt image or a recorded call.

    export CIRCUIT_API_KEY=dc-...        # free key from https://decisioncircuits.com/#api
    uv run python examples/12_images_and_audio.py

Nothing about a circuit changes when the state is an image or a clip. The
questions are typed the same way, the gates are the same code, and the
model behind them is chosen by name: circuit-vl-4b reads images,
circuit-audio-7b listens. `Image(...)` and `Audio(...)` are the whole
state; a path, a URL, or bytes, plus an optional caption.
"""

import os
import sys
from pathlib import Path

from decision_circuits import Audio, Circuit, Image, Q, argmax
from decision_circuits.backends import SystemOne

key = os.environ.get("CIRCUIT_API_KEY")
if not key:
    sys.exit("set CIRCUIT_API_KEY (get one at https://decisioncircuits.com/#api)")
api = SystemOne("https://api.decisioncircuits.com/v1/systemone", api_key=key)
data = Path(__file__).parent / "data"

# 1. A receipt: extract the total, check it, decide whether to file it.
receipt = Circuit()
receipt.choice("total", "What is the total on the receipt?", {"100.52": None, "90.47": None, "7.50": None, "103.62": None, "Cannot tell": None})
receipt.noul("card", "Was it paid by card?")
receipt.gate("total", argmax("total", min_confidence=0.5))
receipt.gate("file", Q("card") >= 0.6, band=0.1, on_uncertain="escalate")

out = receipt.run(api, Image(data / "receipt.png"), model="circuit-vl-4b")
print("receipt:")
for name, g in out["gates"].items():
    print(f"  {name:<6} value={g['value']!r:<10} outcome={g['outcome']:<9} p={g.get('p')}")

# 2. A support call: route it and decide whether a person should hear it.
call = Circuit()
call.choice("kind", "Which kind of water-utility call is this?", {"billing": None, "metering": None, "taste and odor": None, "outage": None, "leak": None})
call.noul("urgent", "Does the caller need something done today?")
call.gate("route", argmax("kind", min_confidence=0.5))
call.gate("human", (Q("urgent") | Q("kind")["leak"]) >= 0.6, band=0.1, on_uncertain="escalate")

out = call.run(api, Audio(data / "support_call.wav", text="Inbound call, Tuesday 9:14"), model="circuit-audio-7b")
print("call:")
for name, g in out["gates"].items():
    print(f"  {name:<6} value={g['value']!r:<10} outcome={g['outcome']:<9} p={g.get('p')}")

# The first call to each model after a quiet spell waits for a GPU to spin up (about a minute); warm calls take under a second.
