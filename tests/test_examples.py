"""The offline examples must keep running; they are the docs."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"
OFFLINE = ["01_first_circuit.py", "03_your_own_backend.py", "07_your_own_integration.py", "08_refund_desk.py"]


@pytest.mark.parametrize("name", OFFLINE)
def test_offline_example_runs(name: str):
    env = {"PATH": "/usr/bin:/bin", "PYTHONPATH": str(EXAMPLES.parent / "src")}
    r = subprocess.run([sys.executable, str(EXAMPLES / name)], capture_output=True, text=True, timeout=60, env=env, check=False)
    assert r.returncode == 0, r.stderr[-2000:]
    assert r.stdout.strip()


def test_refund_desk_routes_every_ticket_somewhere():
    import importlib.util

    spec = importlib.util.spec_from_file_location("refund_desk", EXAMPLES / "08_refund_desk.py")
    assert spec and spec.loader
    desk = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(desk)
    backend = desk.WithFacts(desk.Scripted(), desk.TODAY)
    decisions = {}
    for tid, text, order in desk.TICKETS:
        out = desk.c.run(backend, {"ticket_id": tid, "ticket": text, "order": order})
        decisions[tid] = desk.decide(out["gates"])
    assert decisions["T1"] == "AUTO REFUND"
    assert decisions["T2"] == "AGENT REVIEW"  # eligible, clean, but $180
    assert decisions["T3"].startswith("HUMAN")  # abuse p=0.62 sits inside the band
    assert decisions["T5"].startswith("HUMAN")
