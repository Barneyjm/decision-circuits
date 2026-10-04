"""Chips: a sub-circuit packaged with pins, built once and mounted twice. Offline.

A support desk wants the same "this is heating up" logic on two sides of a conversation: the
customer and the agent. Instead of writing the gates twice, package them as a chip with input
pins and a `who` param, test the chip on its own, then wire two copies into one circuit. The
param goes into the chip's own question, so each copy asks about its own side.

    uv run python examples/14_chips.py
"""

import json

from decision_circuits import Chip, Circuit, G, Q, at_least

# 1. The chip: three input pins, one param, one question of its own, one output pin.
heat = Chip("heat", inputs=["hostile", "money", "pii"], outputs=["hot"], params=["who"], version="1.0", description="Two or more signs of a dispute heating up")
heat.noul("threat", "Does {who} threaten legal action, a chargeback or a public complaint?")
heat.gate("signs", at_least(2, "hostile", "money", "pii", "threat"))
heat.gate("hot", G("signs") >= 0.6, on_uncertain="escalate")

# 2. Test vectors ship with the chip. A number is a noul's P(yes); a dict is a choice.
heat.add_test({"hostile": 0.9, "money": 0.9, "pii": 0.1, "threat": 0.2}, {"hot": True}, name="angry about money")
heat.add_test({"hostile": 0.1, "money": 0.2, "pii": 0.1, "threat": 0.1}, {"hot": False}, name="calm")
heat.add_test({"hostile": 0.7, "money": 0.5, "pii": 0.1, "threat": 0.3}, {"hot": "escalate"}, name="on the fence")
print("chip tests failing:", heat.test())

# 3. The host circuit asks its own questions and mounts the chip twice.
c = Circuit()
c.noul("customer_hostile", "Is the customer's message hostile?")
c.noul("agent_hostile", "Is the agent's reply hostile or dismissive?")
c.choice("topic", "What is the conversation about?", {"refund": None, "billing": None, "other": None})
c.noul("pii", "Does the conversation contain personal information about a private individual?")

money = Q("topic")["refund"] | Q("topic")["billing"]
c.gate("money", money >= 0.5)
customer = c.mount(heat, "customer", {"hostile": "customer_hostile", "money": G("money"), "pii": "pii"}, params={"who": "the customer"})
agent = c.mount(heat, "agent", {"hostile": "agent_hostile", "money": G("money"), "pii": "pii"}, params={"who": "the agent"})
c.gate("supervisor", (customer["hot"] | agent["hot"]) >= 0.5, on_uncertain="escalate")

print("\nquestions sent to the model in one request:", list(c.questions))
print("  customer.threat:", c.questions["customer.threat"]["instructions"])
print("  agent.threat:   ", c.questions["agent.threat"]["instructions"])

# 4. Answers, normally from a model. Each mounted copy asked its own threat question.
answers = {
    "customer_hostile": {"type": "noul", "noul": 0.92},
    "agent_hostile": {"type": "noul", "noul": 0.08},
    "topic": {"type": "choice", "choice": "refund", "probabilities": {"refund": 0.81, "billing": 0.12, "other": 0.07}, "confidence": 0.5},
    "pii": {"type": "noul", "noul": 0.05},
    "customer.threat": {"type": "noul", "noul": 0.66},
    "agent.threat": {"type": "noul", "noul": 0.02},
}
results = c.evaluate(answers)
for name in ("money", "customer.hot", "agent.hot", "supervisor"):
    r = results[name]
    print(f"{name:<13} -> {r['value']!s:<6} p={r['p']:.2f}  {r['outcome']}")

# 5. A chip is plain JSON: commit it, publish it in a package, load it elsewhere.
blob = json.dumps(heat.to_dict())
again = Circuit.from_dict(json.loads(blob))
print(f"\n{len(blob)} bytes of JSON; loads back as {type(again).__name__} {again.name!r}, tests failing: {again.test()}")

# 6. The diagram draws each mounted chip as a box.
print("\n" + c.to_mermaid(results=results, answers=answers, plain=True))
