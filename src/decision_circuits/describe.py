"""A circuit in plain English.

`describe(circuit)` writes Markdown a person can read without knowing the DSL: the questions
the model is asked, in their own words; what each gate decides, and when it holds back
instead; and, for each mounted chip, what its pins are wired to. Pass a run's `gates` and
`answers` and each line also says what happened.

    print(c.describe())
    out = c.run(backend, state)
    print(c.describe(out["gates"], out["answers"]))

Written from the gate expressions, not the compiled helpers, so `(Q("a") | Q("b")) >= 0.6`
reads as "any of ..." rather than as a chain of numbered helper gates.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from decision_circuits.dsl import (
    And,
    Categorical,
    G,
    GateDef,
    Not,
    Or,
    Q,
    Threshold,
    answer_outcomes,
    child_mounts,
    expr_refs,
    local_name,
    mount_owner,
    question_text,
    verdict,
)

if TYPE_CHECKING:
    from decision_circuits.dsl import Circuit

KIND = {"noul": "yes/no", "choice": "pick one", "score": "scale", "multi": "pick any", "locate": "point to", "rank": "rank", "match": "match"}
POLICY = {"abstain": "it abstains", "escalate": "it escalates to a person"}


def describe(circuit: Circuit, results: Mapping[str, Any] | None = None, answers: Mapping[str, Any] | None = None) -> str:
    return _Writer(circuit, results or {}, answers or {}).write()


class _Writer:
    def __init__(self, circuit: Circuit, results: Mapping[str, Any], answers: Mapping[str, Any]):
        self.c = circuit
        self.results = results
        self.answers = answers
        self.gates = {g.name: g for g in circuit.gates}
        self.here: str | None = None  # the chip whose gate is being written

    # ---------------------------------------------------------- layout
    def write(self) -> str:
        from decision_circuits.chips import Chip

        c = self.c
        if isinstance(c, Chip):
            title = f"# Chip `{c.name}`" + (f" v{c.version}" if c.version else "")
            lines = [title, ""]
            if c.description:
                lines += [c.description, ""]
            if c.inputs:
                lines += ["**Input pins:** " + ", ".join(f"`{p}`" for p in c.inputs), ""]
            if c.params:
                lines += [
                    "**Params:** "
                    + ", ".join(f"`{{{p}}}`" for p in c.params)
                    + ", set when it is mounted (in question text as `{name}`, in the logic as `P(name)`)",
                    "",
                ]
            if c.outputs:
                lines += ["**Output pins:** " + ", ".join(f"`{p}`" for p in c.outputs), ""]
        else:
            lines = ["# Circuit", ""]
            n = len(c.questions)
            if n:
                lines += [f"The model answers {n} question{'s' if n != 1 else ''} about the state, in one request; the gates below decide in code.", ""]
            lines += self.outcome_summary()
        lines += self.section(None, depth=2)
        return "\n".join(lines).rstrip() + "\n"

    def section(self, ns: str | None, depth: int) -> list[str]:
        out: list[str] = []
        qs = [qid for qid in self.c.questions if mount_owner(self.c, qid) == ns]
        gs = [g for g in self.c.gates if mount_owner(self.c, g.name) == ns]
        h = "#" * depth
        if qs:
            out += [f"{h} Asks" if ns else f"{h} Questions", ""]
            out += [self.question_line(qid) for qid in qs] + [""]
        if gs:
            out += [f"{h} Decides" if ns else f"{h} Decisions", ""]
            for g in gs:
                out += self.gate_lines(g) + [""]
        for inner in child_mounts(self.c, ns):
            out += self.chip_block(inner, depth)
        return out

    def outcome_summary(self) -> list[str]:
        """After a run: where the circuit's own routes sent the state, first."""
        out = []
        for g in self.c.gates:
            r = self.results.get(g.name)
            if r is None or not isinstance(g.body, Categorical) or g.body.op != "route" or mount_owner(self.c, g.name):
                continue
            status = r.get("rules") or []
            if r.get("outcome") in ("decided", "default"):
                held = status.index("held") + 1 if "held" in status else None
                out.append(f"**Outcome ({g.name}):** **{r['value']}**" + (f", by rule {held}." if held else ", since no rule held."))
            else:
                n = status.index("unsure") + 1 if "unsure" in status else 0
                action, cond = g.body.rules[n - 1] if n else (None, None)
                unsure = [
                    f"`{ref}` ({self.results[ref]['p']:.0%})" if self.results[ref].get("p") is not None else f"`{ref}`"
                    for ref in (expr_refs(cond) if cond is not None else [])
                    if ref in self.results and self.results[ref].get("uncertain")
                ]
                cause = f"{' and '.join(unsure)} too close to call" if unsure else "too close to call"
                out.append(f"**Outcome ({g.name}):** ⚠ **escalated to a person**: rule {n} (**{action}**) hinges on {cause}.")
        return [*out, ""] if out else []

    def chip_block(self, ns: str, depth: int) -> list[str]:
        self.here = None
        m = self.c.mounts[ns]
        h = "#" * depth
        out = [f"{h} Chip `{ns}`: {m['chip']}", ""]
        if m["pins"]:
            out += ["Wired:", ""]
            out += [f"- `{pin}` ← {self.ref(target)}" for pin, target in m["pins"].items()] + [""]
        if m.get("outputs"):
            out += ["Outputs: " + ", ".join(f"`{ns}.{o}`" for o in m["outputs"]), ""]
        return out + self.section(ns, depth + 1)

    # ---------------------------------------------------------- lines
    def short(self, name: str) -> str:
        return local_name(self.c, name)

    def question_line(self, qid: str) -> str:
        q = self.c.questions[qid]
        line = f"- **{self.short(qid)}** ({KIND.get(q['type'], q['type'])}): {question_text(q)}"
        options = self.options(q)
        if options:
            line += f" *[{options}]*"
        if qid in self.answers:
            main, others = answer_outcomes(q, self.answers[qid])
            line += f" → **{main}**" + (f" ({', '.join(others)})" if others else "")
        return line

    @staticmethod
    def options(q: Mapping[str, Any]) -> str:
        crit = q.get("criteria")
        if q["type"] in ("choice", "multi", "rank") and isinstance(crit, Mapping):
            return " / ".join(map(str, crit))
        if q["type"] == "score" and isinstance(crit, list):
            return " < ".join(str(x) if isinstance(x, str) else str(i) for i, x in enumerate(crit))
        return ""

    def gate_lines(self, g: GateDef) -> list[str]:
        self.here = mount_owner(self.c, g.name)
        head = f"- **{self.short(g.name)}**: "
        body = g.body
        lines: list[str]
        if isinstance(body, Categorical):
            lines = self.categorical(body, g)
        else:
            tau, inner = (body.tau, body.inner) if isinstance(body, Threshold) else (0.5, body)
            items = self.items(inner)
            if items:
                lines = [f"yes when it is at least {tau:.0%} likely that {items[0]}:", *[f"  - {x}" for x in items[1]]]
            else:
                lines = [f"yes when it is at least {tau:.0%} likely that {self.say(inner)}."]
            lines.append(self.band_note(g, tau))
        lines[0] = head + lines[0]
        r = self.results.get(g.name)
        if r is not None:
            lines.append("  - " + self.outcome(r))
        return lines

    def band_note(self, g: GateDef, tau: float) -> str:
        lo, hi = max(0.0, tau - g.band_), min(1.0, tau + g.band_)
        return f"  - *Between {lo:.0%} and {hi:.0%} it is too close to call, so {self.policy(g)}.*"

    def policy(self, g: GateDef) -> str:
        if g.on_uncertain_ == "default":
            return f"it falls back to {g.default_!r}"
        return POLICY[g.on_uncertain_]

    def categorical(self, b: Categorical, g: GateDef) -> list[str]:
        conf = f" If its confidence is under {b.min_confidence:g}, {self.policy(g)}." if b.min_confidence else ""
        refs = b.inputs if b.inputs is not None else [b.input or ""]
        listed = [f"  - {self.ref(r)}" for r in refs]
        pooled_multi = b.inputs is None and self.qtype(b.input) == "multi"
        if b.op == "route":
            rules = [f"  {i}. **{action}** if {self.say(cond)}" for i, (action, cond) in enumerate(b.rules or [], 1)]
            held = "it escalates to a person" if g.on_uncertain_ == "escalate" else self.policy(g)
            return [
                "picks one action: the first of these that holds.",
                *rules,
                f"  - otherwise **{b.otherwise}**",
                f"  - *If a rule it needs is too close to call, {held} rather than trying the next one.*",
            ]
        if b.op == "argmax":
            return [f"picks the most likely answer to {self.ref(b.input or '')}.{conf}"]
        if b.op == "majority":
            return [f"picks the answer most of these agree on (more than half must).{conf}", *listed]
        if b.op == "verify":
            return [
                f"takes the answer to {self.ref(b.input or '')}, then checks it with {self.say(b.check)}; if the check is under {b.tau:.0%}, {self.policy(g)}.{conf}"
            ]
        if b.op == "order":
            cuts = ", ".join(f"{c:g}" for c in b.cutpoints or [])
            return [f"sorts {self.ref(b.input or '')} into {len(b.cutpoints or []) + 1} buckets by its expected level, cut at {cuts} (bucket 0 is the lowest)."]
        if b.op == "consistent":
            rel = {
                "same": "agree (one question asked two ways)",
                "complement": "are opposites (a question and its negation)",
                "implies": "are consistent: the first cannot be likelier than the second",
            }[b.relation or "same"]
            return [f"checks that these two {rel}; if they don't, {self.policy(g)}.", *listed]
        what = f"the options of {self.ref(b.input or '')}" if pooled_multi else "these"
        if b.op == "at_least":
            return [
                f"yes when it is at least {b.tau:.0%} likely that {b.k} or more of {what} hold (taken as independent):",
                *([] if pooled_multi else listed),
                self.band_note(g, b.tau),
            ]
        if b.op == "count":
            return [f"counts how many of {what} hold (taken as independent); the most likely count wins.{conf}", *([] if pooled_multi else listed)]
        return [f"{b.op} over {what}", *listed]

    def outcome(self, r: Mapping[str, Any]) -> str:
        word, pct, _ = verdict(r)
        if r.get("outcome") == "escalate":
            return f"⚠ **Escalated to a person**{pct}: too close to call."
        if r.get("outcome") == "abstain":
            return f"⚠ **Abstained**{pct}: too close to call."
        extra = " (the fallback; it was too close to call)" if r.get("outcome") == "default" else ""
        return f"Result: **{word}**{pct}{extra}."

    # ---------------------------------------------------------- expressions
    def qtype(self, ref: str | None) -> str | None:
        q = self.c.questions.get((ref or "").partition(":")[0])
        return q["type"] if q else None

    def items(self, e: Any) -> tuple[str, list[str]] | None:
        """A top-level AND / OR with three or more parts, as a lead-in and a bulleted list."""
        if isinstance(e, And | Or) and len(e.parts) >= 3:
            return ("all of these hold" if isinstance(e, And) else "any of these holds", [self.say(p) for p in e.parts])
        return None

    def say(self, e: Any) -> str:
        if isinstance(e, Q | G):
            return self.ref(e.ref)
        if isinstance(e, Not):
            return f"not ({self.say(e.inner)})"
        if isinstance(e, And | Or):
            return (" and " if isinstance(e, And) else " or ").join(self.say(p) for p in e.parts)
        if isinstance(e, Threshold):
            return f"({self.say(e.inner)}, at least {e.tau:.0%} likely)"
        return repr(e)

    def ref(self, ref: str) -> str:
        base, _, opt = ref.partition(":")
        q = self.c.questions.get(base)
        if q is not None:
            text = f"“{question_text(q)}”"
            if not opt:
                return f"the answer to {text} is yes" if q["type"] == "noul" else text
            if q["type"] == "score":
                levels = q.get("criteria") or []
                i = int(opt) if opt.isdigit() else -1
                name = levels[i] if 0 <= i < len(levels) and isinstance(levels[i], str) else f"level {opt}"
                return f"the answer to {text} is {name}"
            if q["type"] == "locate" and opt == "none":
                return f"nothing in the state answers {text}"
            return f"the answer to {text} is {opt}" if q["type"] != "multi" else f"the answers to {text} include {opt}"
        if base in self.gates:
            name = f"the `{self.short(base) if mount_owner(self.c, base) == self.here else base}` decision"
            if not opt:
                return f"{name} is yes"
            return f"{name} is {opt}" if opt not in ("True", "False") else f"{name} is {'yes' if opt == 'True' else 'no'}"
        return f"the `{base}` input pin is {opt or 'yes'}"
