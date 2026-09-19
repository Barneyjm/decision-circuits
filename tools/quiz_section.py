"""Rebuild the site's benchmark quiz from docs/bench/items.json.

    uv run python tools/quiz_section.py

items.json holds real grid items with the family's answers from the hosted
API (see the circuit repo for how they are fetched). This rewrites the
<section id="quiz"> cards and the `const qz = [...]` data in docs/index.html.
"""

from __future__ import annotations

import html
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PAGE = ROOT / "docs" / "index.html"
ITEMS = ROOT / "docs" / "bench" / "items.json"


def options(q):
    if q["type"] == "noul":
        return [("yes", "Yes"), ("no", "No")]
    if q["type"] == "choice":
        return [(k, k) for k in q["criteria"]]
    return [(str(i), lvl) for i, lvl in enumerate(q["criteria"])]


def model_pick(q, a):
    if q["type"] == "noul":
        return ("yes" if a["noul"] >= 0.5 else "no", max(a["noul"], 1 - a["noul"]))
    if q["type"] == "choice":
        return (a["choice"], a["probabilities"][a["choice"]])
    k = max(a["probabilities"], key=a["probabilities"].get)
    return (k, a["probabilities"][k])


def state_html(it):
    if it["kind"] == "image":
        cap = f'<p class="cap">{html.escape(it["text"])}</p>' if it.get("text") else ""
        return f'<img src="{it["media"]}" alt="a rendered document from the vision benchmark" loading="lazy">{cap}'
    if it["kind"] == "audio":
        return f'<audio controls preload="none" src="{it["media"]}"></audio>'
    st = it["state"]
    if isinstance(st, str):
        return f'<pre class="state">{html.escape(st)}</pre>'
    if isinstance(st, list) and st and isinstance(st[0], dict) and "role" in st[0]:
        return '<div class="thread">' + "".join(f"<p><b>{html.escape(m['role'])}</b> {html.escape(m['text'])}</p>" for m in st) + "</div>"
    return f'<pre class="state">{html.escape(json.dumps(st, ensure_ascii=False, indent=1))}</pre>'


def main() -> None:
    items = json.loads(ITEMS.read_text())
    cards, data = [], []
    for i, it in enumerate(items):
        q = it["question"]
        ref = max(it["ref"], key=it["ref"].get)
        pick, p = model_pick(q, it["answer"])
        data.append({"ref": ref, "model": it["model"], "pick": pick, "p": round(p, 2)})
        crit = q.get("criteria") if q["type"] != "score" else {}
        btns = "".join(
            f'<button type="button" data-k="{html.escape(k)}" title="{html.escape(str((crit or {}).get(k) or ""))}">{html.escape(lbl)}</button>'
            for k, lbl in options(q)
        )
        cards.append(
            f'      <article class="qz" data-i="{i}">\n'
            f'        <p class="src">{html.escape(it["label"])} · answered by {html.escape(it["model"])}</p>\n'
            f"        {state_html(it)}\n"
            f'        <p class="ask">{html.escape(q["instructions"])}</p>\n'
            f'        <div class="opts">{btns}</div>\n'
            f'        <p class="verdict" aria-live="polite"></p>\n'
            f"      </article>"
        )
    page = PAGE.read_text()
    page, n = re.subn(
        r'(    <div class="qzs">\n).*?(\n    </div>\n  </section>)', lambda m: m.group(1) + "\n".join(cards) + m.group(2), page, count=1, flags=re.DOTALL
    )
    assert n == 1, "quiz cards block not found"
    page, n = re.subn(r"  const qz = \[.*?\];\n", "  const qz = " + json.dumps(data) + ";\n", page, count=1, flags=re.DOTALL)
    assert n == 1, "qz data not found"
    PAGE.write_text(page)
    print(f"{len(items)} items; models right on {sum(d['pick'] == d['ref'] for d in data)}")


if __name__ == "__main__":
    main()
