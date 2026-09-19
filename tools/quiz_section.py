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


QUIZ_JS = r"""  // The quiz: one card at a time, on the clock. Answer, then see the label, the circuit model's answer and time, and a chat model's answer and time.
  const qz = __DATA__;
  const cards = [...document.querySelectorAll('.qz')];
  const fmt = ms => ms == null ? '?' : ms < 1000 ? Math.round(ms) + ' ms' : (ms / 1000).toFixed(1) + ' s';
  let ix = -1, you = 0, models = 0, llms = 0, yourMs = 0, circuitMs = 0, llmMs = 0, t0 = 0, tick = null;
  cards.forEach(c => { c.hidden = true; });
  function showCard(i) {
    ix = i; const card = cards[i]; card.hidden = false; card.scrollIntoView({ block: 'start', behavior: 'smooth' });
    t0 = performance.now(); const clock = card.querySelector('.clock');
    tick = setInterval(() => { clock.textContent = ((performance.now() - t0) / 1000).toFixed(1) + ' s'; }, 100);
  }
  function summary(done) {
    const n = ix + 1;
    $('qz-score').textContent = 'You ' + you + '/' + n + ' in ' + fmt(yourMs) + ' · circuit models ' + models + '/' + n + ' in ' + fmt(circuitMs) + ' · chat model ' + llms + '/' + n + ' in ' + fmt(llmMs)
      + (done ? (you > models ? '. You beat the models on answers.' : you === models ? '. Level on answers.' : '. The models win on answers.') + (yourMs < circuitMs ? ' And on time.' : ' The circuit was ' + Math.round(yourMs / Math.max(1, circuitMs)) + '× faster.') : '');
  }
  cards.forEach((card, i) => {
    const d = qz[i];
    card.querySelectorAll('.opts button').forEach(btn => btn.addEventListener('click', () => {
      if (i !== ix) return;
      clearInterval(tick); const took = performance.now() - t0; yourMs += took;
      card.querySelector('.clock').textContent = (took / 1000).toFixed(1) + ' s';
      const k = btn.dataset.k;
      card.querySelectorAll('.opts button').forEach(b => { b.disabled = true; if (b.dataset.k === d.ref) b.classList.add('right'); });
      btn.classList.add('you');
      const youRight = k === d.ref, modelRight = d.pick === d.ref, llmRight = d.llm && d.llm.answer === d.ref;
      you += youRight; models += modelRight; llms += llmRight; circuitMs += d.circuit_ms || 0; llmMs += (d.llm && d.llm.wall_ms) || 0;
      let v = (youRight ? '<b>You got it</b>' : '<b>Not this one</b>') + ' in <span class="t">' + fmt(took) + '</span>. Label: <b>' + d.ref + '</b>.<br>'
        + d.model + ': <b>' + d.pick + '</b> at p ' + d.p.toFixed(2) + (modelRight ? '' : ', wrong') + ', <span class="t">' + fmt(d.circuit_ms) + '</span> on the server (' + fmt(d.rtt_ms) + ' round trip from a laptop).<br>';
      if (d.llm && d.llm.answer != null) {
        v += (d.llm.model === 'whisper-small + claude' ? 'Whisper, then Claude on the transcript' : 'Claude, via the CLI') + ': <b>' + d.llm.answer + '</b>' + (llmRight ? '' : ', wrong') + ', <span class="t">' + fmt(d.llm.wall_ms) + '</span>'
          + (d.llm.transcribe_ms ? ' (' + fmt(d.llm.transcribe_ms) + ' of it transcribing: “' + d.llm.transcript + '”)' : '') + '.';
      } else if (d.llm && d.llm.note) { v += 'Chat model: ' + d.llm.note + '.'; }
      card.querySelector('.verdict').innerHTML = v;
      const next = card.querySelector('.next');
      if (i + 1 < cards.length) { next.hidden = false; next.addEventListener('click', () => { next.hidden = true; showCard(i + 1); }, { once: true }); summary(false); }
      else { summary(true); }
    }));
  });
  $('qz-start').addEventListener('click', () => { $('qzs').classList.add('live'); $('qz-score').textContent = ''; showCard(0); });
"""


def main() -> None:
    items = json.loads(ITEMS.read_text())
    cards, data = [], []
    for i, it in enumerate(items):
        q = it["question"]
        ref = max(it["ref"], key=it["ref"].get)
        pick, p = model_pick(q, it["answer"])
        t = it.get("timing", {})
        data.append(
            {
                "ref": ref,
                "model": it["model"],
                "pick": pick,
                "p": round(p, 2),
                "circuit_ms": t.get("circuit", {}).get("server_ms"),
                "rtt_ms": t.get("circuit", {}).get("rtt_ms"),
                "llm": t.get("llm"),
            }
        )
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
            f'        <button type="button" class="next" hidden>Next</button>\n'
            f"      </article>"
        )
    page = PAGE.read_text()
    page, n = re.subn(
        r'(    <div class="qzs"[^>]*>\n).*?(\n    </div>\n  </section>)', lambda m: m.group(1) + "\n".join(cards) + m.group(2), page, count=1, flags=re.DOTALL
    )
    assert n == 1, "quiz cards block not found"
    js = QUIZ_JS.replace("__DATA__", json.dumps(data))
    page, n = re.subn(r"  // quiz:start\n.*?  // quiz:end\n", "  // quiz:start\n" + js + "  // quiz:end\n", page, count=1, flags=re.DOTALL)
    assert n == 1, "quiz script markers not found"
    PAGE.write_text(page)
    print(f"{len(items)} items; models right on {sum(d['pick'] == d['ref'] for d in data)}")


if __name__ == "__main__":
    main()
