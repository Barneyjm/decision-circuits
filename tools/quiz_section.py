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


QUIZ_JS = r"""  // The quiz: a carousel. Each card starts its clock when it slides in and stops it when you pick; the last card averages.
  const qz = __DATA__;
  const track = $('track'), slides = [...track.children], cards = slides.filter(s => s.dataset.i !== undefined);
  const fmt = ms => ms == null ? '?' : ms < 1000 ? Math.round(ms) + ' ms' : (ms / 1000).toFixed(1) + ' s';
  let at = 0, t0 = 0, tick = null;
  const tally = { you: 0, circuit: 0, llm: 0, yourMs: 0, circuitMs: 0, llmMs: 0, n: 0 };
  function goTo(i) {
    at = i; track.style.transform = 'translateX(' + (-100 * i) + '%)';
    track.style.height = slides[i].offsetHeight + 'px';
    slides.forEach((s, j) => { s.setAttribute('aria-hidden', j === i ? 'false' : 'true'); });
    const card = slides[i];
    if (card.dataset.i !== undefined) {
      t0 = performance.now(); const clock = card.querySelector('.clock');
      clearInterval(tick); tick = setInterval(() => { clock.textContent = ((performance.now() - t0) / 1000).toFixed(1) + ' s'; }, 100);
    }
    if (card.classList.contains('end')) finish();
  }
  cards.forEach(card => {
    const d = qz[+card.dataset.i];
    card.querySelectorAll('.opts button').forEach(btn => btn.addEventListener('click', () => {
      if (slides[at] !== card || btn.disabled) return;
      clearInterval(tick); const took = performance.now() - t0;
      card.querySelector('.clock').textContent = (took / 1000).toFixed(1) + ' s';
      card.querySelectorAll('.opts button').forEach(b => { b.disabled = true; if (b.dataset.k === d.ref) b.classList.add('right'); });
      btn.classList.add('you');
      const k = btn.dataset.k, youRight = k === d.ref, modelRight = d.pick === d.ref, llmRight = !!(d.llm && d.llm.answer === d.ref);
      tally.n += 1; tally.you += youRight; tally.circuit += modelRight; tally.llm += llmRight; tally.yourMs += took; tally.circuitMs += d.circuit_ms || 0; tally.llmMs += (d.llm && d.llm.wall_ms) || 0;
      let v = '<b>' + (youRight ? 'You got it' : 'Not this one') + '</b> in <span class="t">' + fmt(took) + '</span>. Label: <b>' + d.ref + '</b>.<br>'
        + d.model + ': <b>' + d.pick + '</b> at p ' + d.p.toFixed(2) + (modelRight ? '' : ', wrong') + ', <span class="t">' + fmt(d.circuit_ms) + '</span> on the server, ' + fmt(d.rtt_ms) + ' round trip from a laptop.<br>';
      if (d.llm && d.llm.answer != null) {
        v += (d.llm.model === 'whisper-small + claude' ? 'Whisper, then Claude on the transcript' : 'Claude, via the CLI') + ': <b>' + d.llm.answer + '</b>' + (llmRight ? '' : ', wrong') + ', <span class="t">' + fmt(d.llm.wall_ms) + '</span>'
          + (d.llm.transcribe_ms ? ' (' + fmt(d.llm.transcribe_ms) + ' of it transcribing: “' + d.llm.transcript + '”)' : '') + '.';
      }
      card.querySelector('.verdict').innerHTML = v;
      card.querySelector('.next').hidden = false;
    }));
    card.querySelector('.next').addEventListener('click', () => goTo(at + 1));
  });
  function finish() {
    const n = tally.n || 1;
    $('qz-final').textContent = 'You ' + tally.you + '/' + tally.n + ', circuit models ' + tally.circuit + '/' + tally.n + ', chat model ' + tally.llm + '/' + tally.n + '.';
    const row = (who, right, ms) => '<tr><td>' + who + '</td><td class="n">' + right + '/' + tally.n + '</td><td class="n">' + fmt(ms / n) + '</td><td class="n">' + fmt(ms) + '</td></tr>';
    $('qz-table').innerHTML = '<tr><th></th><th>right</th><th>average per question</th><th>total</th></tr>' + row('you', tally.you, tally.yourMs) + row('circuit models, server time', tally.circuit, tally.circuitMs) + row('chat model, wall time', tally.llm, tally.llmMs);
  }
  $('qz-start').addEventListener('click', () => goTo(1));
  $('qz-again').addEventListener('click', () => location.reload());
  goTo(0);
  window.addEventListener('resize', () => { track.style.height = slides[at].offsetHeight + 'px'; });
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
            f'        <div class="head"><p class="src">{i + 1} of {len(items)} · {html.escape(it["label"])}</p><span class="clock">0.0 s</span></div>\n'
            f"        {state_html(it)}\n"
            f'        <p class="ask">{html.escape(q["instructions"])}</p>\n'
            f'        <div class="opts">{btns}</div>\n'
            f'        <p class="verdict" aria-live="polite"></p>\n'
            f'        <button type="button" class="next" hidden>Next</button>\n'
            f"      </article>"
        )
    page = PAGE.read_text()
    intro = (
        '      <article class="qz intro">\n'
        '        <p class="ask">Eight questions from the benchmark, one at a time, on the clock.</p>\n'
        '        <p class="blurb">Each card starts timing when it appears and stops when you pick. Then it shows what the circuit model answered and how long it took, and what a chat model answered and how long that took. The last card averages it all.</p>\n'
        '        <button type="button" class="next" id="qz-start">Start the quiz</button>\n'
        "      </article>"
    )
    end = (
        '      <article class="qz end">\n'
        '        <p class="ask" id="qz-final"></p>\n'
        '        <div class="tbl"><table id="qz-table"></table></div>\n'
        '        <button type="button" class="next" id="qz-again">Play again</button>\n'
        "      </article>"
    )
    page, n = re.subn(
        r'(    <div class="track" id="track">\n).*?(\n    </div>\n    </div>\n  </section>)',
        lambda m: m.group(1) + "\n".join([intro, *cards]) + "\n" + end + m.group(2),
        page,
        count=1,
        flags=re.DOTALL,
    )
    assert n == 1, "quiz cards block not found"
    js = QUIZ_JS.replace("__DATA__", json.dumps(data))
    page, n = re.subn(r"  // quiz:start\n.*?  // quiz:end\n", lambda _m: "  // quiz:start\n" + js + "  // quiz:end\n", page, count=1, flags=re.DOTALL)
    assert n == 1, "quiz script markers not found"
    PAGE.write_text(page)
    print(f"{len(items)} items; models right on {sum(d['pick'] == d['ref'] for d in data)}")


if __name__ == "__main__":
    main()
