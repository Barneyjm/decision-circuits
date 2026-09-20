"""Rebuild the site's benchmark quiz from docs/bench/items.json.

    uv run python tools/quiz_section.py

items.json holds real grid items with the family's answers from the hosted
API (see the circuit repo for how they are fetched). This rewrites the card
markup between the quiz:html markers and the `const qz = [...]` data in
docs/index.html.

The quiz is one static card. Top to bottom: where you are and the clock,
the question, the answer buttons, then previous/next. Picking stops the
clock and prints the breakdown under the buttons; the last Next shows the
whole board.
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


CARD = """    <div class="qz" id="qzcard">
      <div class="qz-intro" id="qz-intro">
        <p class="ask">__N__ questions from the benchmark, one at a time, on the clock.</p>
        <p class="blurb">The clock starts when a question appears and stops when you pick. Each answer shows what the circuit model said and how long it took, against a chat model on the same question. The last card puts every answer and every time side by side.</p>
        <button type="button" class="next" id="qz-start">Start the quiz</button>
      </div>

      <div class="qz-play" id="qz-play" hidden>
        <div class="head">
          <p class="src"><span id="qz-pos"></span> · <span id="qz-label"></span></p>
          <span class="clock" id="qz-clock">0.0 s</span>
        </div>
        <div class="qz-body" id="qz-body"></div>
        <p class="ask" id="qz-ask"></p>
        <div class="opts" id="qz-opts"></div>
        <p class="verdict" id="qz-verdict" aria-live="polite"></p>
        <div class="nav">
          <button type="button" class="next ghost" id="qz-prev">Previous</button>
          <button type="button" class="next" id="qz-next">Next question</button>
        </div>
      </div>

      <div class="qz-results" id="qz-results" hidden>
        <p class="ask" id="qz-final"></p>
        <p class="blurb" id="qz-speed"></p>
        <div class="tbl"><table class="sheet" id="qz-board"></table></div>
        <div class="tbl"><table id="qz-table"></table></div>
        <div class="nav">
          <button type="button" class="next ghost" id="qz-back">Back to the questions</button>
          <button type="button" class="next" id="qz-again">Play again</button>
        </div>
      </div>

__TEMPLATES__
    </div>
"""

QUIZ_JS = r"""// The quiz: one static card, in its own scope.
// The clock runs while a question is open, picking stops it, and the last Next shows the board.
(function () {
  const qz = __DATA__;
  const N = qz.length, picks = qz.map(() => null);
  let at = 0, t0 = 0, tick = null;
  const fmt = ms => ms == null ? '?' : (ms / 1000).toFixed(ms < 1000 ? 2 : 1) + ' s';   // seconds, always, so the gap reads at a glance
  const secs = fmt;
  const ratio = (slow, fast) => !slow || !fast ? null : slow / fast;
  const xs = r => (r >= 10 ? Math.round(r) : r.toFixed(1)) + '×';
  const esc = s => String(s == null ? '—' : s).replace(/[&<>]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;' })[c]);
  const mark = ok => '<span class="' + (ok ? 'ok' : 'no') + '">' + (ok ? '✓' : '✗') + '</span>';
  const right = (d, g) => !!g && g.k === d.ref;
  const lab = (d, k) => { const o = d.opts.find(o => o.k === k); return o ? o.l : k; };

  function stopClock() { clearInterval(tick); tick = null; }
  function startClock() {
    stopClock(); t0 = performance.now(); $('qz-clock').textContent = '0.0 s';
    tick = setInterval(() => { $('qz-clock').textContent = secs(performance.now() - t0); }, 100);
  }
  function verdict(i) {
    const d = qz[i], g = picks[i], modelRight = d.pick === d.ref, llmRight = !!(d.llm && d.llm.answer === d.ref);
    const vsYou = ratio(g.ms, d.circuit_ms), vsLlm = ratio(d.llm && d.llm.wall_ms, d.circuit_ms);
    let v = '<b>' + (right(d, g) ? 'You got it' : 'Not this one') + '</b> in <span class="t">' + fmt(g.ms) + '</span>. Label: <b>' + esc(lab(d, d.ref)) + '</b>.<br>'
      + esc(d.model) + ': <b>' + esc(lab(d, d.pick)) + '</b> at p ' + d.p.toFixed(2) + (modelRight ? '' : ', wrong') + ' in <span class="t">' + fmt(d.circuit_ms) + '</span> on the server'
      + (vsYou ? ', <b>' + xs(vsYou) + ' faster than you</b>' : '') + ' (' + fmt(d.rtt_ms) + ' round trip from a laptop).<br>';
    if (d.llm && d.llm.answer != null) {
      v += (d.llm.model === 'whisper-small + claude' ? 'Whisper, then Claude on the transcript' : 'Claude, via the CLI') + ': <b>' + esc(lab(d, d.llm.answer)) + '</b>' + (llmRight ? '' : ', wrong') + ' in <span class="t">' + fmt(d.llm.wall_ms) + '</span>'
        + (vsLlm ? ', <b>' + xs(vsLlm) + ' slower than the circuit model</b>' : '')
        + (d.llm.transcribe_ms ? ' (' + fmt(d.llm.transcribe_ms) + ' of it transcribing: “' + esc(d.llm.transcript) + '”)' : '') + '.';
    }
    return v;
  }
  function paint(i) {
    const d = qz[i], g = picks[i];
    [...$('qz-opts').children].forEach((b, j) => {
      b.disabled = true;
      if (d.opts[j].k === d.ref) b.classList.add('right');
      if (d.opts[j].k === g.k) b.classList.add('you');
    });
    $('qz-clock').textContent = secs(g.ms);
    $('qz-verdict').innerHTML = verdict(i);
  }
  function pick(i, k) {
    if (picks[i] || i !== at) return;
    stopClock(); picks[i] = { k: k, ms: performance.now() - t0 };
    paint(i); nav();
  }
  function nav() {
    $('qz-prev').disabled = at === 0;
    $('qz-next').disabled = !picks[at];
    $('qz-next').textContent = at === N - 1 ? 'See the results' : 'Next question';
  }
  function showQ(i) {
    at = i;
    $('qz-intro').hidden = true; $('qz-results').hidden = true; $('qz-play').hidden = false;
    const d = qz[i];
    $('qz-pos').textContent = 'Question ' + (i + 1) + ' of ' + N;
    $('qz-label').textContent = d.label;
    const body = $('qz-body');
    body.replaceChildren(document.querySelector('template[data-i="' + i + '"]').content.cloneNode(true));
    $('qz-ask').textContent = d.ask;
    const opts = $('qz-opts'); opts.replaceChildren();
    d.opts.forEach(o => {
      const b = document.createElement('button');
      b.type = 'button'; b.textContent = o.l; if (o.t) b.title = o.t;
      b.addEventListener('click', () => pick(i, o.k));
      opts.appendChild(b);
    });
    if (picks[i]) { stopClock(); paint(i); } else { $('qz-verdict').innerHTML = ''; startClock(); }
    nav();
  }
  function results() {
    stopClock();
    $('qz-play').hidden = true; $('qz-results').hidden = false;
    const answered = picks.filter(Boolean).length || 1;
    const sum = f => qz.reduce((a, d, i) => a + (f(d, picks[i]) || 0), 0);
    const yourMs = sum((d, g) => g && g.ms), cMs = sum(d => d.circuit_ms), lMs = sum(d => d.llm && d.llm.wall_ms);
    const you = qz.filter((d, i) => right(d, picks[i])).length;
    const circuit = qz.filter(d => d.pick === d.ref).length;
    const llm = qz.filter(d => d.llm && d.llm.answer === d.ref).length;
    $('qz-final').textContent = 'You ' + you + '/' + N + ', circuit models ' + circuit + '/' + N + ', chat model ' + llm + '/' + N + '.';
    const cell = (ans, ok, ms) => '<td><span class="a">' + esc(ans) + '</span> ' + mark(ok) + '<br><span class="t">' + fmt(ms) + '</span></td>';
    $('qz-board').innerHTML = '<tr><th></th><th>question</th><th>you</th><th>circuit model</th><th>chat model</th></tr>'
      + qz.map((d, i) => {
        const g = picks[i] || {};
        return '<tr><td class="n">' + (i + 1) + '</td><td>' + esc(d.label) + '</td>'
          + cell(g.k == null ? null : lab(d, g.k), right(d, g), g.ms)
          + cell(lab(d, d.pick), d.pick === d.ref, d.circuit_ms)
          + cell(d.llm && d.llm.answer != null ? lab(d, d.llm.answer) : null, !!(d.llm && d.llm.answer === d.ref), d.llm && d.llm.wall_ms) + '</tr>';
      }).join('');
    const row = (who, r, ms) => {
      const x = ratio(ms, cMs);
      return '<tr><td>' + who + '</td><td class="n">' + r + '/' + N + '</td><td class="n">' + fmt(ms / answered) + '</td><td class="n">' + fmt(ms) + '</td>'
        + '<td class="n">' + (x == null || Math.abs(x - 1) < 0.05 ? '—' : xs(x) + ' slower') + '</td></tr>';
    };
    $('qz-table').innerHTML = '<tr><th></th><th>right</th><th>average per question</th><th>total</th><th>against the circuit models</th></tr>'
      + row('you', you, yourMs) + row('circuit models, server time', circuit, cMs) + row('chat model, wall time', llm, lMs);
    const vsYou = ratio(yourMs, cMs), vsLlm = ratio(lMs, cMs);
    $('qz-speed').textContent = 'The circuit models answered all ' + N + ' in ' + fmt(cMs) + ' of server time'
      + (vsYou ? ' — ' + xs(vsYou) + ' faster than you' : '') + (vsLlm ? ' and ' + xs(vsLlm) + ' faster than the chat model' : '') + '.';
  }
  $('qz-start').addEventListener('click', () => showQ(0));
  $('qz-prev').addEventListener('click', () => showQ(Math.max(0, at - 1)));
  $('qz-next').addEventListener('click', () => { if (at === N - 1) results(); else showQ(at + 1); });
  $('qz-back').addEventListener('click', () => showQ(N - 1));
  $('qz-again').addEventListener('click', () => location.reload());
})();
"""


def main() -> None:
    items = json.loads(ITEMS.read_text())
    templates, data = [], []
    for i, it in enumerate(items):
        q = it["question"]
        ref = max(it["ref"], key=it["ref"].get)
        pick, p = model_pick(q, it["answer"])
        t = it.get("timing", {})
        crit = q.get("criteria") if q["type"] != "score" else {}
        data.append(
            {
                "label": it["label"],
                "ask": q["instructions"],
                "opts": [{"k": k, "l": lbl, "t": str((crit or {}).get(k) or "")} for k, lbl in options(q)],
                "ref": ref,
                "model": it["model"],
                "pick": pick,
                "p": round(p, 2),
                "circuit_ms": t.get("circuit", {}).get("server_ms"),
                "rtt_ms": t.get("circuit", {}).get("rtt_ms"),
                "llm": t.get("llm"),
            }
        )
        templates.append(f'      <template data-i="{i}">{state_html(it)}</template>')

    card = CARD.replace("__N__", str(len(items))).replace("__TEMPLATES__", "\n".join(templates))
    page = PAGE.read_text()
    page, n = re.subn(
        r"(    <!-- quiz:html:start -->\n).*?(    <!-- quiz:html:end -->)",
        lambda m: m.group(1) + card + m.group(2),
        page,
        count=1,
        flags=re.DOTALL,
    )
    assert n == 1, "quiz card markers not found"
    js = "\n".join(("  " + ln if ln.strip() else ln) for ln in QUIZ_JS.replace("__DATA__", json.dumps(data)).splitlines()) + "\n"
    page, n = re.subn(r"  // quiz:start\n.*?  // quiz:end\n", lambda _m: "  // quiz:start\n" + js + "  // quiz:end\n", page, count=1, flags=re.DOTALL)
    assert n == 1, "quiz script markers not found"
    PAGE.write_text(page)
    print(f"{len(items)} items; models right on {sum(d['pick'] == d['ref'] for d in data)}")


if __name__ == "__main__":
    main()
