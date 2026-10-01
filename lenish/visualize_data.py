"""
Visualize the CUAD dataset (data/CUADv1.json, data/test.json, ...).

Pure standard library - no numpy/pandas/matplotlib needed. Reads the
SQuAD-style JSON and writes a single self-contained HTML dashboard with
inline SVG charts plus a clause highlighter for sample contracts.

Usage:
    python visualize_data.py                          # uses data/CUADv1.json
    python visualize_data.py --file data/test.json --out test_report.html
    python visualize_data.py --samples 15
"""

import argparse
import html
import json
import os
import re
from collections import Counter, defaultdict

QUESTION_PREFIX = re.compile(r'^Highlight the parts \(if any\) of this contract related to "(.*?)"')


def category_of(qa):
    """Category name for a qa entry: prefer the id suffix, fall back to the question."""
    qid = qa.get("id", "")
    if "__" in qid:
        return qid.rsplit("__", 1)[1]
    m = QUESTION_PREFIX.match(qa.get("question", ""))
    return m.group(1) if m else "Unknown"


def load(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def analyze(doc):
    """Walk the SQuAD structure once and collect everything the report needs."""
    stats = {
        "contracts": 0,
        "questions": 0,
        "answered": 0,
        "answer_spans": 0,
        "categories": defaultdict(lambda: {"q": 0, "answered": 0, "spans": 0, "lens": []}),
        "doc_chars": [],            # contract length in characters
        "spans_per_answered": Counter(),
        "titles": [],
    }

    for entry in doc["data"]:
        stats["contracts"] += 1
        title = entry.get("title", "(untitled)")
        for para in entry["paragraphs"]:
            context = para["context"]
            stats["doc_chars"].append(len(context))
            stats["titles"].append((title, len(context), len(para["qas"])))
            for qa in para["qas"]:
                cat = category_of(qa)
                c = stats["categories"][cat]
                stats["questions"] += 1
                c["q"] += 1
                answers = qa.get("answers") or []
                if answers and not qa.get("is_impossible", False):
                    stats["answered"] += 1
                    c["answered"] += 1
                    c["spans"] += len(answers)
                    stats["answer_spans"] += len(answers)
                    stats["spans_per_answered"][min(len(answers), 10)] += 1
                    for a in answers:
                        c["lens"].append(len(a["text"]))
    return stats


def samples(doc, n, max_spans=400):
    """Grab the first n contracts with their answer spans, for the highlighter."""
    out = []
    for entry in doc["data"][:n]:
        para = entry["paragraphs"][0]
        spans = []
        for qa in para["qas"]:
            for a in (qa.get("answers") or []):
                start = a["answer_start"]
                spans.append({"s": start, "e": start + len(a["text"]), "c": category_of(qa)})
        spans.sort(key=lambda s: s["s"])
        out.append({
            "title": entry.get("title", "(untitled)"),
            "context": para["context"],
            "spans": spans[:max_spans],
        })
    return out


# ---------------------------------------------------------------- SVG charts

def bar_chart(rows, width=900, row_h=22, pad_left=330, color="#4f7cff"):
    """Horizontal bars. rows = [(label, value, annotation), ...]"""
    if not rows:
        return "<p>no data</p>"
    vmax = max(r[1] for r in rows) or 1
    plot_w = width - pad_left - 90
    height = row_h * len(rows) + 10
    parts = ['<svg viewBox="0 0 %d %d" width="100%%" role="img">' % (width, height)]
    for i, (label, value, note) in enumerate(rows):
        y = i * row_h + 4
        w = max(1.0, plot_w * value / vmax)
        parts.append(
            '<text x="%d" y="%d" text-anchor="end" class="lbl">%s</text>'
            '<rect x="%d" y="%d" width="%.1f" height="%d" rx="3" fill="%s"/>'
            '<text x="%.1f" y="%d" class="val">%s</text>'
            % (pad_left - 8, y + 12, html.escape(str(label)),
               pad_left, y, w, row_h - 7, color,
               pad_left + w + 6, y + 12, html.escape(note))
        )
    parts.append("</svg>")
    return "".join(parts)


def histogram(values, bins=30, width=900, height=260, color="#22a06b", xlabel=""):
    if not values:
        return "<p>no data</p>"
    lo, hi = min(values), max(values)
    if hi == lo:
        hi = lo + 1
    step = (hi - lo) / bins
    counts = [0] * bins
    for v in values:
        counts[min(bins - 1, int((v - lo) / step))] += 1
    cmax = max(counts) or 1
    pad_l, pad_b, pad_t = 55, 34, 10
    plot_w, plot_h = width - pad_l - 15, height - pad_b - pad_t
    bw = plot_w / bins
    parts = ['<svg viewBox="0 0 %d %d" width="100%%" role="img">' % (width, height)]
    for f in (0, 0.5, 1.0):
        y = pad_t + plot_h * (1 - f)
        parts.append('<line x1="%d" y1="%.1f" x2="%d" y2="%.1f" class="grid"/>'
                     '<text x="%d" y="%.1f" text-anchor="end" class="val">%d</text>'
                     % (pad_l, y, width - 15, y, pad_l - 8, y + 4, int(cmax * f)))
    for i, c in enumerate(counts):
        h = plot_h * c / cmax
        x = pad_l + i * bw
        parts.append('<rect x="%.1f" y="%.1f" width="%.1f" height="%.1f" fill="%s">'
                     '<title>%d contracts in [%.0f, %.0f]</title></rect>'
                     % (x, pad_t + plot_h - h, max(1.0, bw - 1.5), h, color,
                        c, lo + i * step, lo + (i + 1) * step))
    for f in (0, 0.25, 0.5, 0.75, 1.0):
        parts.append('<text x="%.0f" y="%d" text-anchor="middle" class="val">%s</text>'
                     % (pad_l + plot_w * f, height - 12, format(int(lo + (hi - lo) * f), ",")))
    parts.append('<text x="%.0f" y="%d" text-anchor="middle" class="lbl">%s</text>'
                 % (pad_l + plot_w / 2, height - 1, html.escape(xlabel)))
    parts.append("</svg>")
    return "".join(parts)


def pct(a, b):
    return 100.0 * a / b if b else 0.0


def median(xs):
    if not xs:
        return 0
    s = sorted(xs)
    m = len(s) // 2
    return s[m] if len(s) % 2 else (s[m - 1] + s[m]) / 2.0


def build_html(stats, sample_docs, source):
    cats = stats["categories"]

    by_rate = sorted(cats.items(), key=lambda kv: pct(kv[1]["answered"], kv[1]["q"]), reverse=True)
    rate_rows = [(k, pct(v["answered"], v["q"]),
                  "%.1f%%  (%d/%d)" % (pct(v["answered"], v["q"]), v["answered"], v["q"]))
                 for k, v in by_rate]

    by_len = sorted(((k, median(v["lens"])) for k, v in cats.items() if v["lens"]),
                    key=lambda kv: kv[1], reverse=True)
    len_rows = [(k, m, "%s chars" % format(int(m), ",")) for k, m in by_len]

    spans = stats["spans_per_answered"]
    span_rows = [("10+" if n == 10 else str(n), spans[n], format(spans[n], ",")) for n in sorted(spans)]

    longest = sorted(stats["titles"], key=lambda t: t[1], reverse=True)[:15]
    long_rows = [(t[:58], n, "%s chars" % format(n, ",")) for t, n, _ in longest]

    cards = [
        ("Contracts", format(stats["contracts"], ",")),
        ("Questions (contract x category)", format(stats["questions"], ",")),
        ("Clause categories", format(len(cats), ",")),
        ("Answered questions", "%s (%.1f%%)" % (format(stats["answered"], ","),
                                                pct(stats["answered"], stats["questions"]))),
        ("Annotated spans", format(stats["answer_spans"], ",")),
        ("Median contract length", "%s chars" % format(int(median(stats["doc_chars"])), ",")),
        ("Total corpus size", "%.1fM chars" % (sum(stats["doc_chars"]) / 1e6)),
        ("Avg spans / answered question",
         "%.2f" % (stats["answer_spans"] / max(1, stats["answered"]))),
    ]
    card_html = "".join('<div class="card"><div class="k">%s</div><div class="v">%s</div></div>'
                        % (html.escape(k), html.escape(v)) for k, v in cards)

    payload = json.dumps(sample_docs).replace("</", "<\\/")

    return """<!doctype html>
<meta charset="utf-8">
<title>CUAD dataset overview</title>
<style>
 :root { color-scheme: light; --fg:#1b1f24; --mut:#5c6773; --line:#e3e7ec; --bg:#fff; }
 body { margin:0; padding:32px 28px 60px; font:14px/1.5 -apple-system,Segoe UI,Roboto,sans-serif;
        color:var(--fg); background:#f7f8fa; }
 h1 { font-size:24px; margin:0 0 4px; } h2 { font-size:17px; margin:34px 0 10px; }
 .sub { color:var(--mut); margin-bottom:22px; }
 section, .cards { max-width:1000px; }
 .panel { background:var(--bg); border:1px solid var(--line); border-radius:10px; padding:16px 18px; }
 .cards { display:grid; grid-template-columns:repeat(auto-fit,minmax(210px,1fr)); gap:12px; }
 .card { background:var(--bg); border:1px solid var(--line); border-radius:10px; padding:12px 14px; }
 .card .k { color:var(--mut); font-size:12px; text-transform:uppercase; letter-spacing:.04em; }
 .card .v { font-size:20px; font-weight:600; margin-top:4px; }
 .lbl { font-size:11px; fill:var(--fg); } .val { font-size:11px; fill:var(--mut); }
 .grid { stroke:var(--line); stroke-width:1; }
 select { font:inherit; padding:5px 8px; border:1px solid var(--line); border-radius:6px; max-width:100%; }
 #ctx { white-space:pre-wrap; font:12px/1.7 ui-monospace,Consolas,monospace; max-height:520px;
        overflow:auto; border:1px solid var(--line); border-radius:8px; padding:14px; margin-top:12px; }
 mark { background:#ffe9a8; border-bottom:2px solid #e0a200; padding:1px 0; }
 .legend { color:var(--mut); font-size:12px; margin-top:8px; }
</style>
<h1>CUAD dataset overview</h1>
<div class="sub">Source: <code>__SOURCE__</code></div>

<div class="cards">__CARDS__</div>

<h2>Answer rate by clause category</h2>
<section class="panel">__RATE__
 <div class="legend">Share of contracts where this clause type was actually found. CUAD is deliberately
 sparse &mdash; most categories are absent from most contracts, which is what makes it a
 "needle in a haystack" task.</div>
</section>

<h2>Contract length distribution</h2>
<section class="panel">__HIST__</section>

<h2>Median answer span length by category</h2>
<section class="panel">__LENS__</section>

<h2>Annotated spans per answered question</h2>
<section class="panel">__SPANS__</section>

<h2>Longest contracts</h2>
<section class="panel">__LONG__</section>

<h2>Clause highlighter</h2>
<section class="panel">
 <select id="pick"></select>
 <div class="legend">Highlighted text = expert-annotated clause spans. Hover a highlight for its category.</div>
 <div id="ctx"></div>
</section>

<script>
const DOCS = __PAYLOAD__;
const pick = document.getElementById('pick'), ctx = document.getElementById('ctx');
DOCS.forEach((d, i) => pick.add(new Option(d.title, i)));
function esc(s) { return s.replace(/[&<>]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;'}[c])); }
function render() {
  const d = DOCS[pick.value | 0];
  let out = '', cur = 0;
  for (const sp of d.spans) {
    if (sp.s < cur) continue;                        // skip overlapping spans
    out += esc(d.context.slice(cur, sp.s));
    out += '<mark title="' + esc(sp.c) + '">' + esc(d.context.slice(sp.s, sp.e)) + '</mark>';
    cur = sp.e;
  }
  out += esc(d.context.slice(cur));
  ctx.innerHTML = out;
  ctx.scrollTop = 0;
}
pick.onchange = render;
render();
</script>
""" \
        .replace("__SOURCE__", html.escape(source)) \
        .replace("__CARDS__", card_html) \
        .replace("__RATE__", bar_chart(rate_rows)) \
        .replace("__HIST__", histogram(stats["doc_chars"], xlabel="characters per contract")) \
        .replace("__LENS__", bar_chart(len_rows, color="#8a5cf6")) \
        .replace("__SPANS__", bar_chart(span_rows, pad_left=90, color="#e2725b")) \
        .replace("__LONG__", bar_chart(long_rows, pad_left=430, color="#2f7d92")) \
        .replace("__PAYLOAD__", payload)


def main():
    ap = argparse.ArgumentParser(description="Visualize the CUAD dataset as a standalone HTML report.")
    ap.add_argument("--file", default=os.path.join("data", "CUADv1.json"))
    ap.add_argument("--out", default="cuad_report.html")
    ap.add_argument("--samples", type=int, default=8,
                    help="contracts to embed in the clause highlighter")
    args = ap.parse_args()

    print("loading %s ..." % args.file)
    doc = load(args.file)
    stats = analyze(doc)
    print("  %s contracts, %s questions, %s spans, %d categories"
          % (format(stats["contracts"], ","), format(stats["questions"], ","),
             format(stats["answer_spans"], ","), len(stats["categories"])))

    page = build_html(stats, samples(doc, args.samples), args.file)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(page)
    print("wrote %s (%.1f MB) - open it in a browser"
          % (args.out, os.path.getsize(args.out) / 1e6))


if __name__ == "__main__":
    main()
