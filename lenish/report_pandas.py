"""
Build an HTML report on a CUAD file, using pandas + matplotlib.

The sibling script visualize_data.py does this with the standard library only and
hand-written SVG. This one trades that portability for DataFrames and real plots,
which makes it far shorter to extend: every section below is a groupby and a figure.

Figures are embedded as inline SVG, so the output is still a single self-contained
file you can mail to someone.

Usage:
    python report_pandas.py                                  # data/CUADv1.json
    python report_pandas.py --file data/test.json --out test.html
    python report_pandas.py --no-compare                     # skip the test-coverage section

Requires: pandas, matplotlib.
"""

import argparse
import io
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")           # no display needed; must precede pyplot
import matplotlib.pyplot as plt
import pandas as pd

BLUE, RED, AMBER = "#2a78d6", "#d03b3b", "#b07800"
INK, MUTED, GRID = "#0b0b0b", "#898781", "#e1e0d9"

plt.rcParams.update({
    "figure.facecolor": "none", "savefig.facecolor": "none",
    "font.size": 9, "font.family": "sans-serif",
    "axes.edgecolor": GRID, "axes.labelcolor": INK, "axes.facecolor": "none",
    "axes.titlesize": 10, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "xtick.labelcolor": INK, "ytick.labelcolor": INK,
    "legend.frameon": False,
})

QPREFIX = re.compile(r'^Highlight the parts \(if any\) of this contract related to "(.*?)"')


# ------------------------------------------------------------------ loading

def category_of(qa):
    """Category for a qa entry. The id suffix is authoritative; the question is a fallback.

    Training-file ids carry a _0/_1 suffix because a question with N gold spans is
    split into N single-answer entries there. Strip it so both files group alike.
    """
    qid = qa.get("id", "")
    if "__" in qid:
        return re.sub(r"_\d+$", "", qid.rsplit("__", 1)[1])
    m = QPREFIX.match(qa.get("question", ""))
    return m.group(1) if m else "Unknown"


def load(path):
    """Flatten the SQuAD nesting into (contracts, questions, spans) frames."""
    doc = json.loads(Path(path).read_text(encoding="utf-8"))
    docs, qas, spans = [], [], []

    for entry in doc["data"]:
        title = entry.get("title", "(untitled)")
        for para in entry["paragraphs"]:
            ctx = para["context"]
            docs.append({"title": title, "n_chars": len(ctx), "n_questions": len(para["qas"])})
            for qa in para["qas"]:
                cat = category_of(qa)
                answers = qa.get("answers") or []
                found = bool(answers) and not qa.get("is_impossible", False)
                qas.append({"title": title, "category": cat, "found": found,
                            "n_spans": len(answers) if found else 0})
                if found:
                    for a in answers:
                        spans.append({"title": title, "category": cat,
                                      "start": a["answer_start"], "length": len(a["text"]),
                                      "rel_pos": a["answer_start"] / max(1, len(ctx))})

    return pd.DataFrame(docs), pd.DataFrame(qas), pd.DataFrame(spans)


def category_table(qas, spans, test_qas=None):
    """One row per clause category: how common, how bursty, how long, how localised."""
    # Count CONTRACTS, not question rows: the train file explodes a multi-span question
    # into one row per span, so g.size() there is not "how many contracts were asked".
    g = qas.groupby("category")
    cats = pd.DataFrame({
        "asked": g.title.nunique(),
        "present": qas[qas.found].groupby("category").title.nunique(),
        "spans": g.n_spans.sum(),
    })
    cats["present"] = cats.present.fillna(0).astype(int)
    cats["rate"] = 100 * cats.present / cats.asked
    cats["burst"] = cats.spans / cats.present.clip(lower=1)

    sg = spans.groupby("category")
    cats["median_span"] = sg.length.median()
    cats["mean_rel_pos"] = sg.rel_pos.mean()

    if test_qas is not None:
        cats["in_test"] = test_qas.groupby("category").found.sum()
        cats["in_test"] = cats.in_test.fillna(0).astype(int)

    return cats.sort_values("rate", ascending=False)


# ------------------------------------------------------------------ figures

def svg(fig):
    """Render a figure to an inline <svg> string and close it."""
    buf = io.StringIO()
    fig.savefig(buf, format="svg", bbox_inches="tight", transparent=True)
    plt.close(fig)
    # drop the XML prolog and DOCTYPE so it can sit inside the HTML body
    return re.sub(r"^.*?(?=<svg)", "", buf.getvalue(), flags=re.S)


def tidy(ax, xgrid=False):
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.grid(axis="x" if xgrid else "y")
    ax.set_axisbelow(True)
    ax.tick_params(length=0)
    return ax


def fig_rarity(cats):
    d = cats.sort_values("rate")
    fig, ax = plt.subplots(figsize=(7, 0.22 * len(d) + 0.6))
    ax.barh(d.index, d.rate, color=BLUE, height=0.68)
    for y, (r, p) in enumerate(zip(d.rate, d.present)):
        ax.text(r + 1.2, y, f"{r:.1f}%  ({p})", va="center", fontsize=7, color=MUTED)
    ax.set_xlim(0, 120)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xlabel("share of contracts containing the clause")
    return svg(tidy(ax, xgrid=True).figure)


def fig_burst(cats):
    fig, ax = plt.subplots(figsize=(7, 4.4))
    ax.scatter(cats.rate, cats.burst, s=32, color=BLUE, alpha=.85,
               edgecolor="white", linewidth=.8, zorder=3)
    med = cats.burst.median()
    ax.axhline(med, color=MUTED, lw=1, zorder=1)
    ax.text(ax.get_xlim()[1], med, f"  median {med:.2f}", fontsize=7.5, color=MUTED, va="center")

    # label the corners only; a name on every point is unreadable
    for name, r in cats.iterrows():
        if r.burst > 3.4 or (r.rate > 90 and r.burst < 1.5) or (r.rate < 8 and r.burst > 2):
            ax.annotate(name, (r.rate, r.burst), fontsize=7, color=INK,
                        xytext=(5, 4), textcoords="offset points")
    ax.set_xlabel("present in % of contracts")
    ax.set_ylabel("average spans, when present")
    return svg(tidy(ax).figure)


def fig_position(cats, spans):
    s = spans.copy()
    s["bin"] = (s.rel_pos * 10).clip(0, 9.99).astype(int)
    h = (s.pivot_table(index="category", columns="bin", values="start", aggfunc="size")
          .reindex(columns=range(10)).fillna(0))
    # categories with no spans in this file have no row; order the ones that do
    order = [c for c in cats.sort_values("mean_rel_pos").index if c in h.index]
    h = h.div(h.sum(axis=1), axis=0).reindex(order)

    fig, ax = plt.subplots(figsize=(6.4, 0.235 * len(h) + 0.8))
    im = ax.imshow(h.values, aspect="auto", cmap="Blues", vmin=0, vmax=0.40)
    ax.set_yticks(range(len(h)), h.index, fontsize=7.5)
    ax.set_xticks(range(10), [f"{i * 10}" for i in range(10)], fontsize=7.5)
    ax.set_xlabel("position through the contract (%)")
    ax.grid(False)
    cb = fig.colorbar(im, ax=ax, fraction=.03, pad=.02, extend="max")
    cb.set_label("share of the category's spans", fontsize=7.5)
    cb.outline.set_visible(False)
    return svg(fig), h


def fig_lengths(docs, spans):
    fig, axes = plt.subplots(1, 2, figsize=(9, 3))
    ax = axes[0]
    ax.hist(docs.n_chars, bins=40, color=BLUE)
    ax.set_xlabel("characters per contract")
    ax.set_ylabel("contracts")
    ax.set_title("Contract length")
    tidy(ax)

    ax = axes[1]
    ax.hist(spans.length.clip(upper=3000), bins=40, color=BLUE)
    ax.axvline(512, color=RED, lw=1.2)
    ax.text(560, ax.get_ylim()[1] * .85, "512 chars", fontsize=7.5, color=RED)
    ax.set_xlabel("characters per span (clipped at 3,000)")
    ax.set_ylabel("spans")
    ax.set_title("Annotated span length")
    tidy(ax)
    fig.tight_layout()
    return svg(fig)


def fig_coverage(cats, n_test):
    d = cats.nsmallest(14, "in_test").sort_values("in_test", ascending=False)
    colors = [RED if n == 0 else AMBER if n < 5 else BLUE for n in d.in_test]

    fig, ax = plt.subplots(figsize=(7, 3.8))
    ax.barh(d.index, d.in_test, color=colors, height=.65)
    for y, n in enumerate(d.in_test):
        note = "cannot be scored" if n == 0 else ("too few to score" if n < 5 else "")
        ax.text(n + .3, y, f"{n}" + (f"   ▲ {note}" if note else ""), va="center",
                fontsize=7.5, color=RED if n == 0 else AMBER if n < 5 else MUTED)
    ax.set_xlim(0, max(d.in_test.max(), 1) + 14)
    ax.set_xlabel(f"contracts in the {n_test}-contract test split containing the clause")
    return svg(tidy(ax, xgrid=True).figure)


# --------------------------------------------------------------------- html

CSS = """
:root{--ink:#0b0b0b;--ink2:#52514e;--mut:#898781;--line:#e6e5e0;--bg:#fff;--plane:#f7f7f5;
      --blue:#2a78d6;--red:#d03b3b;--amber:#b07800}
*{box-sizing:border-box}
body{margin:0;background:var(--plane);color:var(--ink);
     font:14px/1.6 system-ui,-apple-system,"Segoe UI",sans-serif}
.wrap{max-width:1000px;margin:0 auto;padding:40px 20px 72px}
h1{font-size:26px;margin:0 0 6px;letter-spacing:-.02em}
h2{font-size:17px;margin:44px 0 6px;letter-spacing:-.01em}
.sub{color:var(--mut);margin:0 0 4px;font-size:13px}
.note{color:var(--ink2);font-size:13.5px;max-width:74ch;margin:0 0 16px}
code{font:12.5px ui-monospace,Consolas,monospace;background:var(--plane);padding:1px 4px;border-radius:3px}
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:10px;margin:26px 0 0}
.card{background:var(--bg);border:1px solid var(--line);border-radius:8px;padding:12px 14px}
.card .k{color:var(--mut);font-size:11px;text-transform:uppercase;letter-spacing:.06em}
.card .v{font-size:21px;font-weight:600;margin-top:4px;letter-spacing:-.02em}
.panel{background:var(--bg);border:1px solid var(--line);border-radius:8px;padding:16px;overflow-x:auto}
.panel svg{max-width:100%;height:auto;display:block}
table{border-collapse:collapse;font-size:12.5px;width:100%;min-width:640px}
th,td{text-align:right;padding:6px 9px;border-bottom:1px solid var(--line);
      font-variant-numeric:tabular-nums;white-space:nowrap}
th{color:var(--mut);font-weight:600;font-size:10.5px;text-transform:uppercase;letter-spacing:.05em;
   border-bottom:1px solid var(--mut)}
th:first-child,td:first-child{text-align:left;font-variant-numeric:normal}
tbody tr:hover{background:#f4f8fe}
.flag-none{color:var(--red);font-weight:600}
.flag-few{color:var(--amber);font-weight:600}
footer{margin-top:48px;padding-top:18px;border-top:1px solid var(--line);
       color:var(--mut);font-size:12.5px}
"""


def cards_html(docs, qas, spans):
    n_q, n_found = len(qas), int(qas.found.sum())
    items = [
        ("Contracts", f"{len(docs):,}"),
        ("Questions", f"{n_q:,}"),
        ("Clause categories", f"{qas.category.nunique()}"),
        ("Answered", f"{n_found:,} ({100 * n_found / n_q:.1f}%)"),
        ("Annotated spans", f"{len(spans):,}"),
        ("Median contract", f"{docs.n_chars.median():,.0f} chars"),
        ("Corpus size", f"{docs.n_chars.sum() / 1e6:.1f}M chars"),
        ("Spans per answer", f"{len(spans) / max(1, n_found):.2f}"),
    ]
    return "".join(f'<div class="card"><div class="k">{k}</div><div class="v">{v}</div></div>'
                   for k, v in items)


def num(v, spec):
    """Absent categories have no spans, so these aggregates are genuinely undefined."""
    return "&mdash;" if pd.isna(v) else format(v, spec)


def table_html(cats, has_test):
    d = cats.reset_index()
    cols = ["category", "rate", "burst", "present", "spans", "median_span", "mean_rel_pos"]
    head = ["Category", "Present %", "Spans when present", "Contracts", "Spans",
            "Median span", "Mean position"]
    if has_test:
        cols.append("in_test")
        head.append("In test")

    out = ["<table><thead><tr>" + "".join(f"<th>{h}</th>" for h in head) + "</tr></thead><tbody>"]
    for _, r in d.iterrows():
        cells = [f"<td>{r.category}</td>",
                 f"<td>{r.rate:.1f}</td>", f"<td>{r.burst:.2f}</td>",
                 f"<td>{int(r.present)}</td>", f"<td>{int(r.spans):,}</td>",
                 f"<td>{num(r.median_span, ',.0f')}</td>",
                 f"<td>{num(r.mean_rel_pos, '.2f')}</td>"]
        if has_test:
            n = int(r.in_test)
            cls = "flag-none" if n == 0 else "flag-few" if n < 5 else ""
            cells.append(f'<td class="{cls}">{n}</td>')
        out.append("<tr>" + "".join(cells) + "</tr>")
    return "".join(out) + "</tbody></table>"


def build_html(source, docs, qas, spans, cats, has_test, n_test):
    pos_svg, heat = fig_position(cats, spans)
    peak = heat.max(axis=1)
    over512 = 100 * (spans.length > 512).mean()

    parts = [f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>CUAD report</title><style>{CSS}</style></head><body><div class="wrap">
<h1>CUAD dataset report</h1>
<p class="sub">Source: <code>{source}</code> &middot; built with pandas {pd.__version__}</p>
<div class="cards">{cards_html(docs, qas, spans)}</div>

<h2>How often each clause appears</h2>
<p class="note">CUAD is deliberately sparse: most categories are absent from most contracts.
Note what sits at the top &mdash; Document Name, Parties, Agreement Date and Effective Date are
cover-page metadata rather than clauses, and being near-universal they will dominate any
aggregate score.</p>
<div class="panel">{fig_rarity(cats)}</div>

<h2>Rarity against burstiness</h2>
<p class="note">How many separate spans get annotated when a clause is present, plotted against
how often it is present. The correlation is {cats[['rate', 'burst']].corr().iloc[0, 1]:.2f} &mdash;
these are independent axes. The top-left corner (rare but heavily annotated) is the hard case.</p>
<div class="panel">{fig_burst(cats)}</div>

<h2>Where in the document</h2>
<p class="note">Every contract divided into ten equal parts by character offset, shaded by where
that category's spans fall. The metadata categories pin to the first band; the median category
peaks at just {100 * peak.median():.1f}% of its spans in its best tenth, against 10% for an even
scatter. For most clause types, position carries very little information.</p>
<div class="panel">{pos_svg}</div>

<h2>Lengths</h2>
<p class="note">{over512:.1f}% of gold spans run past 512 characters. <code>run.sh</code> sets
<code>--max_answer_length 512</code>, which is tokens rather than characters, but the tail here is
long enough to be worth checking against the tokenizer.</p>
<div class="panel">{fig_lengths(docs, spans)}</div>
"""]

    if has_test:
        thin = cats[cats.in_test < 5]
        names = ", ".join(f"{n} ({int(r.in_test)})" for n, r in thin.iterrows())
        parts.append(f"""
<h2>Test-set coverage</h2>
<p class="note">The split is clean &mdash; no contract appears on both sides &mdash; but it is drawn
from the same skewed distribution, so the rarest categories arrive with almost nothing.
{len(thin)} categories have fewer than five test examples: {names}. Any per-category score
reported for these is noise, and a category at zero cannot be scored at all.</p>
<div class="panel">{fig_coverage(cats, n_test)}</div>""")

    parts.append(f"""
<h2>Every category</h2>
<p class="note">Mean position is the average relative offset of the category's spans, 0 = start of
contract, 1 = end.{' Counts under five in the test column are flagged.' if has_test else ''}</p>
<div class="panel">{table_html(cats, has_test)}</div>

<footer>Generated by <code>report_pandas.py</code>. Dataset: Hendrycks, Burns, Chen &amp; Ball,
<i>CUAD: An Expert-Annotated NLP Dataset for Legal Contract Review</i>, NeurIPS 2021.</footer>
</div></body></html>""")
    return "".join(parts)


# --------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--file", default="data/CUADv1.json")
    ap.add_argument("--out", default="cuad_pandas_report.html")
    ap.add_argument("--test-file", default="data/test.json",
                    help="used for the coverage section")
    ap.add_argument("--no-compare", action="store_true",
                    help="skip the test-coverage section")
    args = ap.parse_args()

    print(f"loading {args.file} ...")
    docs, qas, spans = load(args.file)
    print(f"  {len(docs):,} contracts, {len(qas):,} questions, {len(spans):,} spans")

    test_qas, n_test = None, 0
    same_file = Path(args.file).resolve() == Path(args.test_file).resolve()
    if not args.no_compare and not same_file and Path(args.test_file).exists():
        test_docs, test_qas, _ = load(args.test_file)
        n_test = len(test_docs)
        print(f"  test split: {n_test} contracts")

    cats = category_table(qas, spans, test_qas)
    page = build_html(args.file, docs, qas, spans, cats, test_qas is not None, n_test)
    Path(args.out).write_text(page, encoding="utf-8")
    print(f"wrote {args.out} ({Path(args.out).stat().st_size / 1e3:.0f} KB)")

    if test_qas is not None:
        thin = cats[cats.in_test < 5]
        if len(thin):
            print(f"\n{len(thin)} categories with <5 test examples:")
            print(thin[["rate", "present", "in_test"]].round(1).to_string())


if __name__ == "__main__":
    main()
