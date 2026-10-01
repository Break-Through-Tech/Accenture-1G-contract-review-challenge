"""
Explore the CUAD dataset from the command line.

Companion to visualize_data.py: that one renders an aggregate HTML dashboard,
this one answers the ad-hoc questions - what does this clause type actually
look like, which contracts have it, where does this phrase appear.

Pure standard library. Reuses the loader and category parser from
visualize_data.py so both tools agree on what a "category" is.

Usage:
    python explore.py cats                          # all 41 categories + answer rates
    python explore.py cat "Governing Law" -n 15     # real annotated spans for a category
    python explore.py doc SUPPLY -v                 # one contract, all its annotations
    python explore.py grep "arbitrat\\w+"            # regex over annotated span text
    python explore.py q "Source Code Escrow"        # the full question text

Every command takes --file (default data/CUADv1.json).
"""

import argparse
import re
import sys
from collections import defaultdict

from visualize_data import category_of, load

# On Windows the default console codec chokes on contract text.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def walk(doc):
    """Yield (title, context, qa) for every question in the file."""
    for entry in doc["data"]:
        title = entry.get("title", "(untitled)")
        for para in entry["paragraphs"]:
            for qa in para["qas"]:
                yield title, para["context"], qa


def answered(qa):
    return bool(qa.get("answers")) and not qa.get("is_impossible", False)


def spans(qa):
    """Answer spans in the order they appear in the contract."""
    return sorted(qa["answers"], key=lambda a: a["answer_start"])


def squeeze(text, width=0):
    """Collapse the whitespace that contract text is full of."""
    out = " ".join(text.split())
    return out[:width] + "..." if width and len(out) > width else out


def resolve(doc, needle, kind):
    """Match a category or contract title case-insensitively, prefix or substring."""
    if kind == "category":
        names = sorted({category_of(qa) for _, _, qa in walk(doc)})
    else:
        names = sorted({e.get("title", "") for e in doc["data"]})

    low = needle.lower()
    hits = [n for n in names if n.lower() == low] \
        or [n for n in names if n.lower().startswith(low)] \
        or [n for n in names if low in n.lower()]

    if not hits:
        sys.exit("no %s matching %r" % (kind, needle))
    if len(hits) > 1:
        print("%d matches, using the first:" % len(hits))
        for h in hits[:8]:
            print("   ", squeeze(h, 90))
        print()
    return hits[0]


# ------------------------------------------------------------------ commands

def cmd_cats(doc, args):
    """One row per clause category, sorted by how often it is present."""
    rows = defaultdict(lambda: {"q": 0, "found": 0, "spans": 0})
    for _, _, qa in walk(doc):
        r = rows[category_of(qa)]
        r["q"] += 1
        if answered(qa):
            r["found"] += 1
            r["spans"] += len(qa["answers"])

    order = sorted(rows.items(), key=lambda kv: kv[1]["found"] / kv[1]["q"], reverse=True)
    print("%-44s %7s %8s %7s" % ("CATEGORY", "FOUND", "RATE", "SPANS"))
    for name, r in order:
        print("%-44s %7d %7.1f%% %7d"
              % (squeeze(name, 43), r["found"], 100.0 * r["found"] / r["q"], r["spans"]))
    print("\n%d categories, %d questions" % (len(rows), sum(r["q"] for r in rows.values())))


def cmd_cat(doc, args):
    """Real annotated spans for one category, so you can see what it looks like."""
    name = resolve(doc, args.name, "category")
    hits = [(t, qa) for t, _, qa in walk(doc) if category_of(qa) == name and answered(qa)]
    total = sum(1 for _, _, qa in walk(doc) if category_of(qa) == name)

    print("=== %s ===" % name)
    print("present in %d/%d contracts (%.1f%%), %d spans total\n"
          % (len(hits), total, 100.0 * len(hits) / max(1, total),
             sum(len(qa["answers"]) for _, qa in hits)))

    for title, qa in hits[:args.n]:
        print("-- %s" % squeeze(title, 88))
        for a in spans(qa):
            print("   [%6d] %s" % (a["answer_start"], squeeze(a["text"], args.width)))
        print()
    if len(hits) > args.n:
        print("(%d more, use -n)" % (len(hits) - args.n))


def cmd_doc(doc, args):
    """Everything annotated in a single contract."""
    title = resolve(doc, args.title, "contract")
    qas = [qa for t, _, qa in walk(doc) if t == title]
    context = next(c for t, c, _ in walk(doc) if t == title)
    found = [qa for qa in qas if answered(qa)]

    print("=== %s ===" % title)
    print("%s chars, %d questions, %d answered, %d spans\n"
          % (format(len(context), ","), len(qas), len(found),
             sum(len(qa["answers"]) for qa in found)))

    for qa in found:
        print("-- %s" % category_of(qa))
        for a in spans(qa):
            print("   [%6d] %s" % (a["answer_start"], squeeze(a["text"], args.width)))

    if args.verbose:
        missing = sorted({category_of(qa) for qa in qas} - {category_of(qa) for qa in found})
        print("\nabsent (%d): %s" % (len(missing), ", ".join(missing)))


def cmd_grep(doc, args):
    """Regex search over annotated span text - what language do these clauses use."""
    pat = re.compile(args.pattern, 0 if args.case else re.I)
    n, by_cat = 0, defaultdict(int)

    for title, _, qa in walk(doc):
        if not answered(qa):
            continue
        for a in qa["answers"]:
            flat = squeeze(a["text"])
            if pat.search(flat):
                by_cat[category_of(qa)] += 1
                n += 1
                if n <= args.n:
                    print("-- %s | %s" % (category_of(qa), squeeze(title, 60)))
                    print("   %s\n" % squeeze(flat, args.width))

    print("%d matching spans" % n)
    for cat, c in sorted(by_cat.items(), key=lambda kv: -kv[1])[:12]:
        print("   %-44s %d" % (squeeze(cat, 43), c))


def cmd_q(doc, args):
    """The exact prompt the model is asked for a category."""
    name = resolve(doc, args.name, "category")
    text = next(qa["question"] for _, _, qa in walk(doc) if category_of(qa) == name)
    print("=== %s ===\n%s" % (name, squeeze(text)))


# --------------------------------------------------------------------- main

def main():
    # Shared flags live on each subparser so they can follow the subcommand.
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--file", default="data/CUADv1.json")
    common.add_argument("-n", type=int, default=10, help="results to print")
    common.add_argument("-w", "--width", type=int, default=300,
                        help="truncate span text at N chars")

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("cats", parents=[common], help="all categories with answer rates")

    p = sub.add_parser("cat", parents=[common], help="annotated spans for one category")
    p.add_argument("name")

    p = sub.add_parser("doc", parents=[common], help="all annotations in one contract")
    p.add_argument("title")
    p.add_argument("-v", "--verbose", action="store_true", help="also list absent categories")

    p = sub.add_parser("grep", parents=[common], help="regex over annotated span text")
    p.add_argument("pattern")
    p.add_argument("-c", "--case", action="store_true", help="case sensitive")

    p = sub.add_parser("q", parents=[common], help="show the question text for a category")
    p.add_argument("name")

    args = ap.parse_args()
    doc = load(args.file)
    {"cats": cmd_cats, "cat": cmd_cat, "doc": cmd_doc,
     "grep": cmd_grep, "q": cmd_q}[args.cmd](doc, args)


if __name__ == "__main__":
    main()
