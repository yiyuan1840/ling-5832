# prepare_data.py -- build the cleaned corpora and print the diagnostics that justify the
# preprocessing choices. Run: python src/prepare_data.py
#
# Provenance: written by Claude (Anthropic) as scaffolding; reviewed by Yiyuan Jia.
#
# Writes one JSON file per (source, level) into data/clean/ and prints two things a human has
# to look at: the full log of every line removed as structure, and the typographic channel
# table that shows which differences between the books are the typesetter rather than the
# author.

import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
sys.path.insert(0, str(Path(__file__).parent))
import prep

HERE = Path(__file__).parent
DATA = HERE.parent / "data"
CLEAN = DATA / "clean"
CLEAN.mkdir(parents=True, exist_ok=True)

BOOKS = [("tolkien", DATA / "hobbit.txt"), ("doyle", DATA / "lostworld.txt")]

# ---------------------------------------------------------------- structural drop log
print("=" * 78)
print("STRUCTURAL REMOVAL LOG")
print("=" * 78)
for source, path in BOOKS:
    b = prep.load_book(path, source, "minimal")
    print(f"\n{source}: {b['n_boilerplate_lines']} boilerplate lines, "
          f"{b['n_dropped_lines']} structural lines, {b['n_hyphen_joins']} hyphen joins")
    for i, reason, text in b["dropped"][:6]:
        print(f"    line {i:6d}  [{reason}]  {text[:60]}")
    if b["n_dropped_lines"] > 6:
        print(f"    ... and {b['n_dropped_lines'] - 6} more")

# ---------------------------------------------------------------- channel table
print("\n" + "=" * 78)
print("TYPOGRAPHIC CHANNELS, rate per 1000 words")
print("=" * 78)
stats = {}
for level in prep.LEVELS:
    for source, path in BOOKS:
        b = prep.load_book(path, source, level)
        stats[(source, level)] = prep.channel_stats(" ".join(b["paragraphs"]))

for level in prep.LEVELS:
    print(f"\n--- level: {level}")
    print(f"    {'channel':30s} {'tolkien':>10s} {'doyle':>10s}   {'verdict':>10s}")
    for name, _ in prep.CHANNELS:
        a = stats[("tolkien", level)][name]["per_1k"]
        d = stats[("doyle", level)][name]["per_1k"]
        hi, lo = max(a, d), min(a, d)
        if hi < 0.01:
            v = "absent"
        elif lo < 0.01:
            v = "DISJOINT"
        elif hi / lo > 2.5:
            v = "skewed"
        else:
            v = "comparable"
        print(f"    {name:30s} {a:10.2f} {d:10.2f}   {v:>10s}")

# ---------------------------------------------------------------- corpus profile
print("\n" + "=" * 78)
print("CORPUS PROFILE (level: minimal)")
print("=" * 78)
print(f"    {'source':10s} {'words':>8s} {'paragraphs':>11s} {'sentences':>10s} {'w/sent':>8s}")
for source, path in BOOKS:
    b = prep.load_book(path, source, "minimal")
    ns = len(b["sentences"])
    print(f"    {source:10s} {b['n_words']:8d} {len(b['paragraphs']):11d} {ns:10d} "
          f"{b['n_words'] / ns:8.1f}")

# ---------------------------------------------------------------- write clean data
print("\n" + "=" * 78)
for level in prep.LEVELS:
    for source, path in BOOKS:
        b = prep.load_book(path, source, level)
        out = CLEAN / f"{source}.{level}.json"
        with open(out, "w", encoding="utf-8") as f:
            json.dump({"source": source, "level": level,
                       "paragraphs": b["paragraphs"],
                       "sentences": b["sentences"],
                       "para_of": b["para_of"]}, f, ensure_ascii=False)
        print(f"wrote {out.relative_to(HERE.parent)}  "
              f"({b['n_words']} words, {len(b['sentences'])} sentences)")

# ---------------------------------------------------------------- gates
print("\n" + "=" * 78)
print("GATES")
print("=" * 78)
LEAKY = ["curly double quote", "curly apostrophe", "em dash", "double hyphen",
         "double space after sentence"]
for name in LEAKY:
    a = stats[("tolkien", "minimal")][name]["per_1k"]
    d = stats[("doyle", "minimal")][name]["per_1k"]
    ok = max(a, d) < 0.01 or min(a, d) / max(a, d) > 0.3
    print(f"    [{'ok ' if ok else 'FAIL'}] {name:32s} tolkien={a:6.2f} doyle={d:6.2f}")
    assert ok, f"channel {name} still leaks at level minimal"

for source in ("tolkien", "doyle"):
    n = stats[(source, "minimal")]["intraword hyphen"]["count"]
    print(f"    [{'ok ' if n > 400 else 'FAIL'}] intraword hyphens survive in {source}: {n}")
    assert n > 400, f"{source} lost its intraword hyphens"

print("\nall gates passed")
