# predict.py -- label the instructor's test set.
#
#   uv run src/predict.py data/tests/HW2-F26-testset.txt
#
# Input is one passage per line, as "ITEM-n<tab>text". Writes the labels to
# results/predictions.tsv and a fuller table with scores to results/predictions.csv.
#
# Unlike main.py, the models here are trained on ALL of both books rather than 90% of each.
# The held out tenth existed to measure accuracy; there is nothing left to measure now, so
# withholding it would only make the models worse at the job they are being asked to do.
#
# Assembly by Claude. The scoring and model code it calls is Yiyuan Jia's.

import csv
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
sys.path.insert(0, str(Path(__file__).parent))

from corpus import load_sentences
from tokenizer import train_tokenizer, encode, pad
from ngram import NgramModel
from authorid import score

HERE = Path(__file__).parent
RESULTS = HERE.parent / "results"
RESULTS.mkdir(exist_ok=True)

# Same settings the report selected.
VOCAB_SIZE = 4000
N = 2
K = 0.01

AUTHORS = ["tolkien", "doyle"]
LABEL = {"tolkien": "Tolkien", "doyle": "Doyle"}

if len(sys.argv) < 2:
    sys.exit("usage: uv run src/predict.py <test file>")
test_path = Path(sys.argv[1])


def read_test_set(path):
    """Read 'ITEM-n<tab>passage' lines. Returns a list of (id, text)."""
    items = []
    with open(path, encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.rstrip("\n")
            if not line.strip():
                continue
            if "\t" not in line:
                sys.exit(f"line {line_number} has no tab separator: {line[:60]!r}")
            item_id, text = line.split("\t", 1)
            items.append((item_id.strip(), text.strip()))
    return items


items = read_test_set(test_path)
print(f"read {len(items)} passages from {test_path.name}")
words = [len(t.split()) for _, t in items]
print(f"   {min(words)} to {max(words)} words, median {sorted(words)[len(words) // 2]}")

# ---------------------------------------------------------------- train on everything
sentences = {a: load_sentences(a) for a in AUTHORS}
tokenizer = train_tokenizer([" ".join(sentences[a]) for a in AUTHORS], vocab_size=VOCAB_SIZE)
V = tokenizer.get_vocab_size()

models = {}
for author in AUTHORS:
    model = NgramModel(N, V, k=K)
    model.fit([pad(encode(tokenizer, s), N) for s in sentences[author]])
    models[author] = model

print(f"\ntrained on all of both books, vocabulary {V}, n={N}, k={K}")
for author in AUTHORS:
    print(f"   {author:8s} {len(sentences[author]):5d} sentences")

# ---------------------------------------------------------------- label each passage
rows = []
for item_id, text in items:
    scores = {a: score(text, tokenizer, models[a], N) for a in AUTHORS}
    guess = min(AUTHORS, key=lambda a: scores[a])
    margin = abs(scores["tolkien"] - scores["doyle"])
    rows.append({
        "id": item_id,
        "prediction": LABEL[guess],
        "tolkien_score": round(scores["tolkien"], 4),
        "doyle_score": round(scores["doyle"], 4),
        "margin": round(margin, 4),
        "n_words": len(text.split()),
    })

# Sort by the numeric part of the id so the output reads in order.
rows.sort(key=lambda r: int(r["id"].split("-")[-1]))

# ---------------------------------------------------------------- write and summarise
labels_path = RESULTS / "predictions.tsv"
with open(labels_path, "w", encoding="utf-8", newline="") as f:
    for r in rows:
        f.write(f"{r['id']}\t{r['prediction']}\n")

detail_path = RESULTS / "predictions.csv"
with open(detail_path, "w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)

print(f"\nwrote {labels_path.relative_to(HERE.parent)} and {detail_path.relative_to(HERE.parent)}")

counts = {}
for r in rows:
    counts[r["prediction"]] = counts.get(r["prediction"], 0) + 1
print("\nlabels assigned:")
for label, n in sorted(counts.items()):
    print(f"   {label:8s} {n:3d}")

print("\nthe ten least confident calls, smallest margin first:")
print(f"   {'id':9s} {'guess':9s} {'margin':>7s} {'words':>6s}")
for r in sorted(rows, key=lambda r: r["margin"])[:10]:
    print(f"   {r['id']:9s} {r['prediction']:9s} {r['margin']:7.3f} {r['n_words']:6d}")
