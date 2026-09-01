# Pull concrete examples from the real results for the report narrative.
import json
from pathlib import Path

here = Path(__file__).parent
rows = json.load(open(here / "vocab_results.json", encoding="utf-8"))["results"]
sample = {s["id"]: s for s in json.load(open(here / "sample.json", encoding="utf-8"))["sample"]}

def show(label, pred, cap=15):
    hits = [r for r in rows if pred(r)]
    print(f"{label} ({len(hits)}):")
    for r in hits[:cap]:
        rest = sample[r["id"]]["rest"]
        print(f"  {r['head']} [{r['len']}] | {rest[:70]}")
    print()

show("rated 不认识", lambda r: r["rating"] == 0)
show("rated 眼熟", lambda r: r["rating"] == 1)
show("claimed but 错了", lambda r: r["rating"] >= 2 and r["verify"] == 0)
show("claimed but 部分对", lambda r: r["rating"] >= 2 and r["verify"] == 1)
