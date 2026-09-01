# Build 5 fixed test forms: A = the original seed-5832 sample; B-E drawn (seed 5833)
# from the remaining population, disjoint from A and from each other.
import json
import random
from pathlib import Path

HERE = Path(__file__).parent
ALLOC = {1: 60, 2: 180, 3: 45, 4: 55, 5: 20}
SEED_BE = 5833

population = json.load(open(HERE / "population.json", encoding="utf-8"))
sampleA = json.load(open(HERE / "sample.json", encoding="utf-8"))
formA = sampleA["sample"]
used = {e["id"] for e in formA}

strata = {k: [] for k in ALLOC}
for e in population:
    if e["i"] not in used:
        strata[min(len(e["head"]), 5)].append(e)

rng = random.Random(SEED_BE)
picks = {k: rng.sample(strata[k], 4 * ALLOC[k]) for k in sorted(ALLOC)}

forms = {"A": formA}
for i, name in enumerate(["B", "C", "D", "E"]):
    items = []
    for k in sorted(ALLOC):
        for e in picks[k][i * ALLOC[k]:(i + 1) * ALLOC[k]]:
            items.append({"id": e["i"], "line": e["line"], "head": e["head"],
                          "rest": e["rest"], "len": k})
    rng.shuffle(items)
    forms[name] = items

meta = dict(sampleA["meta"], site="ling-5832 vocab test v2",
            forms={"A": {"seed": 5832}, "B-E": {"seed": SEED_BE, "note": "disjoint from A and each other"}})

with open(HERE / "forms.json", "w", encoding="utf-8") as f:
    json.dump({"meta": meta, "forms": forms}, f, ensure_ascii=False, indent=1)

# checks: sizes, strata counts, pairwise disjointness
all_ids = []
for name, items in forms.items():
    counts = {}
    for it in items:
        counts[it["len"]] = counts.get(it["len"], 0) + 1
    assert len(items) == 360 and counts == ALLOC, (name, counts)
    all_ids.extend(it["id"] for it in items)
assert len(all_ids) == len(set(all_ids)) == 1800, "forms overlap!"
print("5 forms x 360 items, strata 60/180/45/55/20 each, all 1800 ids unique")
print("form B first 5:", [it["head"] for it in forms["B"][:5]])
