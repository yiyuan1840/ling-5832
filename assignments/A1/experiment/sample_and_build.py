# Draw the stratified random sample (seed=5832) and build the self-test app vocab_test.html.
import json
import random
from pathlib import Path

HERE = Path(__file__).parent
SEED = 5832
ALLOC = {1: 60, 2: 180, 3: 45, 4: 55, 5: 20}  # 5 == "5 or more chars"

with open(HERE / "population.json", encoding="utf-8") as f:
    population = json.load(f)

strata = {k: [] for k in ALLOC}
for e in population:
    strata[min(len(e["head"]), 5)].append(e)

rng = random.Random(SEED)
sample = []
for k in sorted(ALLOC):
    picks = rng.sample(strata[k], ALLOC[k])
    for e in picks:
        sample.append({"id": e["i"], "line": e["line"], "head": e["head"],
                       "rest": e["rest"], "len": k})
rng.shuffle(sample)  # presentation order mixes strata

meta = {
    "dictionary": "《现代汉语词典》（第7版）full-text TXT (github.com/CNMan/XDHYCD7th)",
    "population_total": len(population),
    "strata_N": {str(k): len(strata[k]) for k in sorted(ALLOC)},
    "strata_n": {str(k): ALLOC[k] for k in sorted(ALLOC)},
    "seed": SEED,
}

with open(HERE / "sample.json", "w", encoding="utf-8") as f:
    json.dump({"meta": meta, "sample": sample}, f, ensure_ascii=False, indent=1)

template = (HERE / "vocab_test_template.html").read_text(encoding="utf-8")
html = template.replace("__DATA__", json.dumps(sample, ensure_ascii=False)) \
               .replace("__META__", json.dumps(meta, ensure_ascii=False))
(HERE / "vocab_test.html").write_text(html, encoding="utf-8")

print(f"population: {len(population)}; sample: {len(sample)}")
for k in sorted(ALLOC):
    print(f"  {k}-char: N={len(strata[k])}, n={ALLOC[k]}")
print("first 5 in presentation order:", [s["head"] for s in sample[:5]])
print("wrote sample.json and vocab_test.html")
