# Stratified analysis of vocab_results.json -> vocabulary-size estimates with CIs.
# Usage: python analyze.py [path-to-results.json]  (default: ./vocab_results.json)
import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).parent
Z = 1.959963985

results_path = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "vocab_results.json"
sample_meta = json.load(open(HERE / "sample.json", encoding="utf-8"))["meta"]
N = {int(k): v for k, v in sample_meta["strata_N"].items()}

payload = json.load(open(results_path, encoding="utf-8"))
rows = payload["results"]
assert len(rows) == sum(sample_meta["strata_n"].values()), f"expected 360 rows, got {len(rows)}"

RATING = ["不认识", "眼熟", "认识", "掌握"]
VERIFY = {2: "对", 1: "部分对", 0: "错了", None: "-"}

# "know" operationalizations
defs = {
    "self-report only (rating>=认识)":       lambda r: r["rating"] >= 2,
    "receptive, lenient (verify 对/部分对)":  lambda r: r["rating"] >= 2 and r["verify"] in (1, 2),
    "receptive, strict (verify 对)":          lambda r: r["rating"] >= 2 and r["verify"] == 2,
    "productive (掌握 & verify 对)":          lambda r: r["rating"] == 3 and r["verify"] == 2,
}

def wilson(k, n):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + Z * Z / n
    c = p + Z * Z / (2 * n)
    h = Z * math.sqrt(p * (1 - p) / n + Z * Z / (4 * n * n))
    return ((c - h) / d, (c + h) / d)

def estimate(pred):
    total, var, lo_w, hi_w = 0.0, 0.0, 0.0, 0.0
    per = {}
    for h in sorted(N):
        sub = [r for r in rows if min(r["len"], 5) == h]
        n_h, k_h = len(sub), sum(1 for r in sub if pred(r))
        p = k_h / n_h
        fpc = 1 - n_h / N[h]
        total += N[h] * p
        var += (N[h] ** 2) * fpc * p * (1 - p) / (n_h - 1)
        wl, wh = wilson(k_h, n_h)
        lo_w += N[h] * wl
        hi_w += N[h] * wh
        per[h] = (n_h, k_h, p)
    se = math.sqrt(var)
    return total, total - Z * se, total + Z * se, lo_w, hi_w, per

print(f"results: {results_path.name}, exported {payload.get('exportedAt','?')}")
print(f"population: {sample_meta['population_total']} entries\n")

print("rating distribution:")
for i, lab in enumerate(RATING):
    print(f"  {lab}: {sum(1 for r in rows if r['rating']==i)}")
claimed = [r for r in rows if r["rating"] >= 2]
over = sum(1 for r in claimed if r["verify"] == 0)
part = sum(1 for r in claimed if r["verify"] == 1)
print(f"claimed known: {len(claimed)}; verified 错了: {over} ({100*over/len(claimed):.1f}%), "
      f"部分对: {part} ({100*part/len(claimed):.1f}%)\n")

for name, pred in defs.items():
    tot, lo, hi, lw, hw, per = estimate(pred)
    print(f"== {name} ==")
    for h in sorted(per):
        n_h, k_h, p = per[h]
        lab = f"{h}-char" if h < 5 else "5+chr "
        print(f"  {lab}: {k_h:3d}/{n_h:3d} known (p={p:.3f})  -> {N[h]*p:8.0f} of {N[h]}")
    print(f"  TOTAL: {tot:.0f}   95% CI (normal): [{lo:.0f}, {hi:.0f}]   "
          f"(Wilson-summed: [{lw:.0f}, {hw:.0f}])\n")

# response-time bonus: median ms to first judgment by rating
try:
    import statistics
    print("median ms to first judgment, by rating:")
    for i, lab in enumerate(RATING):
        ms = [r["ms1"] for r in rows if r["rating"] == i and r.get("ms1")]
        if ms:
            print(f"  {lab}: {statistics.median(ms):.0f} ms (n={len(ms)})")
except Exception:
    pass
