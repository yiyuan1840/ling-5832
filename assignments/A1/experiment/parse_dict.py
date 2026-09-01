# Parse XDHYCD7th.txt (《现代汉语词典》第7版 full text) into a population of headword entries.
# Each entry line starts with 【headword】 followed by pinyin and definition.
import re
import json
import collections
from pathlib import Path

DICT = Path(__file__).parent.parent / "XDHYCD7th" / "XDHYCD7th.txt"

entries = []
pat = re.compile(r"^【(.+?)】(.*)$")
with open(DICT, encoding="utf-8") as f:
    for lineno, line in enumerate(f, 1):
        m = pat.match(line.rstrip("\n"))
        if m:
            head, rest = m.group(1), m.group(2)
            entries.append({"i": len(entries), "line": lineno, "head": head, "rest": rest})

print(f"total entries: {len(entries)}")

by_len = collections.Counter(min(len(e["head"]), 5) for e in entries)
for k in sorted(by_len):
    label = f"{k}-char" if k < 5 else "5+ char"
    print(f"{label}: {by_len[k]}")

# distinct written forms vs entries (same form, multiple pronunciations/senses)
forms = collections.Counter(e["head"] for e in entries)
print(f"distinct written forms: {len(forms)}")
dup = sum(c for c in forms.values() if c > 1)
print(f"entries sharing a form with another entry: {dup}")

# any headwords with unexpected characters (Latin, digits)?
weird = [e["head"] for e in entries if re.search(r"[A-Za-z0-9]", e["head"])]
print(f"headwords containing Latin/digits: {len(weird)}  e.g. {weird[:10]}")

out = Path(__file__).parent / "population.json"
with open(out, "w", encoding="utf-8") as f:
    json.dump(entries, f, ensure_ascii=False)
print(f"wrote {out}")
