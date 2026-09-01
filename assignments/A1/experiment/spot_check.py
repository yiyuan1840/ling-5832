import json
import linecache
from pathlib import Path

here = Path(__file__).parent
s = json.load(open(here / "sample.json", encoding="utf-8"))["sample"]
d = here.parent / "XDHYCD7th" / "XDHYCD7th.txt"
for e in s[:4] + s[-4:]:
    src = linecache.getline(str(d), e["line"]).rstrip()
    ok = src.startswith("【" + e["head"] + "】")
    print(ok, e["len"], e["head"], "|", src[:50])
