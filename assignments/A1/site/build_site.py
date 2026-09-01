# Build the public GitHub Pages site into <repo root>/docs/ from the seeded sample.
import json
import shutil
from pathlib import Path

HERE = Path(__file__).parent                     # assignments/A1/site
REPO = HERE.parent.parent.parent                 # ling5832 repo root
DOCS = REPO / "docs" / "vocab-test"
DOCS.mkdir(parents=True, exist_ok=True)

data = json.load(open(HERE.parent / "experiment" / "sample.json", encoding="utf-8"))
meta = dict(data["meta"], site="ling-5832 vocab test v1")

template = (HERE / "site_template.html").read_text(encoding="utf-8")
html = template.replace("__DATA__", json.dumps(data["sample"], ensure_ascii=False)) \
               .replace("__META__", json.dumps(meta, ensure_ascii=False))
(DOCS / "index.html").write_text(html, encoding="utf-8")
shutil.copy(HERE / "stats.js", DOCS / "stats.js")
print(f"wrote {DOCS/'index.html'} ({len(html):,} chars) and stats.js")
