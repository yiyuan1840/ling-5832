# Extract the built page's inline script for a node syntax check.
import re
from pathlib import Path

page = Path(__file__).parent.parent.parent.parent / "docs" / "vocab-test" / "index.html"
h = page.read_text(encoding="utf-8")
m = re.findall(r"<script>(.*?)</script>", h, re.S)
out = page.parent / "_inline.js"
out.write_text(m[0], encoding="utf-8")
print("placeholders left:", "__DATA__" in h or "__META__" in h)
print("forms present:", all(f'"{f}": [' in h or f'"{f}":[' in h for f in "ABCDE"))
