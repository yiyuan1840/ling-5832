# render_report.py -- turn report/assignment2.md into assignment2.pdf.
#
#   uv run render_report.py
#
# Edit the Markdown, run this, get the PDF. The styling lives here so the Markdown stays
# plain text with nothing but content in it.

import re
import subprocess
import sys
from pathlib import Path

import markdown

HERE = Path(__file__).parent
SOURCE = HERE / "report" / "assignment2.md"
BUILT_HTML = HERE / "report" / "assignment2.built.html"
PDF = HERE / "assignment2.pdf"
EDGE = Path(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe")

# Print styling, carried over from Assignment 1 so the two reports look alike.
CSS = """
@page { size: letter; margin: 0.6in; }
html, body { margin: 0; padding: 0; }
body { font-family: "Times New Roman", serif; font-size: 10pt; line-height: 1.24; color: #000; }
h1 { font-size: 14pt; margin: 0 0 2pt; text-align: center; }
h1 + p { text-align: center; font-size: 9.5pt; margin-bottom: 8pt; }
h2 { font-size: 11pt; margin: 8pt 0 3pt; }
p { margin: 0 0 4pt; text-align: justify; }
table { border-collapse: collapse; margin: 5pt auto 4pt; font-size: 8.5pt;
        page-break-inside: avoid; }
th, td { border: 0.75pt solid #444; padding: 1.5pt 6pt; text-align: center; }
th { font-weight: bold; }
ol, ul { margin: 0 0 4.5pt; padding-left: 18pt; }
li { margin-bottom: 3pt; text-align: justify; }
code { font-family: "Consolas", monospace; font-size: 9pt; }
em:last-child { font-size: 9pt; }
"""

text = SOURCE.read_text(encoding="utf-8")
body = markdown.markdown(text, extensions=["tables", "sane_lists"])
BUILT_HTML.write_text(
    f'<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
    f"<title>Assignment 2</title>\n<style>{CSS}</style>\n</head>\n<body>\n{body}\n</body>\n</html>\n",
    encoding="utf-8")

if not EDGE.exists():
    sys.exit(f"Edge not found at {EDGE}")

subprocess.run([str(EDGE), "--headless", "--disable-gpu", "--no-pdf-header-footer",
                f"--print-to-pdf={PDF}", str(BUILT_HTML)], capture_output=True, check=True)
print(f"wrote {PDF.name} ({PDF.stat().st_size // 1024} KB)")

try:
    from pypdf import PdfReader
except ImportError:
    sys.exit(0)

reader = PdfReader(str(PDF))
print(f"pages: {len(reader.pages)}")
for i, page in enumerate(reader.pages):
    print(f"   page {i + 1}: {len(page.extract_text())} characters")

# The report style avoids hyphens and dashes in prose. Strip code names first, since those
# legitimately contain them.
full = "".join(p.extract_text() for p in reader.pages)
scrubbed = re.sub(r"(\w+\.py|models\.BPE|trainers\.BpeTrainer|train_tokenizer|neg_log_prob"
                  r"|cross_entropy|initial_alphabet|ByteLevel\.alphabet|split_sentences"
                  r"|uv run|src/|n.gram|pre.tokeniz\w*|add.k|add.1|out of vocabulary)",
                  "", full, flags=re.I)
print(f"dashes left in prose: {len(re.findall(r'[-\u2010-\u2015]', scrubbed))}")
