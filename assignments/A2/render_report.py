# render_report.py -- turn report/assignment2.md into assignment2.pdf.
#
#   uv run render_report.py              report/assignment2.md -> assignment2.pdf
#   uv run render_report.py submission   report/submission.md  -> submission.pdf
#
# Edit the Markdown, run this, get the PDF. The styling lives here so the Markdown stays
# plain text with nothing but content in it.

import re
import subprocess
import sys
import tempfile
from pathlib import Path

import markdown

HERE = Path(__file__).parent
NAME = sys.argv[1] if len(sys.argv) > 1 else "assignment2"
SOURCE = HERE / "report" / f"{NAME}.md"
PDF = HERE / f"{NAME}.pdf"

if not SOURCE.exists():
    sys.exit(f"no such source: {SOURCE}")
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

if not EDGE.exists():
    sys.exit(f"Edge not found at {EDGE}")

text = SOURCE.read_text(encoding="utf-8")
body = markdown.markdown(text, extensions=["tables", "sane_lists"])
page = (f'<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
        f"<title>Assignment 2</title>\n<style>{CSS}</style>\n</head>\n"
        f"<body>\n{body}\n</body>\n</html>\n")

# Edge needs a file to read. It is written into the project, rendered, then removed, so it does
# not linger next to the Markdown looking like something you are meant to edit.
built = HERE / "_report_build.html"

# Remove any previous PDF first. Edge does not reliably overwrite one, and the failure is
# quiet: the page content comes out correct but the hyperlink annotations are dropped.
PDF.unlink(missing_ok=True)

try:
    built.write_text(page, encoding="utf-8")
    # --user-data-dir gives this run its own throwaway profile. Without it, headless Edge
    # tries to hand the job to an already running Edge and hangs indefinitely if you happen
    # to have the browser open.
    with tempfile.TemporaryDirectory() as profile:
        subprocess.run([str(EDGE), "--headless", "--disable-gpu", "--no-pdf-header-footer",
                        f"--user-data-dir={profile}",
                        f"--print-to-pdf={PDF}", str(built)],
                       capture_output=True, check=True, timeout=120)
finally:
    built.unlink(missing_ok=True)
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
full = re.sub(r"\S*(github\.com|github\.io|https?://)\S*", "", full)
scrubbed = re.sub(r"(\w+\.py|models\.BPE|trainers\.BpeTrainer|train_tokenizer|neg_log_prob"
                  r"|cross_entropy|initial_alphabet|ByteLevel\.alphabet|split_sentences"
                  r"|uv run|src/|n.gram|pre.tokeniz\w*|add.k|add.1|out of vocabulary"
                  r"|a2.submitted)",
                  "", full, flags=re.I)
print(f"dashes left in prose: {len(re.findall(r'[-\u2010-\u2015]', scrubbed))}")
