# Verify the rendered PDF: page count, and no hyphen/dash characters outside URLs.
import re
from pypdf import PdfReader

r = PdfReader(r"c:\dev\personal\ling5832\assignments\A1\assignment1-part1.pdf")
print("pages:", len(r.pages))
full = "".join(p.extract_text() for p in r.pages)
# strip URLs (repo name contains a hyphen) before scanning
scrubbed = re.sub(r"\S*(github\.io|github\.com)\S*", "", full)
hits = [(m.start(), scrubbed[max(0, m.start()-25):m.start()+25].replace("\n", " "))
        for m in re.finditer(r"[-\u2010\u2011\u2012\u2013\u2014\u2015]", scrubbed)]
print("dash/hyphen occurrences outside URLs:", len(hits))
for pos, ctx in hits[:10]:
    print("  ...", ctx, "...")
print("form C numbers present:", "57,760" in full and "64,676" in full and "5,522" in full)
