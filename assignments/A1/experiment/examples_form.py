# Pull narrative examples from a results file taken on any of the five forms.
import json
from pathlib import Path

here = Path(__file__).parent
payload = json.load(open(here / "vocab_results.json", encoding="utf-8"))
form = payload.get("form", "A")
forms = json.load(open(here / "forms.json", encoding="utf-8"))["forms"]
lookup = {it["id"]: it for it in forms[form]}
rows = payload["results"]
print(f"form {form}, {len(rows)} rows")

def show(label, pred, cap=12):
    hits = [r for r in rows if pred(r)]
    print(f"\n{label} ({len(hits)}):")
    for r in hits[:cap]:
        rest = lookup[r["id"]]["rest"]
        print(f"  {r['head']} [{r['len']}] | {rest[:65]}")

show("rated 不认识", lambda r: r["rating"] == 0)
show("claimed but 错了", lambda r: r["rating"] >= 2 and r["verify"] == 0)
show("claimed but 部分对", lambda r: r["rating"] >= 2 and r["verify"] == 1)

blank = [r for r in rows if not lookup[r["id"]]["rest"].strip()]
claimed_blank = [r for r in blank if r["rating"] >= 2]
print(f"\nblank-definition items: {len(blank)}, of which claimed: {len(claimed_blank)}")
