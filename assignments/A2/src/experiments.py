# experiments.py -- produce every number that appears in the report.
#
#   uv run src/experiments.py
#
# Kept separate from main.py so the report's tables are reproducible in one command, and so
# main.py stays short. Nothing here is used by the pipeline itself.
#
# Assembly by Claude; every model function it calls is Yiyuan Jia's.

import re
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
sys.path.insert(0, str(Path(__file__).parent))

from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders

from corpus import load_sentences, train_test_split, make_passages
from tokenizer import train_tokenizer, encode, pad, SPECIALS
from ngram import NgramModel
from authorid import score

AUTHORS = ["tolkien", "doyle"]

train_s, test_s = {}, {}
for a in AUTHORS:
    train_s[a], test_s[a] = train_test_split(load_sentences(a))
train_text = [" ".join(train_s[a]) for a in AUTHORS]


def header(title):
    print("\n" + "=" * 74)
    print(title)
    print("=" * 74)


# ------------------------------------------------------------------ 1. corpus
header("TABLE 1. Corpus")
print(f"   {'author':10s} {'sentences':>10s} {'words':>9s} {'held out':>9s} {'word types':>11s}")
for a in AUTHORS:
    allw = " ".join(train_s[a] + test_s[a])
    print(f"   {a:10s} {len(train_s[a]) + len(test_s[a]):10d} {len(allw.split()):9d} "
          f"{len(test_s[a]):9d} {len(set(re.findall(r'[a-z]+', allw.lower()))):11d}")
hv = set(re.findall(r"[a-z]+", " ".join(train_s['tolkien'] + test_s['tolkien']).lower()))
dv = set(re.findall(r"[a-z]+", " ".join(train_s['doyle'] + test_s['doyle']).lower()))
print(f"   word types shared by both books: {len(hv & dv)}")

# ------------------------------------------------------------------ 2. pre-tokenizers
header("TABLE 2. Pre-tokenization strategy (vocab 4000)")


def build(pretok, vocab_size):
    tk = Tokenizer(models.BPE(unk_token="<unk>"))
    kw = dict(vocab_size=vocab_size, special_tokens=SPECIALS, show_progress=False)
    if pretok == "whitespace only":
        tk.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    elif pretok == "whitespace + punctuation":
        tk.pre_tokenizer = pre_tokenizers.Sequence(
            [pre_tokenizers.WhitespaceSplit(), pre_tokenizers.Punctuation()])
    elif pretok == "byte level":
        tk.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        tk.decoder = decoders.ByteLevel()
        kw["initial_alphabet"] = pre_tokenizers.ByteLevel.alphabet()
    tk.train_from_iterator(train_text, trainers.BpeTrainer(**kw))
    return tk


print(f"   {'strategy':26s} {'tok/word':>9s} {'punct-glued types':>18s} {'test perplexity':>16s}")
for name in ("whitespace only", "whitespace + punctuation", "byte level"):
    tk = build(name, 4000)
    words = len(" ".join(train_text).split())
    ntok = sum(len(tk.encode(t).ids) for t in train_text)
    glued = sum(1 for t in tk.get_vocab()
                if len(t) > 2 and any(c in t for c in '"\'!?,.;:“”’'))
    m = NgramModel(2, tk.get_vocab_size(), k=0.01)
    m.fit([pad(tk.encode(s).ids, 2) for s in train_s["tolkien"]])
    pp = m.perplexity([pad(tk.encode(s).ids, 2) for s in test_s["tolkien"]])
    print(f"   {name:26s} {ntok / words:9.2f} {glued:18d} {pp:16.1f}")

# ------------------------------------------------------------------ 3. vocab size
header("TABLE 3. Vocabulary size (bigram, k=0.01, Tolkien)")
print(f"   {'requested':>10s} {'realized':>9s} {'tok/word':>9s} {'perplexity':>11s} {'accuracy':>9s}")
for vs in (500, 1000, 2000, 4000, 8000, 16000):
    tk = train_tokenizer(train_text, vocab_size=vs)
    V = tk.get_vocab_size()
    words = len(" ".join(train_text).split())
    ntok = sum(len(encode(tk, t)) for t in train_text)
    ms = {}
    for a in AUTHORS:
        m = NgramModel(2, V, k=0.01)
        m.fit([pad(encode(tk, s), 2) for s in train_s[a]])
        ms[a] = m
    pp = ms["tolkien"].perplexity([pad(encode(tk, s), 2) for s in test_s["tolkien"]])
    right = tot = 0
    for truth in AUTHORS:
        for p in make_passages(test_s[truth], 50):
            g = min(AUTHORS, key=lambda a: score(p, tk, ms[a], 2))
            right += g == truth
            tot += 1
    print(f"   {vs:10d} {V:9d} {ntok / words:9.2f} {pp:11.1f} {right / tot:9.1%}")

# ------------------------------------------------------------------ 4. n and k
header("TABLE 4. Model order and smoothing (vocab 4000)")
tk = train_tokenizer(train_text, vocab_size=4000)
V = tk.get_vocab_size()
print(f"   {'n':>2} {'k':>7} {'perplexity':>11} {'unseen contexts':>16s} {'accuracy':>9s}")
for n in (2, 3):
    tr = {a: [pad(encode(tk, s), n) for s in train_s[a]] for a in AUTHORS}
    te = {a: [pad(encode(tk, s), n) for s in test_s[a]] for a in AUTHORS}
    for k in (1.0, 0.5, 0.1, 0.01, 0.001):
        ms = {}
        for a in AUTHORS:
            m = NgramModel(n, V, k=k)
            m.fit(tr[a])
            ms[a] = m
        pp = ms["tolkien"].perplexity(te["tolkien"])
        unseen = seen = 0
        for s in te["tolkien"]:
            for i in range(n - 1, len(s)):
                unseen += ms["tolkien"].totals[tuple(s[i - (n - 1):i])] == 0
                seen += 1
        right = tot = 0
        for truth in AUTHORS:
            for p in make_passages(test_s[truth], 50):
                g = min(AUTHORS, key=lambda a: score(p, tk, ms[a], n))
                right += g == truth
                tot += 1
        print(f"   {n:>2} {k:>7} {pp:11.1f} {unseen / seen:15.1%} {right / tot:9.1%}")

# ------------------------------------------------------------------ 5. passage length
header("TABLE 5. Accuracy by passage length (bigram, k=0.01, vocab 4000)")
ms = {}
for a in AUTHORS:
    m = NgramModel(2, V, k=0.01)
    m.fit([pad(encode(tk, s), 2) for s in train_s[a]])
    ms[a] = m
print(f"   {'words':>6s} {'passages':>9s} {'accuracy':>9s}")
for wp in (10, 25, 50, 100, 200):
    right = tot = 0
    for truth in AUTHORS:
        for p in make_passages(test_s[truth], wp):
            g = min(AUTHORS, key=lambda a: score(p, tk, ms[a], 2))
            right += g == truth
            tot += 1
    print(f"   {wp:6d} {tot:9d} {right / tot:9.1%}")

# ------------------------------------------------------------------ 6. what is it using
header("TABLE 6. What is the classifier actually reading?")
NAMES = ["Bilbo", "Gandalf", "Thorin", "Baggins", "Gollum", "Smaug", "Bard", "Beorn", "Elrond",
         "Balin", "Dwalin", "Fili", "Kili", "Dori", "Nori", "Ori", "Oin", "Gloin", "Bifur",
         "Bofur", "Bombur", "Challenger", "Summerlee", "Roxton", "Malone", "Gladys", "McArdle",
         "Zambo", "Enmore"]


def fix_quotes(t):
    t = t.replace("“", '"').replace("”", '"')
    t = t.replace("‘", "'").replace("’", "'")
    t = re.sub(r"--+|[–—]", " -- ", t)
    return re.sub(r"\s+", " ", t)


def mask_names(t):
    for nm in NAMES:
        t = re.sub(r"\b" + nm + r"\b", "PERSON", t)
    return t


def variant(fn, wp):
    tr = {a: [fn(s) for s in train_s[a]] for a in AUTHORS}
    te = {a: [fn(s) for s in test_s[a]] for a in AUTHORS}
    t2 = train_tokenizer([" ".join(tr[a]) for a in AUTHORS], vocab_size=4000)
    V2 = t2.get_vocab_size()
    mm = {}
    for a in AUTHORS:
        m = NgramModel(2, V2, k=0.01)
        m.fit([pad(encode(t2, s), 2) for s in tr[a]])
        mm[a] = m
    right = tot = 0
    for truth in AUTHORS:
        for p in make_passages(te[truth], wp):
            g = min(AUTHORS, key=lambda a: score(p, t2, mm[a], 2))
            right += g == truth
            tot += 1
    return right / tot


print(f"   {'text given to the model':34s} {'50 words':>9s} {'10 words':>9s}")
for label, fn in (("unchanged", lambda s: s),
                  ("quotes and dashes normalized", fix_quotes),
                  ("character names masked", mask_names),
                  ("both", lambda s: mask_names(fix_quotes(s)))):
    print(f"   {label:34s} {variant(fn, 50):9.1%} {variant(fn, 10):9.1%}")

# ------------------------------------------------------------------ 7. cross perplexity
header("TABLE 7. Each model on each author (bigram, k=0.01, vocab 4000)")
print(f"   {'model':10s} {'on Tolkien':>12s} {'on Doyle':>12s}")
for a in AUTHORS:
    row = [ms[a].perplexity([pad(encode(tk, s), 2) for s in test_s[b]]) for b in AUTHORS]
    print(f"   {a:10s} {row[0]:12.1f} {row[1]:12.1f}")

print("\ndone")
