# toy_bpe.py -- a teaching demo. Watch BPE learn merges on the lecture's toy corpus, see what
# each pre-tokenization strategy does to a real sentence, and check the <unk> claim.
#
#   python src/toy_bpe.py              the merge walkthrough and pre-tokenizer comparison
#   python src/toy_bpe.py --check-unk  the out-of-vocabulary experiment (needs your TODO 2)
#
# Provenance: written by Claude (Anthropic) as a teaching aid; not part of the graded pipeline.

import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
sys.path.insert(0, str(Path(__file__).parent))

from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders

HERE = Path(__file__).parent
DATA = HERE.parent / "data"

# ------------------------------------------------------------------ 1. the toy corpus
# This is the corpus from lecture 3, slide 9. The lecture treats whitespace as the leading
# character of a word, which is what Metaspace does with its visible underscore.
TOY = ["set new new renew reset renew"]

print("=" * 78)
print("1. BPE MERGES ON THE LECTURE'S TOY CORPUS")
print("=" * 78)
print(f"corpus: {TOY[0]!r}")
print("\nBPE starts from single characters and repeatedly merges the most frequent adjacent")
print("pair. Each merge adds one new vocabulary item. Here they are in the order learned:\n")

tk = Tokenizer(models.BPE())
tk.pre_tokenizer = pre_tokenizers.Metaspace()
tk.decoder = decoders.Metaspace()
tk.train_from_iterator(TOY, trainers.BpeTrainer(vocab_size=22, show_progress=False))

merges = json.loads(tk.to_str())["model"]["merges"]
for i, m in enumerate(merges, 1):
    pair = tuple(m) if isinstance(m, list) else tuple(m.split(" "))
    print(f"   merge {i:2d}:  ({pair[0]!r}, {pair[1]!r})  ->  {''.join(pair)!r}")

print("\nNow encode some strings with the merges above, applied greedily in learned order:")
for s in ["set", "new", "reset", "rereset", "set renew"]:
    toks = tk.encode(s).tokens
    print(f"   {s!r:14s} -> {toks}")
print("\nNote 'rereset', a word the corpus never contained, still gets encoded. That is the")
print("whole point of subword tokenization: an open vocabulary over a fixed-size table.")

# ------------------------------------------------------------------ 2. pre-tokenizers
print("\n" + "=" * 78)
print("2. WHAT EACH PRE-TOKENIZATION STRATEGY DOES")
print("=" * 78)
print("Pre-tokenization runs BEFORE BPE and fixes boundaries no merge may cross.")
print("Watch what happens to the quote characters in particular.\n")

SAMPLE = '"Good morning!" said Bilbo, and he meant it -- the sun was shining.'
print(f"input: {SAMPLE}\n")

configs = {
    "ws   (Whitespace)": pre_tokenizers.Whitespace(),
    "wsp  (WhitespaceSplit + Punctuation)": pre_tokenizers.Sequence(
        [pre_tokenizers.WhitespaceSplit(), pre_tokenizers.Punctuation()]),
    "byte (ByteLevel)": pre_tokenizers.ByteLevel(add_prefix_space=False),
    "meta (Metaspace)": pre_tokenizers.Metaspace(),
}
for name, pre in configs.items():
    pieces = [p for p, _ in pre.pre_tokenize_str(SAMPLE)]
    print(f"   {name}")
    print(f"      {pieces[:16]}")

# ------------------------------------------------------------------ 3. vocabulary effect
print("\n" + "=" * 78)
print("3. WHAT REACHES THE VOCABULARY, ON THE REAL CORPUS")
print("=" * 78)
clean = DATA / "clean" / "tolkien.minimal.json"
if not clean.exists():
    print("   run src/prepare_data.py first")
else:
    texts = [" ".join(json.load(open(clean, encoding="utf-8"))["paragraphs"])]
    doyle = DATA / "clean" / "doyle.minimal.json"
    texts.append(" ".join(json.load(open(doyle, encoding="utf-8"))["paragraphs"]))
    for name, pre, dec in (
            ("wsp ", pre_tokenizers.Sequence(
                [pre_tokenizers.WhitespaceSplit(), pre_tokenizers.Punctuation()]), None),
            ("meta", pre_tokenizers.Metaspace(), decoders.Metaspace())):
        t = Tokenizer(models.BPE(unk_token="<unk>"))
        t.pre_tokenizer = pre
        if dec:
            t.decoder = dec
        t.train_from_iterator(texts, trainers.BpeTrainer(
            vocab_size=8000, special_tokens=["<unk>", "<s>", "</s>"], show_progress=False))
        dirty = [tok for tok in t.get_vocab()
                 if len(tok) > 3 and any(c in tok for c in '"\'!?,.;:')]
        print(f"\n   {name}: {len(dirty)} vocabulary items contain a punctuation character")
        if dirty:
            print(f"         examples: {sorted(dirty)[:8]}")
    print("\n   That is the finding: with no punctuation split, BPE bakes the punctuation")
    print("   into word types, so the vocabulary itself records which edition the text")
    print("   came from. Splitting punctuation first makes the leak structurally impossible.")

# ------------------------------------------------------------------ 4. the unk experiment
if "--check-unk" in sys.argv:
    print("\n" + "=" * 78)
    print("4. DOES BYTE-LEVEL BPE REALLY HAVE NO OUT-OF-VOCABULARY WORDS?")
    print("=" * 78)
    import bpe as bpemod
    probe = ["café", "你好", "☃", "naïve"]
    print(f"probe strings the corpus never contains: {probe}\n")
    for seeded in (False, True):
        t = Tokenizer(models.BPE(unk_token="<unk>"))
        t.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        t.decoder = decoders.ByteLevel()
        kw = dict(vocab_size=2000, special_tokens=["<unk>", "<s>", "</s>"], show_progress=False)
        if seeded:
            kw["initial_alphabet"] = pre_tokenizers.ByteLevel.alphabet()
        t.train_from_iterator(texts, trainers.BpeTrainer(**kw))
        n_unk = sum(t.encode(p).ids.count(0) for p in probe)
        exact = all(t.decode(t.encode(p).ids) == p for p in probe)
        print(f"   initial_alphabet seeded = {str(seeded):5s} ->  <unk> tokens = {n_unk:2d}"
              f"   exact round trip = {exact}")
    print("\n   Lecture 3 slide 19 says BPE has no out-of-vocabulary words because the")
    print("   vocabulary starts from UTF-8. True of the algorithm, false of the default")
    print("   configuration. This is the kind of observation the report asks for.")
