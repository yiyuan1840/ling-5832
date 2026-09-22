# test_part1.py -- checks for the three TODOs in bpe.py. Run: python src/test_part1.py
#
# Every check prints one line and asserts. A failure tells you which TODO is wrong and what
# the expected behaviour was. Exit code is non-zero if anything fails.
#
# Provenance: written by Claude (Anthropic); the code under test is Yiyuan Jia's.

import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
sys.path.insert(0, str(Path(__file__).parent))

import bpe as bpemod
from bpe import BpeTokenizer, SPECIALS
from tokenizers import pre_tokenizers

HERE = Path(__file__).parent
CLEAN = HERE.parent / "data" / "clean"

ok = lambda label, cond, note="": (
    print(f"   [{'ok  ' if cond else 'FAIL'}] {label}" + (f"   {note}" if note else "")),
    cond)[1]
failures = []


def check(label, cond, note=""):
    if not ok(label, cond, note):
        failures.append(label)


if not (CLEAN / "tolkien.minimal.json").exists():
    sys.exit("run src/prepare_data.py first")
TEXTS = [" ".join(json.load(open(CLEAN / f"{s}.minimal.json", encoding="utf-8"))["paragraphs"])
         for s in ("tolkien", "doyle")]
SENTS = json.load(open(CLEAN / "tolkien.minimal.json", encoding="utf-8"))["sentences"][:200]

print("=" * 78)
print("ASSIGNMENT COMPLIANCE (static scan, needs no code from you)")
print("=" * 78)
# The assignment says to use Hugging Face's BPE implementation but none of their pre-trained
# tokenizers. That is easy to satisfy and easy to claim, so it is checked here instead.
BANNED = ["from_pretrained", "AutoTokenizer", "GPT2Tokenizer", "hf_hub_download",
          "snapshot_download", "BertTokenizer",
          # tiktoken is in pyproject.toml and is fine to experiment with, but every encoding
          # it offers is a pre-trained OpenAI vocabulary, so it must not reach the pipeline.
          "tiktoken", "get_encoding", "encoding_for_model"]
# This file is excluded from its own scan: it necessarily contains the banned names as the
# string literals it searches for.
sources = {p.name: p.read_text(encoding="utf-8") for p in HERE.glob("*.py")
           if p.name != "test_part1.py"}
# Look for actual call or import syntax rather than a bare mention, so that prose in a
# comment or docstring explaining what we do NOT do does not trip the check.
hits = []
for fname, src in sources.items():
    for b in BANNED:
        for line in src.splitlines():
            stripped = line.lstrip()
            if stripped.startswith("#"):
                continue
            if b + "(" in line or (stripped.startswith(("import ", "from ")) and b in line):
                hits.append((fname, b, stripped[:50]))
check("no module loads a pre-trained tokenizer", not hits, f"found {hits}")
check("BPE comes from the Hugging Face tokenizers package",
      "models.BPE" in sources["bpe.py"] and "trainers.BpeTrainer" in sources["bpe.py"])
check("no third-party n-gram or NLP toolkit is imported",
      not any(m in src for src in sources.values()
              for m in ("import nltk", "import transformers", "from nltk", "kenlm", "srilm")))

print("\n" + "=" * 78)
print("TODO 1: the whitespace + punctuation pre-tokenizer")
print("=" * 78)
try:
    pre, dec, lossless = BpeTokenizer(pretok="wsp")._make_pretokenizer()
    got = [p for p, _ in pre.pre_tokenize_str('He said "no!" -- twice.')]
    check("every punctuation character is its own pre-token",
          got == ["He", "said", '"', "no", "!", '"', "-", "-", "twice", "."],
          f"got {got}")
    check("decoder is None and lossless is False", dec is None and lossless is False)
except NotImplementedError:
    check("TODO 1 implemented", False, "still raises NotImplementedError")

print("\n" + "=" * 78)
print("TODO 2: the trainer")
print("=" * 78)
try:
    t = BpeTokenizer(vocab_size=2000, pretok="wsp").train(TEXTS)
    check("special tokens take indices 0, 1, 2",
          [t.token_to_id(s) for s in SPECIALS] == [0, 1, 2],
          f"got {[t.token_to_id(s) for s in SPECIALS]}")
    check("realized vocabulary size is the requested 2000", t.vocab_size == 2000,
          f"got {t.vocab_size}")
    dirty = [tok for tok in t.vocab() if len(tok) > 2 and any(c in tok for c in '"\'!?,.;:')]
    check("no vocabulary item fuses punctuation into a word (the leak cannot enter)",
          not dirty, f"found {dirty[:5]}")

    # The vocabulary must come from OUR training material, not from anywhere else. Two
    # tokenizers trained on the two different books should disagree, and each should contain
    # words specific to the book it saw.
    only_t = BpeTokenizer(vocab_size=2000, pretok="wsp").train([TEXTS[0]])
    only_d = BpeTokenizer(vocab_size=2000, pretok="wsp").train([TEXTS[1]])
    check("vocabulary is learned from the corpus, not loaded from elsewhere",
          set(only_t.vocab()) != set(only_d.vocab()))
    check("a Tolkien-only vocabulary contains Tolkien words",
          any(w in only_t.vocab() for w in ("Bilbo", "hobbit", "Gandalf")))
    check("a Doyle-only vocabulary does not",
          not any(w in only_d.vocab() for w in ("Bilbo", "Gandalf")))

    probe = ["café", "你好", "☃", "naïve"]
    b = BpeTokenizer(vocab_size=2000, pretok="byte").train(TEXTS)
    n_unk = sum(b.encode(p).count(b.unk_id) for p in probe)
    check("byte-level emits no <unk> on unseen characters (initial_alphabet seeded)",
          n_unk == 0, f"got {n_unk} <unk> tokens; did you pass initial_alphabet?")
    check("byte-level round trips exactly",
          all(b.decode(b.encode(p)) == p for p in probe))
    check("byte-level reports itself lossless", b.lossless is True)

    w = BpeTokenizer(vocab_size=2000, pretok="wsp").train(TEXTS)
    check("wsp reports itself lossy (punctuation spacing is not recoverable)",
          w.lossless is False)
except NotImplementedError:
    check("TODO 2 implemented", False, "still raises NotImplementedError")

print("\n" + "=" * 78)
print("TODO 3: sentence padding")
print("=" * 78)
try:
    t = BpeTokenizer(vocab_size=2000, pretok="wsp").train(TEXTS)
    raw = t.encode("Bilbo went home.")
    p2 = t.encode_sentences(["Bilbo went home."], 2)[0]
    p3 = t.encode_sentences(["Bilbo went home."], 3)[0]
    check("bigram padding is one BOS then the sentence then EOS",
          p2 == [t.bos_id] + raw + [t.eos_id], f"got {p2}")
    check("trigram padding is two BOS then the sentence then EOS",
          p3 == [t.bos_id, t.bos_id] + raw + [t.eos_id], f"got {p3}")
    check("one sequence returned per input sentence",
          len(t.encode_sentences(SENTS, 3)) == len(SENTS))

    # The identity that makes the two model orders comparable. An n-gram model of order n
    # scores len(seq) - (n - 1) events, so with (n - 1) BOS both orders score len(ids) + 1.
    s2 = t.encode_sentences(SENTS, 2)
    s3 = t.encode_sentences(SENTS, 3)
    e2 = sum(len(s) - 1 for s in s2)
    e3 = sum(len(s) - 2 for s in s3)
    check("bigram and trigram score the same number of events", e2 == e3,
          f"bigram {e2} vs trigram {e3}")
    check("event count equals tokens plus one end symbol per sentence",
          e2 == sum(len(t.encode(s)) for s in SENTS) + len(SENTS))
except NotImplementedError:
    check("TODO 3 implemented", False, "still raises NotImplementedError")

print("\n" + "=" * 78)
if failures:
    print(f"{len(failures)} check(s) failed:")
    for f in failures:
        print(f"   - {f}")
    sys.exit(1)
print("Part 1 complete. All checks passed.")
