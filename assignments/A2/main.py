# main.py -- run the whole thing end to end and print the results.
#
#   uv run main.py
#
# This is the file that connects the four parts. Nothing new is computed here; it just
# calls the pieces in order and prints what they say.
#
# Assembly by Claude. All the logic it calls is Yiyuan Jia's.

import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
sys.path.insert(0, str(Path(__file__).parent / "src"))

from corpus import load_sentences, train_test_split, make_passages
from tokenizer import train_tokenizer, encode, pad
from ngram import NgramModel
from authorid import score

# Settings. These are the choices the assignment asks us to explore; see the report for why.
VOCAB_SIZE = 4000
N = 2
K = 0.01
PASSAGE_WORDS = 50

AUTHORS = ["tolkien", "doyle"]

# ---------------------------------------------------------------- 1. load and split
train_sentences = {}
test_sentences = {}
for author in AUTHORS:
    all_sentences = load_sentences(author)
    train_sentences[author], test_sentences[author] = train_test_split(all_sentences)

print("CORPUS")
for author in AUTHORS:
    n_train = len(train_sentences[author])
    n_test = len(test_sentences[author])
    words = len(" ".join(train_sentences[author]).split())
    print(f"   {author:8s} {n_train:5d} training sentences ({words} words), {n_test} held back")

# ---------------------------------------------------------------- 2. one shared vocabulary
# Trained on BOTH authors' training text, so the two models below are comparable. Training
# text only: fitting a vocabulary to text we later test on would be cheating.
training_text = [" ".join(train_sentences[author]) for author in AUTHORS]
tokenizer = train_tokenizer(training_text, vocab_size=VOCAB_SIZE)
V = tokenizer.get_vocab_size()

print(f"\nTOKENIZER\n   {V} pieces in the vocabulary")
example = "In a hole in the ground there lived a hobbit."
print(f"   {example!r}")
print(f"   -> {[tokenizer.id_to_token(i) for i in encode(tokenizer, example)]}")

# ---------------------------------------------------------------- 3. one model per author
models = {}
for author in AUTHORS:
    model = NgramModel(N, V, k=K)
    model.fit([pad(encode(tokenizer, s), N) for s in train_sentences[author]])
    models[author] = model

# ---------------------------------------------------------------- 4. perplexity
print(f"\nPERPLEXITY (n={N}, k={K})")
print(f"   {'model':10s} {'on its own author':>18s} {'on the other':>14s}")
for author in AUTHORS:
    own = models[author].perplexity(
        [pad(encode(tokenizer, s), N) for s in test_sentences[author]])
    other_author = [a for a in AUTHORS if a != author][0]
    other = models[author].perplexity(
        [pad(encode(tokenizer, s), N) for s in test_sentences[other_author]])
    print(f"   {author:10s} {own:18.1f} {other:14.1f}")

print("\n   Each model is far less surprised by its own author than by the other one.")
print("   That gap is what makes author identification work.")

# ---------------------------------------------------------------- 5. author identification
print(f"\nAUTHOR IDENTIFICATION ({PASSAGE_WORDS}-word passages)")
right = 0
total = 0
per_author = {author: [0, 0] for author in AUTHORS}

for true_author in AUTHORS:
    passages = make_passages(test_sentences[true_author], PASSAGE_WORDS)
    for passage in passages:
        # Score the passage under each author's model. The lower score wins, because a
        # lower score means that model found the passage less surprising.
        tolkien_score = score(passage, tokenizer, models["tolkien"], N)
        doyle_score = score(passage, tokenizer, models["doyle"], N)

        if tolkien_score < doyle_score:
            guess = "tolkien"
        else:
            guess = "doyle"

        right += guess == true_author
        total += 1
        per_author[true_author][0] += guess == true_author
        per_author[true_author][1] += 1

for author in AUTHORS:
    got, seen = per_author[author]
    print(f"   {author:10s} {got:4d} / {seen:4d} correct")
print(f"   {'overall':10s} {right:4d} / {total:4d} = {right / total:.1%}")
