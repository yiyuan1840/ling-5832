import sys
import re
import math
from pathlib import Path
from collections import defaultdict, Counter

base_dir = Path(__file__).parent.parent
sys.path.insert(0, str(base_dir / "src"))

from tokenizer import train_tokenizer, encode, pad   # "find a module named tokenizer"
from ngram import NgramModel


with open(base_dir / "data" / "hobbit.txt", encoding="utf-8") as f:
    hobbit = f.read()

with open(base_dir / "data" / "lostworld.txt", encoding="utf-8") as f:
    lostworld = f.read()

tokenizer = train_tokenizer([hobbit, lostworld], vocab_size=4000)

def split_sentences(text):
    text = text.replace("\n", " ")
    parts = re.split(r"(?<=[.!?]) +", text)

    sentences = []
    for part in parts:
        part = part.strip()
        if part != "":
            sentences.append(part)
    return sentences


all_sentences = split_sentences(hobbit)

cut = int(len(all_sentences) * 0.9)
train_sentences = all_sentences[:cut]
test_sentences = all_sentences[cut:]

print("train:", len(train_sentences))
print("test: ", len(test_sentences))

def average_surprise(model, sequences):
    total = 0.0
    number_of_predictions = 0

    for sequence in sequences:
        surprise, count = model.neg_log_prob(sequence)
        total = total + surprise
        number_of_predictions = number_of_predictions + count

    return total / number_of_predictions

def perplexity(model, sequences):
    return math.exp(average_surprise(model, sequences))

train_sequences = [pad(encode(tokenizer, s), 2) for s in train_sentences]
test_sequences = [pad(encode(tokenizer, s), 2) for s in test_sentences]

model = NgramModel(2, tokenizer.get_vocab_size(), k=0.01)
model.fit(train_sequences)

print("perplexity on training text:", perplexity(model, train_sequences))
print("perplexity on held-out text:", perplexity(model, test_sequences))

for n in [2, 3]:
    train_sequences = [pad(encode(tokenizer, s), n) for s in train_sentences]
    test_sequences = [pad(encode(tokenizer, s), n) for s in test_sentences]

    for k in [1.0, 0.1, 0.01, 0.001]:
        model = NgramModel(n, tokenizer.get_vocab_size(), k=k)
        model.fit(train_sequences)
        print(n, k, perplexity(model, test_sequences))


unseen = 0
total = 0
for sequence in test_sequences:
    for i in range(n - 1, len(sequence)):
        context = tuple(sequence[i - (n - 1):i])
        if model.totals[context] == 0:
            unseen = unseen + 1
        total = total + 1

print(unseen, "of", total, "contexts were never seen in training")