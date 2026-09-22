import sys
import re
from pathlib import Path
from collections import defaultdict, Counter


base_dir = Path(__file__).parent.parent
sys.path.insert(0, str(base_dir / "src"))


from tokenizer import train_tokenizer, encode, pad   # "find a module named tokenizer"

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


hobbit_sentences = split_sentences(hobbit)

print("number of sentences:", len(hobbit_sentences))
# print(hobbit_sentences[1])

sequences = []
for sentence in hobbit_sentences:
    numbers = encode(tokenizer, sentence)
    padded = pad(numbers, 2)
    sequences.append(padded)

# print("how many sequences:", len(sequences))
# print()
# print("one sentence:", "In a hole in the ground there lived a hobbit.")
example = pad(encode(tokenizer, "In a hole in the ground there lived a hobbit."), 2)
# print("as numbers:  ", example)

pieces = []
for number in example:
    pieces.append(tokenizer.id_to_token(number))
# print("as pieces:   ", pieces)

counts = defaultdict(Counter)
totals = Counter()

for sequence in sequences:
    for i in range(len(sequence) - 1):
        context = sequence[i]
        next_number = sequence[i + 1]

        counts[context][next_number] = counts[context][next_number] + 1
        totals[context] = totals[context] + 1

# print("contexts we have seen:", len(counts))

the_id = tokenizer.token_to_id("the")

print("'the' has number", the_id)
print("it appeared as a context", totals[the_id], "times")
print()

for next_number, how_many in counts[the_id].most_common(6):
    piece = tokenizer.id_to_token(next_number)
    print("   ", piece, how_many)



dwarves_id = tokenizer.token_to_id("dwarves")

top = counts[the_id][dwarves_id]
bottom = totals[the_id]

print("count of 'the' then 'dwarves':", top)
print("count of 'the' as a context:  ", bottom)
print("probability:", top / bottom)

professor_id = tokenizer.token_to_id("Professor")

print("count of 'the' then 'Professor':", counts[the_id][professor_id])
print("probability:", counts[the_id][professor_id] / totals[the_id])