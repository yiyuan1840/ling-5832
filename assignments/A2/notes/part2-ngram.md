# Part 2: count word patterns and turn them into probabilities

Keep working in `trial/spaghetti.py`. Six steps.

You can delete your Part 1 experiments and start the file with this, since your tokenizer is
finished now:

```python
import sys
import re
from pathlib import Path

base_dir = Path(__file__).parent.parent
sys.path.insert(0, str(base_dir / "src"))

from tokenizer import train_tokenizer, encode, pad

with open(base_dir / "data" / "hobbit.txt", encoding="utf-8") as f:
    hobbit = f.read()

with open(base_dir / "data" / "lostworld.txt", encoding="utf-8") as f:
    lostworld = f.read()

tokenizer = train_tokenizer([hobbit, lostworld], vocab_size=4000)
```

---

## What we are building

A **language model** answers one question: how likely is this piece of text?

That sounds vague until you write it down. The probability of a sentence is the probability of
its first word, times the probability of the second given the first, times the probability of
the third given the first two, and so on.

The problem is the "and so on". To know the probability of a word given the nine words before
it, you would need to have seen those exact ten words together, and you never have.

So we cheat, deliberately. We assume a word depends only on the **one or two words right before
it** and we ignore everything earlier. Looking back one word is called a **bigram** model.
Looking back two is a **trigram** model. That assumption is wrong, obviously, but it makes the
counting possible, and it works surprisingly well.

Everything in this part is: count what follows what, then turn the counts into probabilities.

---

## Step 1. Cut the text into sentences

Our tokenizer works on strings. We need sentences to feed it, because sentences are what get the
start and end markers.

```python
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
print(hobbit_sentences[5])
```

```
number of sentences: 4929
```

The rule is "split after a full stop, question mark or exclamation mark, when followed by a
space". It is rough. It will break on `Mr. Baggins`. That is fine for now, and it is on the list
of things to improve later.

---

## Step 2. Turn sentences into padded number lists

```python
sequences = []
for sentence in hobbit_sentences:
    numbers = encode(tokenizer, sentence)
    padded = pad(numbers, 2)
    sequences.append(padded)

print("how many sequences:", len(sequences))
print()
print("one sentence:", "In a hole in the ground there lived a hobbit.")
example = pad(encode(tokenizer, "In a hole in the ground there lived a hobbit."), 2)
print("as numbers:  ", example)

pieces = []
for number in example:
    pieces.append(tokenizer.id_to_token(number))
print("as pieces:   ", pieces)
```

```
as numbers:   [1, 330, 60, 1253, 97, 96, 718, 229, 1326, 60, 501, 16, 2]
as pieces:    ['<s>', 'In', 'a', 'hole', 'in', 'the', 'ground', 'there', 'lived', 'a', 'hobbit', '.', '</s>']
```

---

## Step 3. Count what follows what

This is the whole of "training" a bigram model. Walk along each sequence and, for every
neighbouring pair, record that the second thing followed the first.

We keep two records:

- `counts` says how many times each thing followed each context.
- `totals` says how many times each context appeared at all.

```python
from collections import defaultdict, Counter

counts = defaultdict(Counter)
totals = Counter()

for sequence in sequences:
    for i in range(len(sequence) - 1):
        context = sequence[i]
        next_number = sequence[i + 1]

        counts[context][next_number] = counts[context][next_number] + 1
        totals[context] = totals[context] + 1

print("contexts we have seen:", len(counts))
```

Now look at what the model learned about the word `the`:

```python
the_id = tokenizer.token_to_id("the")

print("'the' has number", the_id)
print("it appeared as a context", totals[the_id], "times")
print()

for next_number, how_many in counts[the_id].most_common(6):
    piece = tokenizer.id_to_token(next_number)
    print("   ", piece, how_many)
```

```
'the' has number 96
it appeared as a context 5659 times

    dwarves 171
    Mountain 111
    hobbit 96
    goblins 69
    door 63
    forest 61
```

That is the model. Nothing more mysterious than a tally.

`defaultdict(Counter)` just means "a dictionary where any key you ask for starts out as an empty
tally". It saves you writing `if context not in counts: counts[context] = Counter()` every time.

---

## Step 4. Counts become probabilities

To get the probability of a word given its context, divide.

```python
dwarves_id = tokenizer.token_to_id("dwarves")

top = counts[the_id][dwarves_id]
bottom = totals[the_id]

print("count of 'the' then 'dwarves':", top)
print("count of 'the' as a context:  ", bottom)
print("probability:", top / bottom)
```

```
count of 'the' then 'dwarves': 171
count of 'the' as a context:   5659
probability: 0.030217...
```

So after the word `the`, about 3 times in 100 the next piece is `dwarves`. In a book about
dwarves, that is believable.

**One thing to be careful about.** The bottom number must be how often `the` appeared *as a
context*, which is what `totals` holds. It is tempting to use "how many times the word `the`
appeared in the book" instead. Those are almost the same but not quite, because a word at the
very end of a sentence appears in the book without ever being a context. If you use the wrong
one your probabilities will not add up to 1. Building `totals` while you count, as above, avoids
the problem entirely.

---

## Step 5. The zero problem, and add-k

Try a pair the book never contains:

```python
professor_id = tokenizer.token_to_id("Professor")

print("count of 'the' then 'Professor':", counts[the_id][professor_id])
print("probability:", counts[the_id][professor_id] / totals[the_id])
```

```
count of 'the' then 'Professor': 0
probability: 0.0
```

Zero. And this is not a harmless zero. Remember that the probability of a whole sentence is
every word's probability **multiplied together**. One zero makes the entire sentence impossible,
no matter how ordinary the rest of it is.

That matters enormously for Part 4. `the Professor` never occurs in the Hobbit but occurs
constantly in The Lost World. If a Tolkien model says a Doyle passage is flatly impossible
rather than merely unlikely, we cannot compare the two models sensibly.

The fix is **add-k smoothing**: pretend you saw everything `k` more times than you did.

```
                        count(context, word) + k
P(word | context) = ------------------------------------
                      total(context) + k * V
```

`V` is the number of things that could come next. The bottom has `k * V` because we added `k` to
every one of the `V` possible next words, so the total grew by `k * V`.

Try it:

```python
V = tokenizer.get_vocab_size() - 1        # every number can be predicted except <s>

for k in [1.0, 0.1, 0.01]:
    unseen = (0 + k) / (totals[the_id] + k * V)
    seen = (171 + k) / (totals[the_id] + k * V)
    print("k =", k)
    print("    'the Professor' (never seen):", unseen)
    print("    'the dwarves'   (seen 171x) :", seen)
```

```
k = 1.0
    'the Professor' (never seen): 0.00010354
    'the dwarves'   (seen 171x) : 0.017809
k = 0.1
    'the Professor' (never seen): 0.00001650
    'the dwarves'   (seen 171x) : 0.028239
k = 0.01
    'the Professor' (never seen): 0.00000175
    'the dwarves'   (seen 171x) : 0.030007
```

Look at what `k = 1` costs. The true, unsmoothed probability of `the dwarves` was 0.0302, and
add-1 knocks it down to 0.0178. That is a 41% cut to something we observed 171 times, all to
make room for thousands of things we never observed once. Add-1 is the textbook default and it
is far too generous. Smaller `k` is gentler. Choosing `k` is one of the things the assignment
asks you to explore.

**Why `V` is the vocabulary size minus one:** every number in the vocabulary could be the next
thing, except `<s>`, which only ever appears at the start and is never predicted.

---

## Step 6. Work in logs

One more change and we are done.

A sentence's probability is many small numbers multiplied together. Multiply 30 numbers around
0.01 and you get 10 to the power of minus 60. Computers run out of precision and the answer
becomes zero.

Logarithms turn multiplication into addition, which does not have that problem.

```python
import math

for p in [0.5, 0.03, 0.0001]:
    print("p =", p, "   -log(p) =", -math.log(p))
```

```
p = 0.5      -log(p) = 0.693...
p = 0.03     -log(p) = 3.506...
p = 0.0001   -log(p) = 9.210...
```

We use the negative log, so the numbers come out positive. The base is a free choice, and the
textbook says so outright: any base works as long as you stay consistent. We use the natural
log, `math.log`, whose unit is called the **nat**. Read the result as **surprise**: a likely word scores low, a surprising word scores high. The
score for a whole sentence is the sum of the surprises of its words.

Now score a real sentence:

```python
sentence = "In a hole in the ground there lived a hobbit."
sequence = pad(encode(tokenizer, sentence), 2)

k = 0.1
total_surprise = 0.0

for i in range(len(sequence) - 1):
    context = sequence[i]
    word = sequence[i + 1]

    top = counts[context][word] + k
    bottom = totals[context] + k * V
    probability = top / bottom

    total_surprise = total_surprise + (-math.log(probability))

print("total surprise, in nats:", total_surprise)
print("number of predictions:  ", len(sequence) - 1)
print("average per prediction: ", total_surprise / (len(sequence) - 1))
```

Try the same thing with a sentence from The Lost World and compare the averages. The Tolkien
model should be more surprised by it. That is Part 4 in miniature, and you have just done it.

---

## Step 7. Move it into the scaffold

Open `src/ngram.py`. It has three short pieces missing, which are steps 3, 5 and 6.

Then show me the file.
