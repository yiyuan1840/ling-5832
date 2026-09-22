# Part 1: turn text into numbers

Type each step into `trial/spaghetti.py` and run it with `uv run trial\spaghetti.py`.
Every step shows what you should see.

Five steps. At the end you will have written the whole of `src/tokenizer.py`.

---

## Why we are doing this

A language model works with numbers, not letters. So we need a fixed list of text pieces, each
with a number. That list is called a **vocabulary**.

Whole words will not do, because new words keep appearing and no list is long enough. Single
letters will not do either, because too much is thrown away. So we use pieces in between: common
words stay whole, rare words get broken up.

**BPE** (byte pair encoding) is the method that decides those pieces. The assignment says to use
Hugging Face's version rather than writing our own.

---

## Step 1. Load the books

```python
from pathlib import Path

base_dir = Path(__file__).parent.parent

with open(base_dir / "data" / "hobbit.txt", encoding="utf-8") as f:
    hobbit = f.read()

with open(base_dir / "data" / "lostworld.txt", encoding="utf-8") as f:
    lostworld = f.read()

print(len(hobbit), len(lostworld))
```

```
521691 443353
```

---

## Step 2. Decide where text may be cut

Before BPE runs, something decides where it is *allowed* to cut. Compare two choices:

```python
from tokenizers import pre_tokenizers

test_text = 'He said "no!" twice.'

splitter = pre_tokenizers.WhitespaceSplit()
results = splitter.pre_tokenize_str(test_text)

pieces = []
for item in results:
    pieces.append(item[0])

print("cut at spaces only:", pieces)
```

```
cut at spaces only: ['He', 'said', '"no!"', 'twice.']
```

`"no!"` stayed in one piece, quote marks and all. That is a problem. BPE would then be free to
learn `"no!"` as a single vocabulary entry, and **our two books use different quote characters**.
A model could then tell the books apart by their punctuation instead of their writing.

So cut at spaces *and* at every punctuation mark:

```python
splitter = pre_tokenizers.Sequence([
    pre_tokenizers.WhitespaceSplit(),
    pre_tokenizers.Punctuation(),
])

results = splitter.pre_tokenize_str(test_text)

pieces = []
for item in results:
    pieces.append(item[0])

print("cut at both:", pieces)
```

```
cut at both: ['He', 'said', '"', 'no', '!', '"', 'twice', '.']
```

Keep this `splitter`. You need it next.

---

## Step 3. Build the vocabulary

This is the assignment's actual requirement for Part 1.

```python
from tokenizers import Tokenizer, models, trainers

tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
tokenizer.pre_tokenizer = splitter

trainer = trainers.BpeTrainer(
    vocab_size=500,
    special_tokens=["<unk>", "<s>", "</s>"],
    show_progress=False,
)

tokenizer.train_from_iterator([hobbit, lostworld], trainer)

print("size:", tokenizer.get_vocab_size())
print("<unk> is", tokenizer.token_to_id("<unk>"))
print("<s> is", tokenizer.token_to_id("<s>"))
print("</s> is", tokenizer.token_to_id("</s>"))
```

```
size: 500
<unk> is 0
<s> is 1
</s> is 2
```

Those three are markers we always want, and the trainer always puts them first, so their numbers
are fixed. `<unk>` means "a piece I have no number for". `<s>` and `</s>` mark where a sentence
starts and ends. You use the last two in step 5.

Both books go in together as one list, giving **one shared vocabulary**. That matters later: two
models can only be compared if they use the same numbers for the same things.

---

## Step 4. Turn a sentence into numbers

```python
encoded = tokenizer.encode("Bilbo went home.")

print(encoded.tokens)
print(encoded.ids)
```

```
['Bilbo', 'went', 'ho', 'me', '.']
[246, 433, 228, 121, 16]
```

Notice `home` came out as two pieces. With only 500 entries the vocabulary cannot afford one for
`home`. Change `vocab_size` to 4000, run again, and you get:

```
['Bilbo', 'went', 'home', '.']
```

That is the trade-off you control: a bigger vocabulary means shorter sequences, but more things
for a model to learn.

---

## Step 5. Add sentence markers

```python
numbers = tokenizer.encode("Bilbo went home.").ids

BOS = 1
EOS = 2

for_bigram = [BOS] + numbers + [EOS]
for_trigram = [BOS, BOS] + numbers + [EOS]

print("bigram: ", for_bigram)
print("trigram:", for_trigram)
```

A **bigram** model predicts each word from the one word before it. The first word of a sentence
has nothing before it, so `<s>` gives it something. A **trigram** model looks at the two words
before, so it needs two.

`</s>` at the end lets the model predict "the sentence is over". Without it the model can never
stop, and its probabilities do not add up to 1.

Now count the predictions each model makes:

```python
print(len(for_bigram) - 1)
print(len(for_trigram) - 2)
```

Both give the same number. A bigram cannot predict its first item and a trigram cannot predict
its first two, so the extra `<s>` makes them even. That matters because later we compare the two
models, and the comparison is only fair if they answered the same number of questions.

---

## Step 6. Move it into the scaffold

Open `src/tokenizer.py`. Two functions have their bodies missing.

- `train_tokenizer` is steps 2 and 3, with `vocab_size` passed in instead of typed.
- `pad` is step 5, written once for any `n`. The obvious version is fine:

```python
markers = []
for i in range(n - 1):
    markers.append(BOS)

return markers + ids + [EOS]
```

Then show me the file and we will read it together before moving on.
