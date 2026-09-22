# Part 4: guess the author

The payoff. You already did this by hand at the end of Part 2, when a Tolkien model was more
surprised by a Conan Doyle sentence than by a Tolkien one. Now we do it properly and measure it.

---

## The idea

Train **two** models. One sees only Tolkien, one sees only Conan Doyle. Given a passage by an
unknown author, ask both how surprised they are. The less surprised one is our guess.

That is the whole method. Everything below is care about doing it honestly.

---

## Step 1. One tokenizer, two models

This is the part where it is easy to go wrong, so it is worth being careful.

You need **one shared tokenizer**, trained on both authors' text together. Then two separate
models built on top of it.

```python
tokenizer = train_tokenizer([tolkien_train_text, doyle_train_text], vocab_size=4000)
V = tokenizer.get_vocab_size()

tolkien_model = NgramModel(2, V, k=0.01)
tolkien_model.fit([pad(encode(tokenizer, s), 2) for s in tolkien_train_sentences])

doyle_model = NgramModel(2, V, k=0.01)
doyle_model.fit([pad(encode(tokenizer, s), 2) for s in doyle_train_sentences])
```

**Why one tokenizer and not one per author.** If each author had their own vocabulary, the same
passage would be chopped into two different lists of numbers of two different lengths. The two
surprise scores would then be answers to two different questions, and comparing them would mean
nothing. Worse, whichever tokenizer was trained on that author's text would chop their writing
into fewer, longer pieces, so that model would look better automatically, regardless of style.

One vocabulary means both models are answering the same question about the same thing. Only
their counts differ, which is exactly what we want to compare.

---

## Step 2. Classify one passage

```python
def average_surprise(model, tokenizer, passage, n):
    sequence = pad(encode(tokenizer, passage), n)
    total, count = model.neg_log_prob(sequence)
    return total / count


passage = "Bilbo was very fond of visitors, and the dwarves sang of gold."

t = average_surprise(tolkien_model, tokenizer, passage, 2)
d = average_surprise(doyle_model, tokenizer, passage, 2)

print("Tolkien model surprise:", t)
print("Doyle model surprise:  ", d)
print("guess:", "Tolkien" if t < d else "Doyle")
```

**Why divide by the number of predictions.** Here is a subtlety worth getting right, because the
usual explanation is wrong. People say you must divide to compare passages fairly. But both
models see the *same* passage chopped the *same* way, so the count is identical for both, and
dividing cannot change which one wins.

The real reason is that it makes the **gap** between the two scores meaningful. An undivided gap
grows with passage length, so you could not compare how confident the model was on a short
passage against a long one. Divided, you can.

---

## Step 3. Measure it properly

One passage proves nothing. Hold out the last 10 percent of each book, cut it into passages, and
count how many you get right.

```python
def make_passages(sentences, words_per_passage):
    passages = []
    current = []
    for sentence in sentences:
        current.append(sentence)
        if len(" ".join(current).split()) >= words_per_passage:
            passages.append(" ".join(current))
            current = []
    return passages
```

Then loop over both authors' held-out passages, classify each, and tally.

With 50-word passages, a bigram model and `k = 0.01`, you should get:

```
accuracy 100%   tolkien 139/139   doyle 116/116
```

---

## Step 4. Be suspicious of 100 percent

A perfect score should worry you, not please you. It usually means the task is easier than you
intended, and the honest thing is to find out why before writing it up.

There are two ways this could be cheating.

**First, the Project Gutenberg licence.** The Doyle file still has its licence attached, and 16
percent of its held-out sentences are licence text, full of words like `Gutenberg` and `eBook`
that never appear in The Hobbit. Strip it out and re-measure. It turns out not to matter here,
but you could not have known that without checking.

**Second, and more interesting: what is the model actually reading?** Two things in these books
have nothing to do with writing style:

- The two editions use **different quote characters**. Tolkien's has curly quotes, Doyle's has
  straight ones. That is a fact about typesetters, not authors.
- The books use **completely different character names**. `Bilbo` against `Challenger`. A model
  that spots those is a name detector, not a style detector.

Test both by removing them and re-measuring:

| what we removed | 50-word passages | 10-word passages |
|---|---|---|
| nothing (baseline) | 100.0% | 94.4% |
| quotes and dashes normalized | 99.6% | 91.9% |
| character names masked | 99.2% | 93.5% |
| both | 97.6% | 90.2% |

**This is the result to report.** Take both crutches away and the classifier still gets 97.6
percent on 50-word passages. So there is genuine signal in how these two authors write, beyond
punctuation and proper nouns. But each crutch was worth something, and more so on short
passages, where there is less real evidence to go on.

Reporting the 100 percent alone would be true and misleading. Reporting the table is honest.

---

## Step 5. Passage length matters most

```
10 words    94.4%
25 words    98.1%
50 words   100.0%
100 words  100.0%
```

Obvious in hindsight: more text, more evidence. Worth measuring because the assignment says the
test passages will be "short" without saying how short, and this tells you how much that matters.

---

## Step 6. Which settings to use

From the same experiment:

| n | k | accuracy |
|---|---|---|
| 2 | 1.0 | 100.0% |
| 2 | 0.1 | 100.0% |
| 2 | 0.01 | 100.0% |
| 3 | 1.0 | 98.8% |
| 3 | 0.1 | 98.4% |
| 3 | 0.01 | 98.8% |

The bigram beats the trigram again, for the same reason as Part 3: the trigram meets unseen
contexts too often and falls back to guessing.

Notice something that might surprise you. `k` barely matters for classification, even though it
mattered enormously for perplexity in Part 3. That makes sense once you see it: changing `k`
shifts *both* models' scores in the same direction, and the decision only depends on which score
is lower. Perplexity cares about the absolute number, classification only about the comparison.

---

## Step 7. Move it into the scaffold

Open `src/authorid.py`. Two short functions.

Then we are done with the code and can write the report.
