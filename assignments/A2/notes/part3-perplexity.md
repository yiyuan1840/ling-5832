# Part 3: measure how good a model is

Short part. One new idea, one new function, and one result that should surprise you.

---

## Step 1. You cannot test on what you trained on

So far you have trained on the whole Hobbit and scored sentences from the whole Hobbit. That
tells you nothing. The model has already seen those exact sentences, so of course it finds them
unsurprising. It is marking its own homework.

To measure anything real, hold some text back.

```python
all_sentences = split_sentences(hobbit)

cut = int(len(all_sentences) * 0.9)
train_sentences = all_sentences[:cut]
test_sentences = all_sentences[cut:]

print("train:", len(train_sentences))
print("test: ", len(test_sentences))
```

```
train: 4436
test:  493
```

The last 10 percent of the book is now held back. This is a crude split, and it has a real flaw:
the end of The Hobbit is a battle, with different characters and different vocabulary from the
beginning. So we are partly measuring "can the model handle the ending". We will fix it later.

---

## Step 2. Average the surprise

Your `neg_log_prob` gives the total surprise of one sentence. Longer sentences have more
surprise simply by being longer, so a total is not comparable between texts. Divide by the
number of predictions to get an average.

```python
def average_surprise(model, sequences):
    total = 0.0
    number_of_predictions = 0

    for sequence in sequences:
        surprise, count = model.neg_log_prob(sequence)
        total = total + surprise
        number_of_predictions = number_of_predictions + count

    return total / number_of_predictions
```

Note this adds up everything first and divides once at the end. That is deliberate. Averaging
each sentence and then averaging the averages would weight a three-word sentence the same as a
forty-word one.

This average has a name: **cross entropy**. Its unit depends on the logarithm you used. We use
the natural log, so the unit is called the **nat**. The textbook says outright that the base is
a free choice, as long as you are consistent about it.

---

## Step 3. Perplexity

Cross entropy is hard to read. Is 5.49 nats good? Perplexity converts it into something you can
picture:

```
perplexity = math.exp(cross_entropy)
```

**What it means.** A perplexity of 243 means the model is as uncertain as if it were choosing
uniformly at random between 243 equally likely options at every step. So it is the model's
*average number of choices*. Lower is better. A perfect model has perplexity 1, meaning it
always knows exactly what comes next.

The exponential has to undo the logarithm you took, so natural log pairs with `math.exp`. If
you had used log base 2 you would raise 2 to the power instead. Either pairing gives the SAME
perplexity, because the base cancels out. Mixing them does not, and fails silently.

```python
def perplexity(model, sequences):
    return math.exp(average_surprise(model, sequences))
```

Keep these as two separate functions rather than one. Part 4 needs the averaging step on its
own, without the exponential, so splitting them here saves writing the same loop twice. In the
scaffold they are called `cross_entropy` and `perplexity`.

---

## Step 4. Check it behaves sensibly

Before trusting any number, check the model does better on text it has seen than on text it has
not:

```python
train_sequences = [pad(encode(tokenizer, s), 2) for s in train_sentences]
test_sequences = [pad(encode(tokenizer, s), 2) for s in test_sentences]

model = NgramModel(2, tokenizer.get_vocab_size(), k=0.01)
model.fit(train_sequences)

print("perplexity on training text:", perplexity(model, train_sequences))
print("perplexity on held-out text:", perplexity(model, test_sequences))
```

```
perplexity on training text: 48.8
perplexity on held-out text: 243.1
```

Good. The model is much more comfortable with what it has already seen. If those two numbers
came out equal, something would be wired wrong.

---

## Step 5. Compare the models

Now the part the assignment asks for. Try both sizes and several values of `k`.

```python
for n in [2, 3]:
    train_sequences = [pad(encode(tokenizer, s), n) for s in train_sentences]
    test_sequences = [pad(encode(tokenizer, s), n) for s in test_sentences]

    for k in [1.0, 0.1, 0.01, 0.001]:
        model = NgramModel(n, tokenizer.get_vocab_size(), k=k)
        model.fit(train_sequences)
        print(n, k, perplexity(model, test_sequences))
```

| n | k | perplexity |
|---|---|---|
| 2 | 1.0 | 679.2 |
| 2 | 0.1 | 312.2 |
| 2 | **0.01** | **243.1** |
| 2 | 0.001 | 324.8 |
| 3 | 1.0 | 2215.5 |
| 3 | 0.1 | 1328.4 |
| 3 | **0.01** | **913.5** |
| 3 | 0.001 | 930.3 |

Two things to take from this.

**Add-1 is bad.** It is the textbook default, and at `k = 1` the bigram scores 679. Drop to
`k = 0.01` and it scores 243, nearly three times better. Go further to 0.001 and it gets worse
again. There is a sweet spot, and finding it is exactly the exploration the assignment wants.

**The trigram is worse than the bigram.** At every single value of `k`. That should bother you,
because a trigram model knows strictly more than a bigram model. It looks back further. It
should win.

---

## Step 6. Why the trigram loses

Count how often the model meets a context it has never seen before:

```python
unseen = 0
total = 0

for sequence in test_sequences:
    for i in range(n - 1, len(sequence)):
        context = tuple(sequence[i - (n - 1):i])
        if model.totals[context] == 0:
            unseen = unseen + 1
        total = total + 1

print(unseen, "of", total, "contexts were never seen in training")
```

```
n = 2:    53 of 12268    0.4%
n = 3:  3712 of 12268   30.3%
```

There it is. A bigram model almost always recognises the single word it is looking back at. A
trigram model has to have seen that exact *pair* of words before, and nearly a third of the
time it has not. When that happens your `prob` falls back to "every word equally likely", which
for a 4000 word vocabulary is a very bad guess made 30 percent of the time.

This is not a bug in your code. It is the central problem with n-gram models, and it gets worse
the further back you look. The Hobbit has about 96,000 words. There are 4000 times 4000, or 16
million, possible word pairs. Almost all of them never occur, so almost any pair you meet is new.

Real systems fix this by **backing off**: if you have never seen this pair, fall back on what
the single previous word tells you, rather than giving up entirely. That is beyond what the
assignment asks for, but it is worth knowing that the fix exists and that this table is the
reason it was invented.

---

## Step 7. Move it into the scaffold

Add one method to `NgramModel` in `src/ngram.py`. There is a TODO waiting for it.

Then show me, and we go to Part 4, where these two models finally get used to guess an author.
