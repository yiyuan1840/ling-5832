# Primer, Part 1: Tokenization

Notes written to go with the code, covering the concepts from the ground up. Read this before
opening `src/bpe.py`. Lecture 3 is the course reference; this fills in what the slides compress
and connects it to the three functions you are about to write.

## 1. The problem tokenization solves

A language model assigns probabilities to sequences. Before you can do that you have to decide
what the sequence is a sequence *of*. That choice is not obvious, and it is not free.

**Words are the intuitive unit and they fail.** Lecture 2 gives the reason in the form of Heaps'
law: vocabulary grows without bound as corpus grows, roughly as a power of corpus size with the
exponent between 0.6 and 0.7. There is no corpus large enough to contain every word you will
later meet. Names, numbers, misspellings, compounds and borrowings keep arriving. So a
word-level model must decide what to do when it meets something new, and the honest answer is
that it cannot say anything useful about it at all. In our corpus this bites hard: the two books
share only 3,362 word types out of 6,171 and 7,824. A word-level model trained on one book meets
unfamiliar words constantly.

**Characters are the obvious fix and they fail differently.** A 26-letter vocabulary never meets
anything new. But it throws away everything the word level knew for free. The model now has to
learn that `t`, `h`, `e` in sequence is a common thing, and with a bigram or trigram window it
cannot even see a whole short word, let alone the relationship between two words. Sequences get
about five times longer, so a trigram of characters covers less context than a bigram of words.

**Subword units are the compromise.** Keep frequent words whole, break rare words into pieces
that are themselves frequent. Now the vocabulary is a fixed size you choose, nothing is ever
truly unseen because you can always fall back on smaller pieces, and common words still get to
be single units. That is the whole idea. Byte-pair encoding is one way to find those pieces.

## 2. How byte-pair encoding works

The algorithm is almost embarrassingly simple, which is part of why it won. It comes from a 1994
data-compression paper by Gage and was brought into NLP by Sennrich, Haddow and Birch in 2016.

Start with a vocabulary of every individual character in your corpus. Then repeat: count every
adjacent pair of current vocabulary items, find the most frequent pair, and add the merged pair
to the vocabulary as a new item. Stop when the vocabulary reaches the size you asked for.

Run `python src/toy_bpe.py` and read section 1 of the output. It trains on the lecture's toy
corpus, `set new new renew reset renew`, and prints the merges in the order learned:

```
merge  1:  ('e', 'w')      -> 'ew'
merge  2:  ('n', 'ew')     -> 'new'
merge  3:  ('r', 'e')      -> 're'
merge  4:  ('_', 're')     -> '_re'
...
```

Two things are worth noticing.

**Merges compound.** Merge 2 uses the output of merge 1. The vocabulary grows by building larger
pieces out of pieces it already has, which is why a vocabulary of a few thousand items can
represent words of any length.

**The order is the model.** Encoding a new string means applying the merges greedily in the
order they were learned. That is why the demo can encode `rereset`, a word the corpus never
contained, as `_re` + `re` + `set`. Nothing is out of vocabulary as long as the individual
characters are in the vocabulary. Hold on to that sentence, because Part 4 of this primer shows
it is not quite true of the library we are using.

## 3. Pre-tokenization, which decides more than BPE does

Before BPE ever runs, something has to decide which character sequences are even *candidates*
for merging. That is pre-tokenization, and it matters more than people expect.

If you hand BPE the raw character stream, it will happily learn a merge across a space, or
across a quote mark, because those are just characters and some pairs of them are frequent. So
every implementation first splits the text into pre-tokens and forbids merges across the
boundaries. What counts as a boundary is a design decision.

Section 2 of the demo output shows four strategies on one sentence. The differences are not
cosmetic. Look at what happens to `morning!"`:

| strategy | result |
|---|---|
| whitespace | `morning`, `!"` |
| whitespace and punctuation | `morning`, `!`, `"` |
| byte level | `Ġmorning`, `!"` |
| metaspace | `▁morning!"` |

Now section 3 of the same output, run on the real corpus at a vocabulary of 8000:

```
wsp :    0 vocabulary items contain a punctuation character
meta: 1463 vocabulary items contain a punctuation character
        examples: ['MAN:', 'able.', 'ably.', 'ace,', 'ace.', 'ack!', ...]
```

This is the single most useful thing in Part 1, and it connects directly to the problem this
whole assignment has. Our two books use different quote characters, different dash conventions
and different sentence spacing, because they were typeset by different people. A model can
identify the author from those marks alone, which tells you nothing about Tolkien or Conan
Doyle. Normalization is one defence. But notice what metaspace does: it fuses the punctuation
into the *vocabulary itself*, so entries like `ace,` and `ack!` exist as units. Splitting
punctuation off first makes that structurally impossible. Pre-tokenization is the first line of
defence against the leak, before normalization gets a chance.

That is the reasoning behind TODO 1.

## 4. The out-of-vocabulary claim, and where it goes wrong

Lecture 3, slide 19, says that with BPE there are no out-of-vocabulary words, because the
vocabulary starts with UTF-8. Every string decomposes into known pieces, worst case single
characters.

That is true of the algorithm. It is not true of the default configuration of the library we are
required to use.

The reason: the initial alphabet is built from the characters the *training corpus* contained.
Our corpus is two English novels. It contains no `é`, no `你`, no snowman. So those characters
are not in the initial alphabet, there is no merge path to them, and they come back as `<unk>`.

Run `python src/toy_bpe.py --check-unk` once you have finished TODO 2 and you will see both
branches measured:

```
initial_alphabet seeded = False ->  <unk> tokens = 11   exact round trip = False
initial_alphabet seeded = True  ->  <unk> tokens =  0   exact round trip = True
```

Passing `initial_alphabet=pre_tokenizers.ByteLevel.alphabet()` seeds all 256 byte characters, so
every possible byte has a vocabulary entry and `<unk>` becomes unreachable. That is the
configuration that actually delivers what the lecture describes.

Worth being precise about what this buys, because it is not free. Under byte-level encoding an
unseen word becomes a long run of byte tokens, and a trigram model has never seen any of those
contexts, so it will fall back to a uniform guess several times in a row. Under a word-level
pre-tokenizer the same word becomes one `<unk>` whose probability at least reflects how often
unknown things happened during training. Which is better is an empirical question, and the sweep
in Part 3 answers it. This is exactly the kind of trade-off the assignment means when it invites
you to explore pre-tokenization strategies.

## 5. Vocabulary size

You choose it. The trade-off, from lecture 3 slide 23: smaller vocabularies need less memory and
fewer parameters, but produce longer sequences, because rare words get chopped into more pieces.

Measured on our corpus of about 172,000 words:

| vocabulary | tokens per word (wsp) | tokens per word (byte) |
|---|---|---|
| 500 | 2.01 | 2.71 |
| 1,000 | 1.72 | 2.14 |
| 2,000 | 1.51 | 1.83 |
| 4,000 | 1.36 | 1.63 |
| 8,000 | 1.26 | 1.50 |
| 16,000 | 1.20 | 1.41 |

There is a trap here that Part 3 will make concrete. Perplexity per *token* is not comparable
across these rows. A smaller vocabulary means more events per word, and each event is
individually easier, so its per-token perplexity looks better while the model is actually worse
at predicting the text. Whenever vocabulary size varies, compare per *word*.

Also always read the realized vocabulary size back from the tokenizer rather than trusting what
you requested. If the corpus cannot support the size you asked for, the trainer quietly gives
you fewer.

## 6. Special tokens

Three, and they take indices 0, 1 and 2 because the trainer assigns special tokens first.

`<unk>` is the fallback for anything unrepresentable. `<s>` and `</s>` mark sentence boundaries.
Lecture 4 slide 31 writes the worked sentence probability as

```
P(<s> I want english food </s>)
  = P(I|<s>) * P(want|I) * P(english|want) * P(food|english) * P(</s>|food)
```

Read that carefully, because TODO 3 is exactly this and Part 2 depends on getting it right.

`<s>` gives the first real word a context. Without it the model cannot represent how likely a
sentence is to begin with a given word, which is real information. A trigram model conditions on
two previous tokens, so it needs two of them.

`</s>` appears as a *prediction*, the last factor. This is the part people skip, and skipping it
quietly breaks the model. Without a way to predict that the sentence ends, the model is not a
probability distribution over strings of varying length. It would assign probability mass to
every possible continuation forever, the numbers would not sum to one, and perplexity would not
be well defined. `</s>` is predicted but never used as a context, because nothing follows it.

The consequence to remember: with `(n-1)` opening symbols and one closing symbol, a bigram and a
trigram model score the *same number of events* on the same text, namely one per real token plus
one for the end. That equality is the only reason their perplexities can be put in the same
table. The test file asserts it.

## 7. What the code does

`prep.py` and `splits.py` are written for you; they are plumbing, and the reasoning behind their
choices is in the comments and in the plan. `prepare_data.py` prints the evidence: which
typographic channels are disjoint between the two books, and that they close after cleaning
while intraword hyphens survive.

Your three functions in `bpe.py`:

1. `_make_pretokenizer`, the `wsp` branch. One expression. Section 3 above is the why.
2. `_make_trainer`. Four arguments, one of which is conditional. Section 4 is the why.
3. `encode_sentences`. Two lines. Section 6 is the why, and it is the most conceptually
   load-bearing of the three.

Then `python src/test_part1.py`.
