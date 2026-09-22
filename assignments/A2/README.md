# A2: Language Models and Author Identification

A byte pair encoding tokenizer, bigram and trigram language models with add k smoothing,
perplexity, and a classifier that guesses which of two authors wrote a passage. Trained on
*The Hobbit* and *The Lost World*.

**Report: [`assignment2.pdf`](assignment2.pdf)** (source: [`report/assignment2.md`](report/assignment2.md))

## Results

On held out 50 word passages the classifier is correct on **255 of 255**, and still reaches
**97.6%** after removing the two easiest cues, the differing quote characters and the character
names.

| | |
|---|---|
| best perplexity | 234.3, bigram, k = 0.01, vocabulary 4,000 |
| trigram perplexity | 891.8, worse, because 29.7% of its contexts are unseen against 0.2% |
| add 1 smoothing | 678.1, nearly three times worse than k = 0.01 |

## Running it

```
uv run main.py              # the whole pipeline, prints corpus, perplexity, accuracy
uv run src/experiments.py   # every table in the report
uv run render_report.py     # report/assignment2.md -> assignment2.pdf
```

## Layout

```
main.py                    runs the four parts in order
render_report.py           builds the PDF from the Markdown

src/tokenizer.py           text to numbers, wraps the Hugging Face BPE
src/ngram.py               counts, add k smoothing, perplexity
src/authorid.py            score a passage against one author's model
src/corpus.py              loading, sentence splitting, train and test split
src/experiments.py         regenerates every number in the report

data/                      the two novels as supplied
notes/                     the tutorial written while building each part
report/assignment2.md      report source
trial/                     scratch files, where each part was written by hand first
reference/                 an earlier, heavier attempt; nothing imports it
```

The assignment text is in [`assignment2-prompt.md`](assignment2-prompt.md).

## Status

All four parts are built and verified. What remains:

- final pass on the report
- author labels for the instructor's test set, once it is released

## Known weaknesses

Listed in section 7 of the report, and worth knowing before reading the numbers. The test set is
the tail of each book rather than a sample throughout, so it partly measures how the models
handle the ending. The sentence splitter breaks on abbreviations such as *Mr. Baggins*. The
scanned Tolkien text contains optical recognition damage that the Doyle text does not, which is
one more small cue the classifier could be using.

## Provenance

The functions that do the work were written by hand: `train_tokenizer` and `pad`, `fit`, `prob`,
`neg_log_prob`, `cross_entropy` and `perplexity`, `score`, and `split_sentences`. The scaffolding
around them, the pipeline wiring, the experiment script and the tutorial notes were written by
Claude (Anthropic) working as a tutor. Section 8 of the report gives the full split, as the
assignment requires.

No pretrained tokenizer and no existing n gram library is used. The only outside package needed
to run the models is `tokenizers`.
