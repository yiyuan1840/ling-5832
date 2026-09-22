# Assignment 2: Language Models and Author Identification

LING 5832 · Yiyuan Jia · September 22, 2026


## Where everything is

All code and the full report are in a public GitHub repository:

**<https://github.com/yiyuan1840/ling-5832/tree/main/assignments/A2>**

| deliverable | location |
|---|---|
| Report | [assignments/A2/assignment2.pdf](https://github.com/yiyuan1840/ling-5832/blob/main/assignments/A2/assignment2.pdf) |
| Tokenizer | [src/tokenizer.py](https://github.com/yiyuan1840/ling-5832/blob/main/assignments/A2/src/tokenizer.py) |
| N gram models, add k smoothing, perplexity | [src/ngram.py](https://github.com/yiyuan1840/ling-5832/blob/main/assignments/A2/src/ngram.py) |
| Author identification | [src/authorid.py](https://github.com/yiyuan1840/ling-5832/blob/main/assignments/A2/src/authorid.py) |
| Corpus loading and splitting | [src/corpus.py](https://github.com/yiyuan1840/ling-5832/blob/main/assignments/A2/src/corpus.py) |
| Everything run end to end | [main.py](https://github.com/yiyuan1840/ling-5832/blob/main/assignments/A2/main.py) |
| The experiments behind every table in the report | [src/experiments.py](https://github.com/yiyuan1840/ling-5832/blob/main/assignments/A2/src/experiments.py) |
| Author ID labels for the test set | to follow once the test set is released |


## Running it

The project uses [uv](https://docs.astral.sh/uv/). From `assignments/A2`:

```
uv run main.py              the whole pipeline
uv run src/experiments.py   every number in the report
```


## Summary of results

A byte pair encoding tokenizer built on the Hugging Face `tokenizers` library, bigram and
trigram models with add k smoothing, perplexity, and an author identification classifier
trained on *The Hobbit* and *The Lost World*.

On held out 50 word passages the classifier is correct on 255 of 255. It still reaches 97.6%
after removing the two easiest cues, the differing quote characters between the two editions
and the character names, which is the more meaningful figure.

Three findings are discussed in the report: add 1 smoothing is nearly three times worse than
k = 0.01; the trigram model loses to the bigram because it meets a context it has never seen
29.7% of the time against 0.2%; and per token perplexity is not comparable across vocabulary
sizes, which makes the apparently best vocabulary the wrong choice.

Section 8 of the report describes which code I wrote and which was scaffolded by Claude
(Anthropic), as the deliverable requires.
