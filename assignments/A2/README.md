# A2: Language Models and Author Identification

**Start here: [`notes/part4-authorid.md`](notes/part4-authorid.md).**

Work through it in `trial/spaghetti.py`, then fill in the two TODOs in
[`src/authorid.py`](src/authorid.py).

## The four parts

1. **Tokenizer** — turn text into numbers. *(done)*
2. **N-gram model** — count which words follow which, turn counts into probabilities. *(done)*
3. **Perplexity** — one number for how surprised a model is by unseen text. *(done)*
4. **Author ID** — train one model per author; the less surprised one wins. *(you are here)*

## Folders

```
notes/part1-tokenizer.md   finished
notes/part2-ngram.md       finished
notes/part3-perplexity.md  finished
notes/part4-authorid.md    the current lesson
trial/spaghetti.py         your scratch file
src/tokenizer.py           finished
src/ngram.py               finished
src/authorid.py            2 TODOs waiting
data/                      the two books
reference/                 a heavier version I wrote earlier; ignore it
```

The assignment text is in [`assignment2-prompt.md`](assignment2-prompt.md).
