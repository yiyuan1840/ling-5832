# splits.py -- deterministic train/dev/test splitting and evaluation-passage construction.
#
# Provenance: written by Claude (Anthropic) as scaffolding; reviewed by Yiyuan Jia.
#
# The unit is the paragraph and the unit of holdout is a contiguous BLOCK of paragraphs.
# Two simpler schemes were rejected:
#   - a single 20% tail holds out only the last chapters, so it measures whether the model
#     handles the ending rather than whether it identifies the author;
#   - interleaving individual sentences puts every held-out sentence's immediate neighbours
#     into training, which leaks most of its rare n-grams and inflates every number.
# Forty blocks with a guard band is the compromise: each held-out block is continuous prose
# long enough to read naturally, and six of them sample the whole book.

import random

SEED = 5832
N_BLOCKS = 40
RATIO = {"train": 28, "dev": 6, "test": 6}
GUARD = 1  # paragraphs dropped on each side of every held-out block


def _block_bounds(word_counts, n_blocks):
    """Cut a paragraph sequence into n_blocks contiguous runs of roughly equal word count.

    Equal word count rather than equal paragraph count, because paragraph length varies by an
    order of magnitude (one-line dialogue turns against descriptive passages) and equal-count
    blocks would make the held-out fraction unpredictable.
    """
    total = sum(word_counts)
    target = total / n_blocks
    bounds, start, acc, made = [], 0, 0, 0
    for i, w in enumerate(word_counts):
        acc += w
        remaining_blocks = n_blocks - made - 1
        if acc >= target * (made + 1) - target / 2 and remaining_blocks > 0 \
                and len(word_counts) - (i + 1) >= remaining_blocks:
            bounds.append((start, i + 1))
            start, made = i + 1, made + 1
    bounds.append((start, len(word_counts)))
    return bounds


def assign_blocks(paragraphs, seed=SEED):
    """Assign contiguous paragraph blocks to train, dev and test.

    Returns a dict with, per split, the list of (start, end) paragraph ranges after the guard
    band has been removed, plus the set of guard paragraph indices that belong to no split.
    """
    counts = [len(p.split()) for p in paragraphs]
    bounds = _block_bounds(counts, N_BLOCKS)
    order = list(range(len(bounds)))
    random.Random(seed).shuffle(order)

    labels = {}
    i = 0
    for split, n in RATIO.items():
        for b in order[i:i + n]:
            labels[b] = split
        i += n

    guards = set()
    for b, (s, e) in enumerate(bounds):
        if labels[b] in ("dev", "test"):
            guards |= set(range(max(0, s - GUARD), s))
            guards |= set(range(e, min(len(paragraphs), e + GUARD)))

    out = {"train": [], "dev": [], "test": [], "guards": sorted(guards)}
    for b, (s, e) in enumerate(bounds):
        keep = [i for i in range(s, e) if i not in guards]
        if keep:
            out[labels[b]].append((keep[0], keep[-1] + 1))
    return out


def paragraph_indices(ranges):
    """Flatten a list of (start, end) ranges into a sorted list of paragraph indices."""
    return [i for s, e in ranges for i in range(s, e)]


def split_text(paragraphs, ranges):
    """Return the paragraphs of one split, in reading order."""
    return [paragraphs[i] for i in paragraph_indices(ranges)]


def build_passages(sentences, para_of, ranges, target_words):
    """Build non-overlapping evaluation passages of about target_words each.

    Sentences are accumulated in reading order until the word target is reached. A passage
    never crosses a block boundary, so no passage mixes text from two different regions of the
    book, and a trailing fragment shorter than half the target is discarded rather than
    emitted as an unrepresentatively short passage.
    """
    keep = set(paragraph_indices(ranges))
    blocks = {}
    for s, e in ranges:
        for i in range(s, e):
            blocks[i] = (s, e)

    passages, cur, cur_block = [], [], None
    for sent, p in zip(sentences, para_of):
        if p not in keep:
            continue
        blk = blocks[p]
        if cur and blk != cur_block:
            _flush(passages, cur, cur_block, target_words)
            cur = []
        cur_block = blk
        cur.append(sent)
        if sum(len(s.split()) for s in cur) >= target_words:
            _flush(passages, cur, cur_block, target_words, force=True)
            cur = []
    if cur:
        _flush(passages, cur, cur_block, target_words)
    return passages


def _flush(passages, cur, block, target_words, force=False):
    text = " ".join(cur)
    n = len(text.split())
    if force or n >= target_words / 2:
        passages.append({"text": text, "n_words": n, "n_sentences": len(cur), "block": block})
