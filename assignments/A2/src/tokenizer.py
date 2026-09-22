# tokenizer.py -- turn text into numbers.
#
# The reasoning behind all of this is in notes/tutorial.md. This file is just the result.
#
# Skeleton by Claude. The two TODO bodies are by Yiyuan Jia.

from tokenizers import Tokenizer, models, trainers, pre_tokenizers

SPECIALS = ["<unk>", "<s>", "</s>"]
UNK, BOS, EOS = 0, 1, 2


def train_tokenizer(texts, vocab_size=4000):
    """Build a vocabulary from some text. Returns a trained Tokenizer.

    TODO 1. This is tutorial steps 2 and 3, with vocab_size passed in.
    """
    splitter = pre_tokenizers.Sequence([
    pre_tokenizers.WhitespaceSplit(),
    pre_tokenizers.Punctuation(),
    # pre_tokenizers.Whitespace(),
    ])
    tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = splitter

    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=SPECIALS,
        show_progress=False,
    )

    tokenizer.train_from_iterator(texts, trainer)
    return tokenizer


def encode(tokenizer, text):
    """Turn a string into a list of numbers."""
    return tokenizer.encode(text).ids


def pad(ids, n):
    """Add sentence start and end markers.

    n is 2 for a bigram model, 3 for a trigram model.
    Result is (n - 1) copies of BOS, then ids, then one EOS.
    """
    if n == 2:
        return [BOS] + ids + [EOS]
    elif n == 3:
        return [BOS, BOS] + ids + [EOS]
    else:
        raise ValueError("n must be 2 or 3, got " + str(n))
