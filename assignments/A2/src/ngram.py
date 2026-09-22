# ngram.py -- count word patterns and turn them into probabilities.
#
# The reasoning is in notes/part2-ngram.md. This file is the result.
#
# Skeleton by Claude. The three TODO bodies are by Yiyuan Jia.

import math
from collections import defaultdict, Counter


class NgramModel:
    """A bigram (n=2) or trigram (n=3) language model with add-k smoothing.

    One difference from the tutorial. There, a bigram context was a single number.
    Here a context is always a TUPLE of (n - 1) numbers, so that the same code works
    for both sizes. For a bigram the tuple has one number in it, like (96,).
    For a trigram it has two, like (96, 171).
    """

    def __init__(self, n, vocab_size, k=1.0):
        self.n = n
        self.vocab_size = vocab_size
        self.k = k

        self.counts = defaultdict(Counter)   # context tuple -> tally of what followed
        self.totals = Counter()              # context tuple -> how often it appeared

    def fit(self, sequences):
        """Count patterns in a list of padded number lists.

        TODO 1. Tutorial step 3.

        For each sequence, look at every position where a prediction happens.
        Predictions start at index (n - 1), because earlier positions are the
        start markers that give the first real word its context.

            for i in range(self.n - 1, len(sequence)):
                context = tuple(sequence[i - (self.n - 1):i])
                word = sequence[i]

        then add one to self.counts[context][word] and one to self.totals[context].
        """
        for sequence in sequences:
            for i in range(self.n - 1, len(sequence)):
                context = tuple(sequence[i - (self.n - 1):i])
                word = sequence[i]

                self.counts[context][word] = self.counts[context][word] + 1
                self.totals[context] = self.totals[context] + 1

    def prob(self, context, word):
        """Probability of `word` following `context`, with add-k smoothing.

        context is a tuple of (n - 1) numbers.

        TODO 2. Tutorial step 5.

            V = self.vocab_size - 1
            top    = how many times word followed context, plus k
            bottom = how many times context appeared, plus k * V
            return top / bottom

        Note this works even for a context never seen. Both counts are then zero,
        so you get k / (k * V) = 1/V, meaning "no idea, every word equally likely".
        That is the correct behaviour, not a special case to handle.
        """
        V = self.vocab_size - 1
        top = self.counts[context][word] + self.k
        bottom = self.totals[context] + self.k * V
        return top / bottom

    def neg_log_prob(self, sequence):
        """Total surprise of one padded sequence, in nats (natural log).

        Returns two things: the total, and how many predictions were made.
        We need both so that perplexity can average over a whole test set later.

        TODO 3. Tutorial step 6.

        Walk the same positions as fit(). For each one work out the probability
        with self.prob(), take -math.log of it, and add it to a running total.
        """
        total = 0.0
        num_predictions = 0
        for i in range(self.n - 1, len(sequence)):
            context = tuple(sequence[i - (self.n - 1):i])
            word = sequence[i]
            prob = self.prob(context, word)
            total -= math.log(prob)
            num_predictions += 1
        return total, num_predictions

    def cross_entropy(self, sequences):
        """Average surprise per prediction over a list of padded sequences, in nats.

        Adding up first and dividing once at the end matters: averaging each sentence
        separately and then averaging those averages would give a three-word sentence
        the same weight as a forty-word one.

        Part 4 uses this directly to score a single passage, by passing a list with one
        sequence in it.
        """
        total = 0.0
        number_of_predictions = 0

        for sequence in sequences:
            surprise, count = self.neg_log_prob(sequence)
            total = total + surprise
            number_of_predictions = number_of_predictions + count

        return total / number_of_predictions

    def perplexity(self, sequences):
        """How surprised this model is by a list of padded sequences.

        Perplexity is the model's average number of choices. A value of 250 means it is
        as uncertain as if it were picking at random from 250 equally likely options.
        Lower is better.

        The exponential must match the logarithm used in neg_log_prob. We use the
        natural log there, so we use math.exp here. Mixing them (log base 2 with
        math.exp, or natural log with 2 ** x) gives a wrong number silently.
        """
        return math.exp(self.cross_entropy(sequences))
