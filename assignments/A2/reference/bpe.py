# bpe.py -- the project's interface to the Hugging Face BPE implementation.
#
# Provenance: scaffolded by Claude (Anthropic); the three functions marked TODO are implemented
# by Yiyuan Jia. No pretrained tokenizer is used anywhere: every vocabulary in this project is
# trained from the assignment's own two text files.
#
# This is the ONLY module that imports `tokenizers`. Everything downstream sees integer
# vocabulary indices and nothing else, which is what the assignment means by "your final
# tokenizer code should provide an interface to the Hugging Face code".
#
# Assignment compliance, clause by clause:
#   "you do not need to implement BPE"          the merge-learning algorithm is entirely
#                                               trainers.BpeTrainer; we write no merge loop.
#   "the standard BPE implementation from       models.BPE plus trainers.BpeTrainer from the
#    Hugging Face"                              `tokenizers` package, which is that library.
#   "don't use any of their pre-trained         nothing is ever downloaded or loaded from the
#    tokenizers"                                Hub. No from_pretrained, no AutoTokenizer.
#                                               test_part1.py scans this directory to prove it.
#   "generate a vocabulary for our training     every vocabulary here is trained from the two
#    material"                                  assignment text files and nothing else.
#
# Why the wrapper matters beyond tidiness: the n-gram models count on vocabulary indices, so if
# a Hugging Face object leaked downstream it would be possible to score two authors under two
# different segmentations without noticing. Keeping the boundary here makes that impossible.

from tokenizers import Tokenizer, Regex, models, trainers, pre_tokenizers, decoders, normalizers

# Index 0, 1 and 2 respectively. The trainer assigns special tokens first, in order, so these
# are stable across every vocabulary size and pre-tokenizer; selfcheck.py pins them.
SPECIALS = ["<unk>", "<s>", "</s>"]
UNK_ID, BOS_ID, EOS_ID = 0, 1, 2

PRETOKS = ("ws", "wsp", "gpt", "byte", "meta")

# The GPT-3 pre-tokenizer regex from lecture 3, slide 18. Splits contractions, then runs of
# letters, digits and punctuation, each optionally preceded by one space so that word-initial
# position stays recoverable.
#
# To be unambiguous, since the name invites the question: this is a published regular
# expression for splitting text, not a pre-trained tokenizer. No GPT vocabulary, no GPT merges
# and no GPT weights are involved. The regex only decides where pre-token boundaries fall; the
# vocabulary is still learned by BPE from the two assignment texts.
GPT_PATTERN = (r"'s|'t|'re|'ve|'m|'ll|'d"
               r"| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+")


class BpeTokenizer:
    """A byte-pair-encoding tokenizer over one fixed vocabulary.

    Parameters
        vocab_size  requested vocabulary size; the realized size can be smaller, so always
                    read it back from the `vocab_size` property rather than trusting this.
        pretok      one of PRETOKS; decides what BPE is allowed to merge across.
        lowercase   folds case inside the tokenizer pipeline, so it is saved into the
                    tokenizer file and cannot diverge between training and prediction.
    """

    def __init__(self, vocab_size=4000, pretok="wsp", lowercase=False):
        if pretok not in PRETOKS:
            raise ValueError("pretok must be one of %r" % (PRETOKS,))
        self.requested_vocab_size = vocab_size
        self.pretok = pretok
        self.lowercase = lowercase
        self._tk = None

    # ---------------------------------------------------------------- construction

    def _make_pretokenizer(self):
        """Return the pre-tokenizer, the decoder, and whether decoding is exactly lossless.

        Pre-tokenization runs before BPE and fixes the boundaries no merge may cross. That
        makes it the first line of defence against the typographic leak: if punctuation is
        split off as its own pre-token, no merge can ever fuse a quote character into a word
        type, so the vocabulary itself cannot encode which edition the text came from.
        """
        if self.pretok == "ws":
            # Whitespace() splits on word/non-word transitions, so a run of punctuation such
            # as ?" stays glued together as a single pre-token.
            return pre_tokenizers.Whitespace(), None, False

        if self.pretok == "wsp":
            # TODO 1 (yours).
            # Return a pre-tokenizer that first splits on whitespace and then splits EVERY
            # punctuation character into its own pre-token. Decoder is None, lossless False.
            #
            # You want pre_tokenizers.Sequence([...]) wrapping two pre-tokenizers:
            # WhitespaceSplit() and Punctuation(). Order matters, think about which runs first.
            #
            # Check your answer by reading the printout from toy_bpe.py, and by the assertion
            # in test_part1.py that no vocabulary item under "wsp" contains a quote character.
            raise NotImplementedError("TODO 1: build the whitespace + punctuation pre-tokenizer")

        if self.pretok == "gpt":
            # The lecture-3 regex, then byte-level mapping with its own regex disabled so the
            # split above is the only one that applies.
            return (pre_tokenizers.Sequence([
                        pre_tokenizers.Split(Regex(GPT_PATTERN), behavior="isolated"),
                        pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False)]),
                    decoders.ByteLevel(), True)

        if self.pretok == "byte":
            # GPT-2 style: the same regex is applied internally by ByteLevel(use_regex=True).
            return pre_tokenizers.ByteLevel(add_prefix_space=False), decoders.ByteLevel(), True

        # "meta": SentencePiece style, marking word-initial position with a visible character.
        # Kept in the comparison because it demonstrates the failure mode: with no punctuation
        # split, BPE merges quote marks and even line breaks into word types.
        return pre_tokenizers.Metaspace(), decoders.Metaspace(), False

    def _make_trainer(self):
        """Return the BpeTrainer for this configuration.

        TODO 2 (yours). Build and return a trainers.BpeTrainer with:
          - vocab_size        self.requested_vocab_size
          - special_tokens    SPECIALS, so that <unk>, <s> and </s> take indices 0, 1, 2
          - show_progress     False, so sweeps do not spam the console
          - initial_alphabet  ONLY for the byte-level pre-tokenizers ("byte" and "gpt"):
                              pre_tokenizers.ByteLevel.alphabet()

        That last argument is the interesting one and it is worth understanding before you
        write it. Lecture 3 slide 19 says byte-pair encoding has no out-of-vocabulary words
        because the vocabulary starts from UTF-8. That is true of the algorithm but NOT of
        this library's default behaviour: unless you seed the trainer with all 256 byte
        characters, the alphabet is only what the training corpus happened to contain, and a
        character the corpus never showed still comes back as <unk>. Seeding it makes <unk>
        structurally unreachable and makes decode(encode(s)) exact.

        You can verify both branches yourself after implementing this, with:
            python src/toy_bpe.py --check-unk
        """
        raise NotImplementedError("TODO 2: build the BpeTrainer")

    def train(self, texts):
        """Train the vocabulary from an iterable of strings. Returns self."""
        tk = Tokenizer(models.BPE(unk_token="<unk>"))
        if self.lowercase:
            tk.normalizer = normalizers.Lowercase()
        pre, dec, lossless = self._make_pretokenizer()
        tk.pre_tokenizer = pre
        if dec is not None:
            tk.decoder = dec
        self._lossless = lossless
        tk.train_from_iterator(list(texts), self._make_trainer())
        self._tk = tk
        return self

    # ---------------------------------------------------------------- persistence

    def save(self, path):
        self._require()
        self._tk.save(str(path))

    @classmethod
    def load(cls, path, pretok="wsp", lowercase=False):
        """Reload a vocabulary this project trained and saved.

        `Tokenizer.from_file` reads a local file we produced with save(). It is not
        `from_pretrained` and it never contacts the Hugging Face Hub, so this path does not
        introduce a pre-trained tokenizer.
        """
        obj = cls(pretok=pretok, lowercase=lowercase)
        obj._tk = Tokenizer.from_file(str(path))
        obj._lossless = pretok in ("byte", "gpt")
        return obj

    # ---------------------------------------------------------------- use

    def _require(self):
        if self._tk is None:
            raise RuntimeError("tokenizer has not been trained; call train() first")

    @property
    def vocab_size(self):
        """The REALIZED vocabulary size. Never assume it equals what was requested."""
        self._require()
        return self._tk.get_vocab_size()

    @property
    def lossless(self):
        """True if decode(encode(s)) reproduces s exactly for arbitrary input."""
        return getattr(self, "_lossless", False)

    unk_id, bos_id, eos_id = UNK_ID, BOS_ID, EOS_ID

    def encode(self, text):
        """Text to a list of vocabulary indices, with no sentence padding."""
        self._require()
        return self._tk.encode(text).ids

    def encode_batch(self, texts):
        self._require()
        return [e.ids for e in self._tk.encode_batch(list(texts))]

    def decode(self, ids):
        self._require()
        return self._tk.decode(list(ids))

    def id_to_token(self, i):
        self._require()
        return self._tk.id_to_token(i)

    def token_to_id(self, t):
        self._require()
        return self._tk.token_to_id(t)

    def vocab(self):
        self._require()
        return self._tk.get_vocab()

    def encode_sentences(self, sentences, n):
        """Encode sentences and add the padding an n-gram model of order n needs.

        TODO 3 (yours). Return a list of lists of vocabulary indices, one per input sentence.
        Each one is the sentence's own indices with boundary symbols added:

            n = 2   [BOS_ID] + ids + [EOS_ID]
            n = 3   [BOS_ID, BOS_ID] + ids + [EOS_ID]

        In general: (n - 1) copies of BOS_ID in front, one EOS_ID at the end.

        Three things worth understanding before you write two lines of code, because the whole
        of Part 2 rests on them.

        Why any BOS at all: without it there is no context for the first real word, so the
        model could not tell you how likely a sentence is to START with "Bilbo". Lecture 4
        slide 31 writes the worked example exactly this way.

        Why (n - 1) of them: a trigram model conditions on the two preceding tokens, so the
        first real token needs two tokens of history to sit in.

        Why EOS is added but never predicted-from: it has to be PREDICTED, or the model is not
        a probability distribution over strings of varying length and perplexity is not well
        defined. It is never a context, because nothing follows it.

        The consequence you should be able to state: with this convention a bigram and a
        trigram model score the SAME NUMBER of events on the same text, len(ids) + 1 each.
        That is the only reason their perplexities can be compared at all, and selfcheck.py
        asserts it.
        """
        raise NotImplementedError("TODO 3: pad the encoded sentences")
