# Assignment 2: Language Models and Author Identification

LING 5832. Verbatim transcription of the assignment text as posted on Canvas, kept here so the
repository records exactly what was asked. Link artifacts from the Canvas page (the repeated
"Download" text) and the original's typographical slips are preserved as written; the
smoothing parameter note reads "if k-1" in the source and means k equal to 1.

---

In this assignment, you will implement a BPE-based tokenizer, several N-gram language models,
and a perplexity measure. You will then use these components to implement a simple language
modeling approach to author identification. Based on some training materials, you will be asked
to use your LM to classify a set of text segments according to their author.

## Resources

We'll be using the following text samples as representative training text: The Hobbit by JRR
TolkienDownload The Hobbit by JRR Tolkien and The Lost World by Arthur Conan DoyleDownload The
Lost World by Arthur Conan Doyle. Both collections are simple text files with no formatting of
any consequence.

## Tokenization

For this part, you will implement a tokenizer and generate a vocabulary based given a corpus.
Specifically, you will take a simple text file (like the training data) and generate a
vocabulary of a specified size (up to you) using a byte-pair encoding approach. You do not need
to implement BPE. Instead you should use the standard BPE implementation from Hugging Face.
Don't use any of their pre-trained tokenizers. Use their implementation of BPE to generate a
vocabulary for our training material. You should feel free to explore different pre-tokenization
strategies and vocabulary sizes. Your final tokenizer code should provide an interface to the
Hugging Face code.

## N-Gram Language Modeling

For this part, you will implement training and inference components for both bigram and trigram
language models. The training consists of collecting counts from text tokenized with your
tokenizer (unigram, bigram and trigram counts). These counts should be computed and stored based
on the index into the vocabulary provided by your tokenizer.

Inference consists of returning the negative log probability of a sequence given a specified
model. To deal with sparsity, use Add-k smoothing (i.e, add-1 smoothing if k-1). Feel free to
explore other values of k as part of your explorations.

Finally, you will also implement perplexity as the per-word average negative log probability of
a test set given a model. You should be able to examine the perplexity of a bigram model to a
trigram model, as well as the relative performance of different values of k in add-k smoothing.

## Author Identification

Using the training data, implement a LM-based approach author identification. You will be given
a test set of short passages from Tolkien and Conan-Doyle. Your task is to classify the texts by
their author. To do this you will use the N-gram LMs trained in the previous part of the HW. To
be specific, you must train a LM for Tolkien and another one for Conan-Doyle using the given
training data. You can then compute the probability of an unseen text against these models and
generate a prediction.

The exact combination of approaches used in this part is up to you (vocabulary, unigram vs
bigram vs trigram models, choice of k for smoothing, etc). You should explore these various
options with an eye towards effective author identification. That is, the final code should
reflect the hyper-parameter choices that work well with this classification task.

## Deliverables

1. A short report on what you did for each part. It should include exactly what you tried across
   each of the required deliverables. Be sure to include any difficulties you encountered or
   important observations.
2. The code (well commented; including description of any use of pre-existing or AI generated
   code).
3. Author ID labels from a test-set we will provide.

## Caveats

You should feel free to use generic python libraries/packages (collections, numpy, regex,
pandas) in your code. Don't use any established solutions that simply implement N-gram models.
