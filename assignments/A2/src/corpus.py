# corpus.py -- load the books and cut them up.
#
# Nothing conceptually new here. split_sentences is the function Yiyuan wrote in the Part 2
# tutorial, moved out of the scratch file so everything can share one copy. The rest is file
# loading and slicing.
#
# Plumbing by Claude; split_sentences by Yiyuan Jia.

import re
from pathlib import Path

DATA = Path(__file__).parent.parent / "data"

BOOKS = {
    "tolkien": "hobbit.txt",
    "doyle": "lostworld.txt",
}


def strip_gutenberg(text):
    """Remove the Project Gutenberg licence wrapper, if there is one.

    The Lost World file has about 379 lines of licence text around the novel. Left in, it
    would put words like "Gutenberg" and "eBook" into the vocabulary, and those appear in
    only one of our two books, so a classifier could use them to tell the books apart. That
    would be measuring the file, not the author. The Hobbit file has no wrapper and is
    returned unchanged.
    """
    start = text.find("*** START")
    end = text.find("*** END")
    if start == -1 or end == -1:
        return text
    return text[text.index("\n", start) + 1:end]


def split_sentences(text):
    """Cut running text into sentences.

    Splits after a full stop, question mark or exclamation mark followed by a space. Rough:
    it breaks on abbreviations like "Mr. Baggins". Good enough for this assignment.
    """
    text = text.replace("\n", " ")
    parts = re.split(r"(?<=[.!?]) +", text)

    sentences = []
    for part in parts:
        part = part.strip()
        if part != "":
            sentences.append(part)
    return sentences


def load_sentences(author):
    """Read one author's book and return it as a list of sentences."""
    with open(DATA / BOOKS[author], encoding="utf-8") as f:
        text = f.read()
    return split_sentences(strip_gutenberg(text))


def train_test_split(sentences, test_fraction=0.1):
    """Hold back the last slice of a book for testing.

    Crude on purpose. Taking the end of the book means the held-out text is the closing
    chapters, which have a different cast and vocabulary from the opening, so we are partly
    measuring how well the model handles the ending. A better split would take pieces from
    throughout the book.
    """
    cut = int(len(sentences) * (1 - test_fraction))
    return sentences[:cut], sentences[cut:]


def make_passages(sentences, words_per_passage):
    """Group consecutive sentences into passages of roughly the given length.

    Used to build the evaluation set for author identification: a passage is what we ask
    the classifier to label.
    """
    passages = []
    current = []
    for sentence in sentences:
        current.append(sentence)
        if len(" ".join(current).split()) >= words_per_passage:
            passages.append(" ".join(current))
            current = []
    return passages
