# authorid.py -- score a passage against one author's model.
#
# To guess an author, score the same passage against each author's model and take the
# lowest. That comparison lives at the call site, in main.py, rather than here: it is three
# obvious lines, and wrapping it in a function meant carrying a dictionary of models around
# to keep each model paired with its author's name.
#
# The reasoning is in notes/part4-authorid.md.
#
# Skeleton by Claude; the body is Yiyuan Jia's.

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from tokenizer import encode, pad


def score(passage, tokenizer, model, n):
    """How surprised one model is by one passage, averaged per prediction.

    Lower means the model finds the passage more ordinary, so the author whose model
    gives the lowest score is the guess.

    Dividing by the number of predictions does not change which model wins, because both
    models see the same passage chopped the same way, so the count is identical for both.
    It matters because it makes the GAP between two scores comparable across passages of
    different lengths, which is what tells you how close a call was.
    """
    sequence = pad(encode(tokenizer, passage), n)
    return model.cross_entropy([sequence])
