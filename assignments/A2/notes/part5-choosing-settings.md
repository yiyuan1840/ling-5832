# Part 5: choose the settings honestly, then finish the report

This part is yours. I am not going to run these experiments or tell you what they conclude.

Note that the vocabulary table I showed you earlier is **not usable as evidence**, because I
computed it on the test set. Whatever it appeared to say, you cannot cite it. Step 1 explains
why, and the numbers may come out differently once measured properly.

---

## The problem to fix

Right now the report picks a vocabulary size of 4,000, a bigram model and k = 0.01. Three things
are wrong with how those were chosen.

**We tuned on the test set.** Every accuracy number in the report comes from the same held out
text we used to compare settings. Choosing the setting that scores best on your test set and
then reporting that score is not an honest estimate of how well the method works. It is the most
common mistake in applied machine learning and the report currently makes it.

**The measurement cannot tell the settings apart.** At 50 word passages almost everything scores
above 99%, so the whole vocabulary sweep spans about two passages out of 255. A difference that
small is noise. You cannot choose between six options using a measurement where five of them are
effectively tied.

**Perplexity could not break the tie.** Section 5 of the report already explains why it is not
comparable across vocabulary sizes.

---

## Step 1. Add a development set

**TODO 1** in `src/corpus.py`. There is a skeleton waiting.

Split each book three ways instead of two: training, development, and test. Choose the settings
using development, then touch the test set exactly once, at the end, to report the final number.

Things to decide for yourself, and to justify in the report:

* How large should dev and test be? You have roughly 4,900 and 3,700 sentences per book, and the
  passage counts fall as the passage length rises. Work out how many passages each split will
  actually contain before committing to the proportions.
* Should the splits be contiguous slices, or taken from throughout the book? Section 2 of the
  report already admits the tail is a weak choice. This is where to fix it if you want to.

## Step 2. Work out what to measure

Before running anything, decide what would actually settle the question.

* At what passage length is the task hard enough that different settings give visibly different
  results? Look at the accuracy by length table you already have for a clue.
* How big does a difference have to be before you believe it? With N passages, a difference of
  one or two is noise. Decide your threshold **before** you see the numbers, otherwise you will
  talk yourself into whichever result appears.

## Step 3. Run the sweep on development data

Vocabulary size, model order, and k. `src/experiments.py` currently measures on test; you will
need it pointed at dev.

Watch for one thing: these three settings may interact. The best k could differ between a
vocabulary of 1,000 and one of 8,000. Decide whether you are sweeping them one at a time or
together, and be able to say why.

## Step 4. Commit to the settings, then measure once

Fix your choices. Run the test set. Report that number and do not go back and adjust.

If the test number is noticeably worse than the dev number, that is worth reporting rather than
hiding. The gap is itself a finding about how much the tuning was fitting noise.

## Step 5. Update the report

Sections that will need changing:

* **1** the headline numbers
* **2** the split description
* **5 and 6** every table, and the vocabulary discussion, which currently claims 4,000 peaks
* **7** add the tuning problem as something you found and fixed, since that is a genuine
  observation about method and the assignment asks for those

Rerun `uv run render_report.py` when done.

---

## Questions I will ask you afterwards

Have answers ready.

1. Why does choosing a setting on the test set inflate the reported result? What is the
   mechanism?
2. Your dev and test sets come from the same two books. What does your final accuracy actually
   estimate, and what does it not?
3. If dev says one vocabulary size and test disagrees, which do you report, and why?
