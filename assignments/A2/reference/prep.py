# prep.py -- corpus loading, structural cleanup, and the three normalization levels.
#
# Provenance: written by Claude (Anthropic) as scaffolding for LING 5832 A2, from structural
# facts measured on the two source files; reviewed and accepted by Yiyuan Jia. This module is
# plumbing, not one of the assignment's intellectual deliverables.
#
# The pipeline order is fixed and every step is line-level until unwrap_paragraphs:
#   strip_gutenberg -> drop_structure -> rejoin_eol_hyphens -> unwrap_paragraphs
#   -> normalize(level) -> split_sentences
#
# Only normalize() depends on the level. The three levels exist so that the contribution of
# typographic cleanup to author identification can be measured rather than assumed:
#   "raw"     nothing is changed. What a careless pipeline produces.
#   "minimal" the author-disjoint typographic channels are folded together.
#   "full"    minimal, plus enumerable scanner-damage repair.

import re
import unicodedata

LEVELS = ("raw", "minimal", "full")
SOURCES = ("tolkien", "doyle")

# --- structural patterns -------------------------------------------------------------------

_GUT_START = re.compile(r"\*\*\*\s*START OF (THE|THIS) PROJECT GUTENBERG", re.I)
_GUT_END = re.compile(r"\*\*\*\s*END OF (THE|THIS) PROJECT GUTENBERG", re.I)

# Tolkien: 19 lines of the form "Chapter IV". Two of them are scanner damage that split the
# numeral ("Chapter X III", "Chapter X IX"), hence the repeated-group pattern.
_TOLKIEN_CHAPTER = re.compile(r"Chapter(\s+[IVXLC]+)+\s*$")

# Doyle: each chapter opens with a deeply indented "CHAPTER XVI" followed by a quoted title.
# The indent requirement is what distinguishes these from in-story text; the novel contains a
# letter, a verse quotation and a newspaper report that are also indented but are real prose.
_DOYLE_CHAPTER = re.compile(r"^\s{15,}CHAPTER\s+[IVXLC]+\s*$")

# Characters that occur in only one of the two books and only as production artifacts. Dropped
# at the "minimal" level so they cannot act as an author signal. Currency and percent signs are
# deliberately NOT here: they occur once or twice but are meaningful in the prose.
_ARTIFACT_CHARS = "™•*#{}[]+&"

# The single line-end hyphen in the Tolkien text that is scanner-mangled punctuation rather than
# a real compound hyphen. Verified by hand against all 89 line-end hyphens; the other 88 are
# genuine compounds that happen to fall at a line break, including three that continue onto an
# uppercase word (North-East, Sackville-Bagginses twice) and two that span a scan page break.
_DASH_AT_EOL = {("hates-", "Smash")}


def is_caps_line(stripped):
    """True if the line has letters and none of them are lowercase.

    Used only for the Tolkien text, whose chapter titles are set in capitals. The Doyle text
    needs no such rule: its only flush-left capitals line is a signature inside a letter.
    """
    return bool(re.fullmatch(r"[^a-z]+", stripped)) and any(c.isupper() for c in stripped)


def strip_gutenberg(text):
    """Return only the text between the Project Gutenberg START and END markers.

    The Doyle file carries 379 lines of licence boilerplate. Left in, it contributes an entire
    vocabulary (gutenberg, foundation, electronic, donations) and the only curly quotes and em
    dashes in that book, which would corrupt the typographic measurements. The Tolkien file has
    no markers and is returned unchanged.
    """
    lines = text.split("\n")
    starts = [i for i, l in enumerate(lines) if _GUT_START.search(l)]
    ends = [i for i, l in enumerate(lines) if _GUT_END.search(l)]
    if not starts or not ends:
        return text, 0
    body = lines[starts[0] + 1:ends[0]]
    return "\n".join(body), len(lines) - len(body)


def drop_structure(lines, source):
    """Remove front matter, chapter headings and tables of contents.

    Returns (kept_lines, dropped) where dropped is a list of (index, reason, text) so the
    driver can print every removal for inspection. Indentation is significant here and is only
    stripped from the right.
    """
    kept, dropped = [], []
    if source == "tolkien":
        for i, line in enumerate(lines):
            t = line.strip()
            if _TOLKIEN_CHAPTER.fullmatch(t):
                dropped.append((i, "chapter", t))
            elif t and is_caps_line(t) and '"' not in t and "”" not in t:
                # The quote test protects line 6869, LEAVE THE PATH!", which is shouted
                # dialogue rather than a heading. No genuine heading contains a quote mark.
                dropped.append((i, "caps-heading", t))
            else:
                kept.append(line)
        return kept, dropped

    # Doyle. Drop each "CHAPTER <roman>" line and the quoted title line that follows it, and
    # drop everything before the first such pair, which is the title page, the epigraph, the
    # foreword and the table of contents.
    chapter_idx = [i for i, l in enumerate(lines) if _DOYLE_CHAPTER.match(l.rstrip())]
    kill = set()
    for i in chapter_idx:
        kill.add(i)
        j = i + 1
        while j < len(lines) and not lines[j].strip():
            j += 1
        if j < len(lines):
            kill.add(j)  # the chapter title
    if chapter_idx:
        first_title = max(k for k in kill if k <= chapter_idx[0] + 3)
        kill |= set(range(0, first_title + 1))
    for i, line in enumerate(lines):
        if i in kill:
            if line.strip():
                dropped.append((i, "front-matter/heading", line.strip()))
        else:
            kept.append(line)
    return kept, dropped


def rejoin_eol_hyphens(lines):
    """Join words the typesetter broke across a line, keeping the hyphen.

    Keeping the hyphen is the right default here: of the 89 line-end hyphens in the Tolkien
    text, 88 are real compound hyphens (seed-cakes, hobbit-hole, drawing-room) that merely
    happen to land at a line break. Only one is a mangled dash, listed in _DASH_AT_EOL.
    Two cases span a scan page break with blank lines between, so the continuation is the next
    non-blank line rather than the next line.
    """
    out = list(lines)
    joined = 0
    i = 0
    while i < len(out):
        stripped = out[i].rstrip()
        if re.search(r"[A-Za-z]-$", stripped):
            j = i + 1
            while j < len(out) and not out[j].strip():
                j += 1
            if j < len(out) and out[j].strip():
                tail = stripped.split()[-1]
                head = out[j].lstrip().split()[0] if out[j].split() else ""
                if (tail, head) in _DASH_AT_EOL:
                    out[i] = stripped[:-1] + " — " + out[j].lstrip()
                else:
                    out[i] = stripped + out[j].lstrip()
                for b in range(i + 1, j + 1):
                    out[b] = ""
                joined += 1
                continue
        i += 1
    return out, joined


def unwrap_paragraphs(lines):
    """Turn hard-wrapped lines into one string per paragraph, split on blank lines.

    Both books are wrapped to a fixed column, so a line break inside a paragraph carries no
    information and would otherwise become a spurious token boundary. Lines are joined with a
    single space; any run of spaces that was already inside a source line survives, which is
    what preserves the sentence-spacing channel for measurement at the "raw" level.
    """
    paragraphs, cur = [], []
    for line in lines:
        if line.strip():
            cur.append(line.strip())
        elif cur:
            paragraphs.append(" ".join(cur))
            cur = []
    if cur:
        paragraphs.append(" ".join(cur))
    return paragraphs


def normalize(text, level):
    """Apply the level's character normalization. This is the only level-dependent step.

    "minimal" closes the seven typographic channels that are fully disjoint between the two
    editions: curly versus straight quotes and apostrophes, em dash versus double hyphen, and
    the sentence-spacing convention. Those channels identify a typesetter, not an author.
    Intraword hyphens are deliberately left alone: both books use them at comparable rates.
    """
    if level not in LEVELS:
        raise ValueError("level must be one of %r" % (LEVELS,))
    if level == "raw":
        return text

    t = unicodedata.normalize("NFKC", text)
    t = t.replace("“", '"').replace("”", '"')
    t = t.replace("‘", "'").replace("’", "'")
    # every dash form becomes a single spaced em dash, which stays distinct from the
    # intraword hyphen so no later pattern has to disambiguate the two
    t = re.sub(r"--+|[–—―]", " — ", t)
    for ch in _ARTIFACT_CHARS:
        t = t.replace(ch, "")
    # the step that closes the largest channel: Doyle double-spaces after every sentence at
    # 37 per 1000 words and Tolkien never does
    t = re.sub(r"\s+", " ", t).strip()

    if level == "full":
        # 14 scanner scars, all inside the songs of the Tolkien text, where a stray solidus
        # replaced or joined a letter (smash\, know/., s/ord, Willow/). Doyle has none.
        # This repairs 13; the fourteenth reads "1/1 ine" for "wine" and is not recoverable by
        # rule, so it is left in and reported rather than guessed at.
        t = re.sub(r"(?<=[A-Za-z])[/\\]|[/\\](?=[A-Za-z])", "", t)
        t = re.sub(r"([!?])\1{2,}", r"\1\1", t)
        t = re.sub(r"\s+", " ", t).strip()
    return t


# --- sentence segmentation ------------------------------------------------------------------

_ABBREV = {
    "mr", "mrs", "ms", "dr", "st", "prof", "no", "esq", "messrs", "capt", "lieut", "col",
    "gen", "sgt", "rev", "hon", "jr", "sr", "vs", "etc", "inc", "ltd", "sir", "co",
}
_CANDIDATE = re.compile(r"([.!?]+[\"'”’)\]]*)(\s+)(?=[\"'“‘(\[]?[A-Z0-9])")


def split_sentences(paragraph):
    """Split a paragraph into sentences. Hand-written; no NLTK or other segmenter.

    A boundary is a run of sentence punctuation, optionally followed by a closing quote or
    bracket, then whitespace, then something that can open a sentence. Two exceptions are
    suppressed: a known abbreviation (Mr., Dr., St.) and a single-letter initial (E. D. Malone,
    J.R.R. Tolkien), both of which end in a period without ending a sentence.
    """
    cuts = []
    for m in _CANDIDATE.finditer(paragraph):
        head = paragraph[:m.start(1)]
        word = re.search(r"([A-Za-z]+)$", head)
        if m.group(1).startswith(".") and word:
            w = word.group(1)
            if w.lower() in _ABBREV or (len(w) == 1 and w.isupper()):
                continue
        cuts.append(m.end(2))
    out, prev = [], 0
    for c in cuts:
        s = paragraph[prev:c].strip()
        if s:
            out.append(s)
        prev = c
    tail = paragraph[prev:].strip()
    if tail:
        out.append(tail)
    return out


# --- measurement --------------------------------------------------------------------------

CHANNELS = [
    ("double space after sentence", lambda t: len(re.findall(r"[.!?][\"'”’)]?  +", t))),
    ("curly double quote", lambda t: t.count("“") + t.count("”")),
    ("straight double quote", lambda t: t.count('"')),
    ("curly apostrophe", lambda t: t.count("’")),
    ("straight apostrophe", lambda t: t.count("'")),
    ("em dash", lambda t: t.count("—")),
    ("double hyphen", lambda t: len(re.findall(r"--", t))),
    ("semicolon", lambda t: t.count(";")),
    ("colon", lambda t: t.count(":")),
    ("exclamation", lambda t: t.count("!")),
    ("question mark", lambda t: t.count("?")),
    ("comma", lambda t: t.count(",")),
    ("intraword hyphen", lambda t: len(re.findall(r"[A-Za-z]-[A-Za-z]", t))),
    ("solidus", lambda t: t.count("/") + t.count("\\")),
]


def channel_stats(text):
    """Count each typographic channel and return raw counts plus rates per 1000 words.

    These are the numbers that decide which differences between the two books are author style
    and which are the typesetter. A channel present in one book and absent in the other is a
    leak; a channel present in both at different rates may be genuine style.
    """
    n_words = len(re.findall(r"\S+", text)) or 1
    return {
        name: {"count": fn(text), "per_1k": fn(text) / n_words * 1000}
        for name, fn in CHANNELS
    }


def load_book(path, source, level):
    """Run the whole pipeline on one book and return a dict of everything downstream needs.

    Keys: paragraphs (list of normalized strings), sentences (flat list), para_of (the
    paragraph index each sentence came from, needed so held-out passages never straddle a
    split boundary), and counts for the driver to report.
    """
    if source not in SOURCES:
        raise ValueError("source must be one of %r" % (SOURCES,))
    with open(path, encoding="utf-8") as f:
        text = f.read()
    text, n_boiler = strip_gutenberg(text)
    lines = [l.rstrip() if l.strip() else "" for l in text.split("\n")]
    # indentation is load-bearing for Doyle, so rstrip only
    lines = [l if l.strip() else "" for l in lines]
    kept, dropped = drop_structure(lines, source)
    kept, n_joined = rejoin_eol_hyphens(kept)
    paragraphs = unwrap_paragraphs(kept)
    paragraphs = [normalize(p, level) for p in paragraphs]
    paragraphs = [p for p in paragraphs if p]

    sentences, para_of = [], []
    for i, p in enumerate(paragraphs):
        for s in split_sentences(p):
            sentences.append(s)
            para_of.append(i)
    return {
        "source": source,
        "level": level,
        "paragraphs": paragraphs,
        "sentences": sentences,
        "para_of": para_of,
        "n_boilerplate_lines": n_boiler,
        "n_dropped_lines": len(dropped),
        "dropped": dropped,
        "n_hyphen_joins": n_joined,
        "n_words": sum(len(p.split()) for p in paragraphs),
    }
