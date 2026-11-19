"""Checker for tell-free rewrites that keep the facts."""
import re
import sys
from pathlib import Path

PIPELINE_DIR = Path(__file__).resolve().parent
REPO_ROOT = PIPELINE_DIR.parent
sys.path.insert(0, str(REPO_ROOT / "rag"))

import tells  # noqa: E402

BANNED = ("speak plainly", "learned not to guess", "fairly rare", "fighting nonsense")
FIX_TELLS = {"staccato", "button_close", "service", "question_back"}

NUMBER_WORDS = [
    "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
    "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen",
    "seventeen", "eighteen", "nineteen", "twenty", "hundred", "thousand",
]


def _words(text):
    return re.findall(r"\S+", text or "")


def _lower_words(text):
    return set(w.strip(".,!?;:\"'()[]{}").lower() for w in _words(text) if w.strip(".,!?;:\"'()[]{}"))


def needs_rewrite(reply):
    reply = reply if reply is not None else ""
    tells_set = tells.reply_tells(reply)
    if tells_set & FIX_TELLS:
        return True
    low = reply.lower()
    return any(b in low for b in BANNED)


def _digit_runs(text):
    return re.findall(r"\d+", text or "")


def _number_words_in(text):
    found = []
    for w in _words(text):
        clean = w.strip(".,!?;:\"'()[]{}").lower()
        if clean in NUMBER_WORDS:
            found.append(clean)
    return found


def _sentence_initial_positions(text):
    positions = set()
    sentence_start = True
    prev_end = 0
    for m in re.finditer(r"\S+", text or ""):
        if sentence_start:
            positions.add(m.start())
        gap = text[prev_end:m.start()]
        if "\n" in gap:
            sentence_start = True
        else:
            sentence_start = False
        word = m.group()
        stripped = word.rstrip().rstrip("\"'").rstrip()
        if stripped.endswith((".", "!", "?")):
            sentence_start = True
        prev_end = m.end()
    return positions


def check(original, rewrite):
    problems = []
    orig = original if original is not None else ""
    rw = rewrite if rewrite is not None else ""

    rewrite_tells = tells.reply_tells(rw)
    for t in sorted(rewrite_tells & FIX_TELLS):
        problems.append(f"still has {t}")

    low_rw = rw.lower()
    if any(b in low_rw for b in BANNED):
        problems.append("banned phrase")

    if rw.count("?") > orig.count("?"):
        problems.append("new question")

    orig_words = orig.split()
    rw_words = rw.split()
    if len(orig_words) == 0:
        ratio = 0.0 if len(rw_words) == 0 else float("inf")
    else:
        ratio = len(rw_words) / len(orig_words)
    if ratio < 0.8 or ratio > 1.3:
        problems.append(f"length ratio {ratio}")

    orig_digit = set(_digit_runs(orig))
    orig_numwords = set(_number_words_in(orig))
    ordered_numbers = []
    seen = set()
    for w in _words(rw):
        for d in re.findall(r"\d+", w):
            if d not in orig_digit and d not in seen:
                seen.add(d)
                ordered_numbers.append(d)
        clean = w.strip(".,!?;:\"'()[]{}").lower()
        if clean in NUMBER_WORDS and clean not in orig_numwords and clean not in seen:
            seen.add(clean)
            ordered_numbers.append(clean)
    for n in ordered_numbers:
        problems.append(f"new number {n}")

    orig_lower_words = _lower_words(orig)
    positions = _sentence_initial_positions(rw)
    seen_names = set()
    ordered_names = []
    for m in re.finditer(r"\S+", rw):
        word = m.group()
        clean = word.strip(".,!?;:\"'()[]{}")
        if not clean or not clean[0].isupper():
            continue
        if m.start() in positions:
            continue
        low = clean.lower()
        if low == "i" or low.startswith(("i'", "i’")):   # the pronoun is not a name
            continue
        if low in orig_lower_words:
            continue
        if low in seen_names:
            continue
        seen_names.add(low)
        ordered_names.append(clean)
    for w in ordered_names:
        problems.append(f"new name {w}")

    return problems
