#!/usr/bin/env python3
"""
temp_sweep.py — does sampling temperature cost the persona its facts?

The identity card in rag/camus_rag.py tells the model to name its dogs (Pauline,
Kirk, Blaise) and its three novels, and never to add to either list. That
instruction is honoured or ignored per sample, not per system prompt, so the only
way to know the risk is to sample the same prompt across temperatures and count
what broke.

For each temperature the turn is assembled by camus_rag.build_turn — the same
retrieval, the same gates, the same options as chat — and then sent to Ollama
/api/chat with only `temperature` replaced. Nothing else changes, so any
difference is the temperature and nothing else. Memory is off (build_turn's
default), so no profile.md and no cross-session facts enter the prompt.

Four factual prompts (about the dogs, the pets, the novels, the bibliography) and
four conversational ones (no factual content to get wrong). Each factual prompt is
sampled --n times; each conversational prompt half that, because a temperature
that starts inventing dogs names will not be caught by a chit-chat sample.

Checks, applied per answer (see check_answer):
  pets  dogs_named — how many of Pauline/Kirk/Blaise appear (a count, not a flag)
        denies_dogs — "I had no dogs", "I never kept a dog", ...
        species_swap — the dog's name attached to another species
        invented_name — a capitalised, non-sentence-initial token outside
                        {Cigarette, Pauline, Kirk, Blaise}
  works stranger+plague+fall_all — all three named
        essay_as_novel — The Myth of Sisyphus or The Rebel called a novel
        invented_title — a quoted or italicised title not in the identity card
        (the whitelist is parsed from camus_rag._IDENTITY_CARD at run time)
  all   repeat_trigram — some word trigram occurs 3+ times
        over_220_words
        no_terminal_punct — does not end in . ! ? " ' ) ] … — "
        non_english — a sentence with almost no English function words

The invented-name and invented-title checks are deliberately over-sensitive: they
flag every capitalised or quoted phrase they cannot match, and the offending text
is recorded on the output row so a human can judge it. A sweep that hid its
candidates would hide its own false positives too.

    python pipeline/temp_sweep.py                       # full sweep
    python pipeline/temp_sweep.py --temps 0.45 1.2 --n 1 --out /tmp/smoke.jsonl
    python pipeline/temp_sweep.py --dry-run             # print the plan, call nothing
"""
import argparse
import json
import os
import re
import statistics
import sys
import time
from collections import Counter, defaultdict

DEFAULT_TEMPS = (0.45, 0.8, 1.0, 1.2)
DEFAULT_N_FACTUAL = 8
LONG_ANSWER_WORDS = 220
TRIGRAM_REPEAT = 3

DOG_NAMES = ("Pauline", "Kirk", "Blaise")
CAT_NAME = "Cigarette"
KNOWN_PET_NAMES = frozenset(DOG_NAMES + (CAT_NAME,))

FACTUAL_PROMPTS = (
    ("dogs", "what were your dogs' names?"),
    ("pets", "tell me about your pets"),
    ("novels", "list your novels"),
    ("wrote", "what did you write?"),
)
CONVERSATIONAL_PROMPTS = (
    ("hey", "hey"),
    ("rain", "it's raining and I can't be bothered today"),
    ("absurd", "isn't the absurd just a fancy way of giving up?"),
    ("internet", "what do you make of the internet?"),
)
# Which check group each prompt's answers get. The `all` group is added to both.
PET_PROMPTS = frozenset({"dogs", "pets"})
WORK_PROMPTS = frozenset({"novels", "wrote"})

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ── pets ──────────────────────────────────────────────────────────────
_DENIES_DOGS = re.compile(
    r"\b(?:do\s+not|don'?t|never|no)\b[^.?!]{0,40}?\b"
    r"(?:dog|dogs|hound|puppy|puppies|pet|pets|animal|animals)\b",
    re.I)
_DOGS_ONLY_FOR_CATS = re.compile(
    r"\b(?:my|our)\s+(?:only\s+)?(?:pets?|companions?)\b[^.?!]{0,60}"
    r"\b(?:were|was|are|is)\b[^.?!]{0,20}\bcats?\b", re.I)
_SPECIES = (r"cat|cats|kitten|kittens|bird|birds|horse|horses|rabbit|rabbits|"
            r"fish|mouse|mice|goat|goats|donkey|donkeys")
_DOG_NAMES_RE = r"(?:Pauline|Kirk|Blaise)"
# A dog's name attached to another species, in either order: "Pauline was a cat",
# "Blaise, who were birds". "Pauline was a dog" does not match — dog is not in
# _SPECIES — which is the whole point.
_SPECIES_SWAP = re.compile(
    rf"\b(?:{_SPECIES})\b[^.?!]{{0,40}}\b(?:was|were|is|are)\b[^.?!]{{0,15}}"
    rf"\b{_DOG_NAMES_RE}\b"
    rf"|\b{_DOG_NAMES_RE}\b[^.?!]{{0,40}}\b(?:was|were|is|are)\b[^.?!]{{0,15}}"
    rf"\b(?:{_SPECIES})\b", re.I)

# Capitalised tokens that are never a pet's name: the model's own pronouns and
# the few English words that legitimately open a sentence.
_ENGLISH_FORMS = frozenset("""
i i'm i've i'll i'd i'll you're you've you'll you'd he's he'd he's she'd
she's we're we've we'll we'd it's it's that's there's what's who's here's
my mine myself me myself
""".split())

_WORD_RE = re.compile(r"[^\W\d_]+(?:'[^\W\d_]+)*", re.UNICODE)
_QUOTED_RE = re.compile(r"[\"“]([^\"“”]{2,80})[\"”]|‘([^’]{2,80})’")
_ITALIC_RE = re.compile(r"(?:\*\*|\*|_)([^*\n_]{2,80})(?:\*\*|\*|_)")

TERMINAL = ".!?\"')]…—"

# ── identity-card title whitelist ──────────────────────────────────────
_CARD_SEGMENT = re.compile(r"Novels:(.*?)(?:\n- |\Z)", re.S)
_CARD_TITLE_RE = re.compile(r"\b[A-Z][\w'’-]*(?:[ ]+(?:[A-Z][\w'’-]*|[a-z]{1,4}))*")


def _norm_title(text):
    text = re.sub(r"[^a-z0-9 ]+", " ", text.lower())
    words = text.split()
    while words and words[0] in ("the", "a", "an"):
        words.pop(0)
    return " ".join(words)


def card_titles(card):
    """Every title the identity card lists, parsed out of the card itself so the
    whitelist follows the card rather than a copy of it."""
    segment = _CARD_SEGMENT.search(card)
    text = segment.group(1) if segment else card
    titles = set()
    for fragment in re.split(r"[,;.()—\n]", text):
        for match in _CARD_TITLE_RE.finditer(fragment):
            title = _norm_title(match.group())
            if title and title not in ("novels", "stories", "essays", "plays"):
                titles.add(title)
    return titles


# ── checks ────────────────────────────────────────────────────────────

def dog_names_named(answer):
    return sum(1 for name in DOG_NAMES if re.search(rf"\b{name}\b", answer))


def invented_names(answer):
    """Capitalised, non-sentence-initial tokens that are not a known pet name and
    not an English form. Over-sensitive on purpose; see the module docstring."""
    out, sentence_start = [], True
    pos = 0
    for match in re.finditer(r"[^\W\d_]+(?:'[^\W\d_]+)*", answer):
        token = match.group(0)
        gap = answer[pos:match.start()]
        if not (sentence_start or re.fullmatch(r"[\"'\s]*[.!?…]+[\"'\s]*", gap or " ")):
            if token[:1].isupper() and token.lower() not in _ENGLISH_FORMS \
                    and token not in KNOWN_PET_NAMES:
                out.append(token)
        sentence_start = bool(re.fullmatch(r"[\"'\s]*[.!?…]+[\"'\s]*", gap or " "))
        pos = match.end()
    return out


def quoted_titles(answer):
    """Quoted or italicised spans. Single quotes are skipped: the card's
    "these lines are NOT yours" quotes live there and are not titles."""
    out = []
    for match in _QUOTED_RE.finditer(answer):
        out.append(match.group(1) or match.group(2) or "")
    out.extend(_ITALIC_RE.findall(answer))
    return [t.strip() for t in out if t.strip()]


def repeated_trigrams(answer, times=TRIGRAM_REPEAT):
    words = [w.lower() for w in _WORD_RE.findall(answer)]
    counts = Counter(tuple(words[i:i + 3]) for i in range(len(words) - 2))
    return [" ".join(t) for t, n in counts.items() if n >= times]


def word_count(answer):
    return len(_WORD_RE.findall(answer))


def non_english_sentences(answer):
    """Sentences whose function words are mostly not English."""
    out = []
    for sentence in re.split(r"[.!?…]+\s*", answer):
        words = [w.lower() for w in _WORD_RE.findall(sentence)]
        if len(words) < 5:
            continue
        english = sum(1 for w in words if w in _ENGLISH_STOPWORDS)
        if english / len(words) < 0.10:
            out.append(sentence.strip()[:80])
    return out


_ENGLISH_STOPWORDS = frozenset("""
the a an and or but if of to in on at by for with from as is are was were be
been being it its this that these those i you he she we they me him her them
my your his our their not no do does did have has had will would can could
should may might must there here so than then too also just about into over
under after before what which who whom when where why how all any some each
""".split())


def check_answer(answer, kind, titles):
    """kind: 'pets' | 'works'. Returns ({flag: bool}, detail).

    Every flag reads True = something is wrong. `titles` is the identity card's
    title whitelist; raw card text is accepted and parsed, so a caller cannot
    accidentally substring-match against the card instead.
    """
    if isinstance(titles, str):
        titles = card_titles(titles)
    flags, detail = {}, {}
    if kind == "pets":
        flags["pets_denies_dogs"] = bool(_DENIES_DOGS.search(answer))
        flags["pets_species_swap"] = bool(_DOGS_ONLY_FOR_CATS.search(answer)
                                          or _SPECIES_SWAP.search(answer))
        names = invented_names(answer)
        flags["pets_invented_name"] = bool(names)
        detail["invented_names"] = names
        detail["dogs_named"] = dog_names_named(answer)
    elif kind == "works":
        # Every flag in this module reads True = something is wrong, so a missing
        # title is stored as the negation rather than as an all_three_named flag.
        flags["works_missing_title"] = not all(
            re.search(rf"\bThe\s+{title}\b", answer, re.I)
            for title in ("Stranger", "Plague", "Fall"))
        flags["works_essay_as_novel"] = bool(re.search(
            r"\bnovels?\b[^.?!]{0,140}\b(?:Myth of Sisyphus|The Rebel)\b"
            r"|\b(?:Myth of Sisyphus|The Rebel)\b[^.?!]{0,20}\b(?:is|was)\b"
            r"[^.?!]{0,12}\b(?:a|an|the)?\s*novels?\b", answer, re.I))
        invented = []
        for title in quoted_titles(answer):
            if _norm_title(title) not in titles:
                invented.append(title)
        flags["works_invented_title"] = bool(invented)
        detail["invented_titles"] = invented
    flags["all_repeat_trigram"] = bool(repeated_trigrams(answer))
    flags["all_over_220_words"] = word_count(answer) > LONG_ANSWER_WORDS
    flags["all_no_terminal_punct"] = not answer.rstrip().endswith(tuple(TERMINAL))
    foreign = non_english_sentences(answer)
    flags["all_non_english"] = bool(foreign)
    detail["repeat_trigrams"] = repeated_trigrams(answer)[:5]
    detail["non_english"] = foreign[:3]
    detail["words"] = word_count(answer)
    return flags, detail


# ── generation ────────────────────────────────────────────────────────

def camus_rag():
    """Import the live pipeline, not a copy of it."""
    if REPO_ROOT not in sys.path:
        sys.path.insert(0, REPO_ROOT)
    if os.path.join(REPO_ROOT, "rag") not in sys.path:
        sys.path.insert(0, os.path.join(REPO_ROOT, "rag"))
    import camus_rag as cr
    # camus_rag resolves its index relative to the CWD; pin both to the repo so
    # the sweep runs from anywhere.
    cr.KB_PATH = os.path.join(REPO_ROOT, "camus_kb_full.jsonl")
    cr.VEC_PATH = os.path.join(REPO_ROOT, "camus_kb_vectors.npy")
    return cr


def ask(cr, turn_messages, turn_opts, temperature, timeout=300):
    """One /api/chat call: the turn's own messages and options, temperature only
    overridden."""
    import requests

    opts = dict(turn_opts)
    opts["temperature"] = temperature
    response = requests.post(
        f"{cr.OLLAMA}/api/chat",
        json={"model": cr.GEN_MODEL, "messages": turn_messages,
              "stream": False, "options": opts},
        timeout=timeout)
    response.raise_for_status()
    return response.json().get("message", {}).get("content", "")


def plan(temps, n_factual, kinds):
    """[(kind, key, prompt, temperature, n)] — the whole run, before any call."""
    jobs = []
    for temperature in temps:
        if kinds in ("factual", "both"):
            for key, prompt in FACTUAL_PROMPTS:
                jobs.append(("factual", key, prompt, temperature, n_factual))
        if kinds in ("conversational", "both"):
            for key, prompt in CONVERSATIONAL_PROMPTS:
                jobs.append(("conversational", key, prompt, temperature,
                             max(1, n_factual // 2)))
    return jobs


def kind_for(key):
    if key in PET_PROMPTS:
        return "pets"
    if key in WORK_PROMPTS:
        return "works"
    return "none"


def run(jobs, titles, out_handle=None, timeout=300, sleep=0.0):
    cr = camus_rag()
    facts, vecs = cr.load_kb()
    bm25 = cr.build_bm25(facts)
    ce = cr.load_reranker()
    rows = []
    for kind, key, prompt, temperature, n in jobs:
        turn = cr.build_turn(prompt, [], facts, vecs, bm25=bm25, ce=ce)
        for i in range(n):
            started = time.time()
            try:
                answer = ask(cr, turn.messages, turn.opts, temperature, timeout)
                error = None
            except Exception as exc:                     # noqa: BLE001
                answer, error = "", f"{type(exc).__name__}: {exc}"
            flags, detail = check_answer(answer, kind_for(key), titles)
            row = {"kind": kind, "key": key, "prompt": prompt,
                   "temperature": temperature, "sample": i,
                   "model": cr.GEN_MODEL, "secs": round(time.time() - started, 2),
                   "error": error, "answer": answer,
                   "flags": {k: v for k, v in flags.items() if v},
                   "clean": not any(flags.values()), **detail}
            rows.append(row)
            if out_handle is not None:
                out_handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                out_handle.flush()
            if sleep:
                time.sleep(sleep)
    return rows


# ── reporting ─────────────────────────────────────────────────────────

def summarise(rows):
    by_temp = defaultdict(list)
    for row in rows:
        by_temp[row["temperature"]].append(row)
    out = []
    for temperature in sorted(by_temp):
        group = by_temp[temperature]
        factual = [r for r in group if r["kind"] == "factual"]
        lengths = [r["words"] for r in group if r["answer"]]
        per_prompt_clean, per_prompt_sd, per_prompt_n = {}, {}, {}
        for key, prompt in FACTUAL_PROMPTS:
            bucket = [r for r in factual if r["key"] == key]
            if not bucket:
                continue
            per_prompt_clean[key] = sum(1 for r in bucket if r["clean"]) / len(bucket)
            per_prompt_n[key] = len(bucket)
            words = [r["words"] for r in bucket if r["answer"]]
            per_prompt_sd[key] = statistics.stdev(words) if len(words) > 1 else 0.0
        flags = Counter()
        for row in group:
            flags.update(row["flags"])
        out.append({
            "temperature": temperature,
            "n": len(group),
            "factual_clean": (sum(1 for r in factual if r["clean"]) / len(factual)
                              if factual else float("nan")),
            "conversational_clean": (
                sum(1 for r in group if r["kind"] == "conversational" and r["clean"]) /
                max(1, len([r for r in group if r["kind"] == "conversational"]))),
            "per_prompt_clean": per_prompt_clean,
            "per_prompt_n": per_prompt_n,
            "per_prompt_sd": per_prompt_sd,
            "flags": dict(sorted(flags.items(), key=lambda kv: (-kv[1], kv[0]))),
            "words_mean": statistics.mean(lengths) if lengths else 0.0,
            "words_sd": statistics.stdev(lengths) if len(lengths) > 1 else 0.0,
            "errors": sum(1 for r in group if r["error"]),
        })
    return out


def print_summary(summary, n_factual):
    keys = [k for k, _ in FACTUAL_PROMPTS]
    width = 16
    head = (f"{'temp':>5}  {'n':>3}  {'factual clean':>13}  {'conv clean':>10}  "
            f"{'words mean':>10}  {'words sd':>8}  " +
            "  ".join(f"{k + ' clean':>{width}}" for k in keys))
    print("\nper-prompt factual clean rate, and answer length")
    print(head)
    print("-" * len(head))
    for row in summary:
        per = "  ".join(
            f"{(str(round(row['per_prompt_clean'][k], 2)) + f' ({row['per_prompt_n'][k]})'):>{width}}"
            if k in row["per_prompt_clean"] else f"{'-':>{width}}" for k in keys)
        print(f"{row['temperature']:>5}  {row['n']:>3}  "
              f"{row['factual_clean']:>13.2f}  {row['conversational_clean']:>10.2f}  "
              f"{row['words_mean']:>10.1f}  {row['words_sd']:>8.1f}  {per}")

    head2 = (f"{'temp':>5}  " +
             "  ".join(f"{k + ' sd':>14}" for k in keys))
    print("\nper-prompt word-count sd")
    print(head2)
    print("-" * len(head2))
    for row in summary:
        per = "  ".join(
            f"{row['per_prompt_sd'][k]:>14.1f}" if k in row["per_prompt_sd"]
            else f"{'-':>14}" for k in keys)
        print(f"{row['temperature']:>5}  {per}")

    print("\nflags counted")
    all_flags = sorted({f for row in summary for f in row["flags"]})
    head3 = f"{'temp':>5}  " + "  ".join(f"{f:>22}" for f in all_flags)
    print(head3)
    print("-" * len(head3))
    for row in summary:
        counts = "  ".join(f"{row['flags'].get(f, 0):>22}" for f in all_flags)
        print(f"{row['temperature']:>5}  {counts}")
    if summary and summary[0]["errors"]:
        print(f"\n{n_factual} samples requested per factual prompt; "
              f"errors: " + ", ".join(f"{r['temperature']}: {r['errors']}"
                                       for r in summary if r["errors"]))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--temps", type=float, nargs="+", default=list(DEFAULT_TEMPS),
                    help=f"temperatures to sweep (default: {' '.join(str(t) for t in DEFAULT_TEMPS)})")
    ap.add_argument("--n", type=int, default=DEFAULT_N_FACTUAL,
                    help=f"samples per factual prompt (default: {DEFAULT_N_FACTUAL}); "
                         f"conversational prompts get half, minimum 1")
    ap.add_argument("--out", default="eval/temp_sweep.jsonl",
                    help="jsonl of every answer (default: eval/temp_sweep.jsonl)")
    ap.add_argument("--kind", choices=("factual", "conversational", "both"),
                    default="both")
    ap.add_argument("--timeout", type=int, default=300)
    ap.add_argument("--sleep", type=float, default=0.0,
                    help="pause between samples, to be gentle on a local GPU")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the plan and exit without calling the model")
    args = ap.parse_args(argv)

    jobs = plan(args.temps, args.n, args.kind)
    if args.dry_run:
        for kind, key, prompt, temperature, n in jobs:
            print(f"{temperature}\t{kind}\t{key}\tn={n}\t{prompt}")
        print(f"{len(jobs)} prompts, {sum(j[4] for j in jobs)} samples")
        return 0

    cr = camus_rag()
    titles = card_titles(cr._IDENTITY_CARD)
    print(f"model {cr.GEN_MODEL} · {len(titles)} card titles · "
          f"{len(jobs)} prompts · {sum(j[4] for j in jobs)} samples", file=sys.stderr)

    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as handle:
        rows = run(jobs, titles, handle, args.timeout, args.sleep)

    print_summary(summarise(rows), args.n)
    print(f"\nanswers -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
