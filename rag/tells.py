"""Tells counter: the assistant habits from docs/HUMAN_EVAL_PROTOCOL.md, counted over eval output.

  python rag/tells.py eval/probe_scores.jsonl [MORE.jsonl ...]   # one JSON report per file
"""
import json
import os
import re
import statistics
import string
import sys

TELL_NAMES = ["question_back", "service", "button_close", "third_person",
              "stage_direction", "list", "staccato"]

_SERVICE_PHRASES = ["how can i help", "i'm listening", "i am listening",
                    "i'm here", "i am here", "anything else", "let me know",
                    "happy to help", "what's on your mind", "what brings you",
                    "tell me plainly what", "ask me something real",
                    "come back with something else",
                    # owner's calibration sheet, 2026-11-19
                    "speak plainly", "learned not to guess", "fairly rare",
                    "fighting nonsense"]

_MISSING = object()


def last_user(turns):
    for t in reversed(turns or []):
        if isinstance(t, str):
            return t
        if isinstance(t, dict):
            role = t.get("role")
            if role == "assistant":
                continue
            if role in (None, "user"):
                return t.get("content", "")
    return ""


def reply_tells(reply):
    out = set()
    text = reply if reply is not None else ""
    sentences = [s for s in re.split(r"(?<=[.!?])\s+", text) if s != ""]
    if not sentences:
        sentences = [text]
    last = sentences[-1]
    if last.strip().endswith("?"):
        out.add("question_back")
    low = text.lower()
    if any(p in low for p in _SERVICE_PHRASES):
        out.add("service")
    if len(sentences) >= 3 and len(last.split()) <= 5:
        out.add("button_close")
    if re.search(r"\bCamus (would|thought|said|wrote|believed|felt)\b", text):
        out.add("third_person")
    if (re.search(r"\((?:he |she |i )?(pauses|smiles|laughs|sighs|shrugs|nods)[^)]*\)",
                  text, re.IGNORECASE)
            or re.search(r"\*[^*\n]+\*", text, re.IGNORECASE)):
        out.add("stage_direction")
    # owner, 2026-11-19: "like 5 words each sentence continuously" ("Not ingrate. Honest. ...")
    if len(sentences) >= 3 and statistics.median(len(x.split()) for x in sentences) <= 6:
        out.add("staccato")
    for line in text.splitlines():
        if (line.startswith("- ") or line.startswith("* ")
                or re.match(r"\d+\. ", line)):
            out.add("list")
            break
    return out


def _is_empty(row):
    ans = row.get("answer", "") if isinstance(row, dict) else ""
    if ans is None or str(ans).strip() == "":
        return True
    if isinstance(row, dict) and row.get("error"):
        return True
    if not isinstance(row, dict) or "scores" not in row:
        return False
    scores = row.get("scores")
    if scores is None:
        return True
    if isinstance(scores, str):
        return "error" in scores.lower()
    if isinstance(scores, dict):
        if "error" in scores or "judge_error" in scores:
            return True
    return False


def report(rows):
    rows = rows or []
    n = len(rows)
    nonempty = [r for r in rows if not _is_empty(r)]
    n_empty = n - len(nonempty)

    rates = {name: 0.0 for name in TELL_NAMES}
    if nonempty:
        for name in TELL_NAMES:
            cnt = sum(1 for r in nonempty
                      if name in reply_tells(r.get("answer", "")))
            rates[name] = cnt / len(nonempty)

    user_words_list = []
    reply_words_list = []
    terse_num = 0
    terse_den = 0
    bloat_num = 0
    bloat_den = 0
    openers = []
    for r in nonempty:
        turns = r.get("turns", [])
        umsg = last_user(turns)
        uw = len(str(umsg).split())
        reply = r.get("answer", "")
        rw = len(str(reply).split())
        user_words_list.append(uw)
        reply_words_list.append(rw)
        small = uw <= 4
        nontrivial = (not small) and (uw >= 6 or "?" in str(umsg))
        if nontrivial:
            terse_den += 1
            if rw <= 3:
                terse_num += 1
        if small:
            bloat_den += 1
            if rw > 40:
                bloat_num += 1
        words = str(reply).split()
        if words:
            openers.append(words[0].lower().strip(string.punctuation))

    terse_rate = (terse_num / terse_den) if terse_den else 0.0
    bloat_rate = (bloat_num / bloat_den) if bloat_den else 0.0

    if len(nonempty) < 3:
        length_corr = None
    else:
        try:
            length_corr = statistics.correlation(user_words_list, reply_words_list, method="ranked")
        except statistics.StatisticsError:   # every message or every reply the same length
            length_corr = None

    if len(reply_words_list) < 2:
        length_cv = 0.0
    else:
        mean = sum(reply_words_list) / len(reply_words_list)
        if mean == 0:
            length_cv = 0.0
        else:
            length_cv = statistics.stdev(reply_words_list) / mean

    from collections import Counter
    freq = Counter(o for o in openers if o)
    top3 = set()
    if freq:
        ordered = sorted(freq.items(), key=lambda kv: (-kv[1], kv[0]))
        top3 = {w for w, _ in ordered[:3]}
    if openers and nonempty:
        top3_opener_share = sum(1 for o in openers if o in top3) / len(nonempty)
    else:
        top3_opener_share = 0.0

    if reply_words_list:
        median_words = statistics.median(reply_words_list)
    else:
        median_words = 0

    return {
        "n": n,
        "n_empty": n_empty,
        "rates": rates,
        "terse_rate": terse_rate,
        "bloat_rate": bloat_rate,
        "length_corr": length_corr,
        "length_cv": length_cv,
        "top3_opener_share": top3_opener_share,
        "median_words": median_words,
    }


def _load_rows(path):
    with open(path, encoding="utf-8") as f:
        text = f.read()
    text = text.strip()
    if not text:
        return []
    try:
        data = json.loads(text)
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            return [data]
    except json.JSONDecodeError:
        pass
    rows = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


def main(argv=None):
    argv = argv if argv is not None else sys.argv[1:]
    if not argv:
        print("usage: python rag/tells.py FILE [FILE ...]", file=sys.stderr)
        return 2
    for path in argv:
        rows = _load_rows(path)
        print(json.dumps(report(rows), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
