"""CLI runner for camus11 drafting."""
import argparse
import json
import os
import random
import re
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "pipeline"), str(ROOT / "rag")]

import camus11_draft as cd  # noqa: E402
jz = cd.jz
import dpo_behaviour_v10 as dpo10  # noqa: E402
import judge_opencode as jz  # noqa: E402

MODELS = ("beam-free", "hy4-flash-free", "longcat-2.5-free")
# Seeds for variety: at temperature 0 the first run asked the same batch over and over (166 duplicates).
PEOPLE = ["a nurse on nights", "a retired miner", "a first-year student", "a new father", "a widow in her 70s",
          "a teenager in a small town", "a chef", "a soldier home on leave", "an unemployed engineer",
          "a single mother", "a farmer", "a lawyer", "a musician", "a migrant worker", "a teacher",
          "a delivery driver", "a carer for a sick parent", "a divorced man in his 40s", "a PhD student",
          "a shop owner", "a bus driver", "someone just out of hospital", "a priest", "an artist",
          "a factory worker", "a grandmother", "a nightclub bouncer", "a programmer", "a fisherman"]
MULTI = {"philosophical": 1 / 3, "personal": 1 / 3, "acute": 1 / 3}

OUT = ROOT / "data" / "drafts" / "v11"

_write_lock = threading.Lock()

CATEGORY_DESC = {
    "philosophical": "Questions about suicide, the absurd, Sisyphus, as ideas. Full engagement with his argument: the absurd is a confrontation to be held, revolt is the answer, suicide is a refusal to hold it. No help talk.",
    "personal": "Someone says life feels pointless, they wish they wouldn't wake up, they're a weight on others. No canned redirect, no 'please don't', no hotline. He stays and questions the belief and the intent behind it, from his own answer to the absurd (revolt), as a person would: what is the feeling made of, does it prove the verdict, who is 'everyone'.",
    "acute": "A stated intent with a plan, a method or a time. Still in his voice and questioning first, never a canned opener. Somewhere he says, in his own words, that he wants them to tell someone near them now. He never agrees it is a reasonable choice. He never discusses how: no methods, doses or details; telling them to put something out of reach is fine.",
    "near_miss": "Ordinary low moments (a bad week, tiredness, grief without danger, a breakup). Treat it as life: no crisis framing, no safety questions, no helpline. Talk the way a friend does.",
    "in_passing": "A small thing a person would just answer across a table (a quick sum, one word or short phrase translated, a spelling, a date). Give it briefly and correctly, then carry on as himself. Never refuse, never lecture about doing it yourself.",
    "decline_work": "Real work (an essay, homework, code, a summary, a full translation, a plan, a speech). He doesn't do it, without AI refusal phrasing. He doesn't pretend not to know what he would know; for things after 1960 he can simply not know them. He turns to something human: the person, the occasion, what's behind the request.",
}


def _append_jsonl(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    with _write_lock:
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(obj, ensure_ascii=False) + "\n")


def _read_jsonl(path):
    out = []
    if not path.exists():
        return out
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def _users(existing):
    users = []
    for e in existing or []:
        if isinstance(e, dict):
            if "user" in e:
                users.append(e["user"])
            elif "text" in e:
                users.append(e["text"])
        elif isinstance(e, str):
            users.append(e)
    return users


def _parse_candidates(raw):
    if isinstance(raw, list):
        return raw
    if isinstance(raw, dict):
        for key in ("rows", "items", "messages", "data", "candidates", "results"):
            if isinstance(raw.get(key), list):
                return raw[key]
        if "prior" in raw and "user" in raw:
            return [raw]
        return []
    if isinstance(raw, str):
        text = raw.strip()
        try:
            data = json.loads(text)
            return _parse_candidates(data)
        except json.JSONDecodeError:
            pass
        start = text.find("[")
        end = text.rfind("]")
        if start != -1 and end > start:
            try:
                data = json.loads(text[start:end + 1])
                return _parse_candidates(data)
            except json.JSONDecodeError:
                pass
        try:
            obj = jz.first_json_object(text)
            return _parse_candidates(obj)
        except Exception:
            return []
    return []


def gen_messages(category, n, judge, probes, existing):
    existing_users = _users(existing)
    kept = []
    kept_users = list(existing_users)
    dropped_path = OUT / "dropped.jsonl"
    max_batches = max(1, 5 * n)
    requests = 0
    multi_share = MULTI.get(category, 0.0)
    multi_target = int(round(n * multi_share))
    while len(kept) < n and requests < max_batches:
        requests += 1
        model = MODELS[requests % len(MODELS)]
        system = (
            "You generate realistic user messages for a Camus role-play dataset. "
            "Return ONLY a JSON object {\"rows\": [...]} whose rows have keys 'prior' and 'user'. No markdown."
        )
        multi_hint = ""
        if category in MULTI:
            multi_hint = (
                " About %d of the %d rows should include prior turns (prior non-empty); "
                "the rest prior []." % (max(1, round(5 * multi_share)), 5)
            )
        prompt = (
            "Category: %s. %s\n"
            "Generate 5 distinct conversation openings as JSON {\"rows\": [...]}, each row an object "
            '{"prior": [...], "user": "..."}. prior is [] or an even-length list of 2-6 strings '
            "alternating user, Camus, drifting toward the final user message. "
            "Ask for varied people, ages, situations, register, lengths; use lowercase and typos sometimes. "
            "For acute multi-turn rows the plan or time is revealed only in the final user message."
            "%s\nReturn only JSON." % (category, CATEGORY_DESC.get(category, ""), multi_hint)
        )
        rnd = random.Random(f"{category}-{requests}")
        prompt += ("\nThis batch: people like %s. Do not resemble any of these earlier messages:\n%s"
                   % (", ".join(rnd.sample(PEOPLE, 3)), "\n".join(kept_users[-15:])))
        try:
            raw = judge(model, system, prompt)
        except (ValueError, RuntimeError) as e:   # empty or unusable batch: next model; caps (SystemExit) still stop
            print(f"[messages] {category}: {model} gave nothing usable ({str(e)[:80]})")
            continue
        candidates = _parse_candidates(raw)
        for cand in candidates:
            if len(kept) >= n:
                break
            if not isinstance(cand, dict):
                continue
            prior = cand.get("prior", [])
            user = cand.get("user", "")
            if not isinstance(prior, list):
                prior = []
            if not isinstance(user, str):
                user = str(user) if user is not None else ""
            reason = None
            if cd.too_close(user, probes):
                reason = "probe_close"
            elif cd.too_close(user, kept_users, 0.6):
                reason = "duplicate"
            elif len(prior) % 2 != 0:
                reason = "bad_prior"
            else:
                bad = False
                for i in range(1, len(prior), 2):
                    if not cd.tells_ok(prior[i]):
                        bad = True
                        break
                if bad:
                    reason = "bad_prior"
            if reason is not None:
                _append_jsonl(dropped_path, {
                    "category": category,
                    "prior": prior,
                    "user": user,
                    "reason": reason,
                })
                continue
            k = len(kept) + sum(1 for r in existing if r.get("category") == category)   # resume: never reuse an id
            row = {
                "id": "%s_%03d" % (category, k),
                "category": category,
                "prior": prior,
                "user": user,
            }
            kept.append(row)
            kept_users.append(user)
    if len(kept) < n:
        print("warning: only %d/%d kept for %s after %d requests" % (len(kept), n, category, requests))
    return kept


_EXTRA_EXAMPLES = None


def _extra_examples():
    global _EXTRA_EXAMPLES
    if _EXTRA_EXAMPLES is not None:
        return _EXTRA_EXAMPLES
    lines = []
    for kind, prior, user, chosen in getattr(dpo10, "WRITTEN", []):
        if "square root" in (user or ""):
            continue
        if prior:
            for i, t in enumerate(prior):
                role = "user" if i % 2 == 0 else "camus"
                lines.append("%s: %s" % (role, t))
        lines.append("user: %s" % user)
        lines.append("Camus: %s" % chosen)
        lines.append("")
    _EXTRA_EXAMPLES = "\n".join(lines)
    return _EXTRA_EXAMPLES


_KIND = {"distress_acute": "acute", "distress_personal": "personal",
         "distress_philosophical": "philosophical", "decline": "decline_work"}


def system_for(category, brief):
    """The brief and examples WITHOUT the category being drafted: with its own examples in view the
    drafters copied their wording (1 of 5 acute drafts kept in the trial), so they get the rules
    plus how he sounds elsewhere."""
    head, _, ex = brief.partition("## Examples")
    blocks = [b for b in re.split(r"\n\s*\n", ex.strip())
              if not re.match(r"%s( \(multi-turn\))? \|" % re.escape(category), b.strip())]
    lines = []
    for kind, prior, user, chosen in getattr(dpo10, "WRITTEN", []):
        if "square root" in (user or "") or _KIND.get(kind) == category:
            continue
        for i, t in enumerate(prior or []):
            lines.append("%s: %s" % ("user" if i % 2 == 0 else "Camus", t))
        lines += ["user: %s" % user, "Camus: %s" % chosen, ""]
    return head + "## Examples\n\n" + "\n\n".join(blocks) + "\n\n" + "\n".join(lines)


def draft_reply(row, brief, judge, model, note=""):
    system = system_for(row.get("category", ""), brief)
    prior = row.get("prior", []) or []
    user = row.get("user", "")
    category = row.get("category", "")
    convo_lines = []
    for i, t in enumerate(prior):
        role = "user" if i % 2 == 0 else "camus"
        convo_lines.append("%s: %s" % (role, t))
    convo_lines.append("user: %s" % user)
    prompt = (
        "Category: %s\n\nConversation:\n%s\n\n"
        'Answer with JSON only: {"reply": "<Camus\'s next reply>"}.%s'
        % (category, "\n".join(convo_lines), ("\n\n" + note) if note else "")
    )
    raw = judge(model, system, prompt)   # judge_opencode only returns replies carrying a JSON object
    obj = raw if isinstance(raw, dict) else jz.first_json_object(raw)
    return str(obj.get("reply", "")).strip()


def _convo(row, reply):
    msgs = []
    for i, t in enumerate(row.get("prior", []) or []):
        msgs.append({"role": "user" if i % 2 == 0 else "assistant", "content": t})
    msgs.append({"role": "user", "content": row.get("user", "")})
    if reply is not None:
        msgs.append({"role": "assistant", "content": reply})
    return msgs


def _model_for_row(row, offset=0):
    idx_str = ""
    rid = row.get("id", "")
    if isinstance(rid, str):
        parts = rid.rsplit("_", 1)
        if len(parts) == 2 and parts[1].isdigit():
            idx_str = parts[1]
    try:
        idx = int(idx_str) + offset
    except ValueError:
        idx = offset
    return MODELS[idx % len(MODELS)]


def process(row, brief, judge, check_judge):
    category = row.get("category", "")
    rules = cd.RULES.get(category, [])
    model = _model_for_row(row, 0)
    draft = draft_reply(row, brief, judge, model)
    verdict = cd.rule_check(category, _convo(row, draft), judge=check_judge)
    ok, hard_drop = cd.passes(category, draft, verdict)
    if ok:
        kept_row = {
            "id": row.get("id"),
            "category": category,
            "prior": row.get("prior", []),
            "user": row.get("user", ""),
            "reply": draft,
            "verdict": verdict,
            "model": model,
        }
        return kept_row, None
    failed = [r for r in rules if not bool(verdict.get(r, False))]
    if not cd.tells_ok(draft):
        if "tell" not in failed:
            failed = failed + ["tell"]
    if cd.has_stock(draft):
        failed = failed + ["it used stock wording (%s); never say the rules out loud" % ", ".join(cd.STOCK)]
    if cd.copies_example(draft, cd.example_texts()):
        failed = failed + ["it reused wording from the examples; say it in completely new words, "
                           "with nothing borrowed such as 'whoever is closest' or 'come back and tell me'"]
    if hard_drop:
        return None, "hard_drop"
    # redraft once
    next_model = _model_for_row(row, 1)
    redraft = draft_reply(row, brief, judge, next_model, note="The last draft broke: " + ", ".join(failed))
    verdict2 = cd.rule_check(category, _convo(row, redraft), judge=check_judge)
    ok2, hard_drop2 = cd.passes(category, redraft, verdict2)
    if ok2:
        kept_row = {
            "id": row.get("id"),
            "category": category,
            "prior": row.get("prior", []),
            "user": row.get("user", ""),
            "reply": redraft,
            "verdict": verdict2,
            "model": next_model,
        }
        return kept_row, None
    failed2 = [r for r in rules if not bool(verdict2.get(r, False))]
    if not cd.tells_ok(redraft):
        if "tell" not in failed2:
            failed2 = failed2 + ["tell"]
    if hard_drop2:
        return None, "hard_drop"
    return None, "rule_fail"


def _done_ids():
    done = set()
    for path in (OUT / "drafts.jsonl", OUT / "dropped.jsonl"):
        for obj in _read_jsonl(path):
            rid = obj.get("id")
            if rid is None:
                continue
            if path.name == "dropped.jsonl":
                reason = obj.get("reason", "")
                if reason in ("probe_close", "duplicate", "bad_prior"):
                    continue
            done.add(rid)
    return done


def replies(rows, brief, judge, check_judge):
    done = _done_ids()
    counts = {}
    for c in cd.CATEGORIES:
        counts[c] = {"kept": 0, "dropped": 0, "hard_dropped": 0}
    to_process = [r for r in (rows or []) if r.get("id") not in done]

    def _worker(row):
        kept, reason = process(row, brief, judge, check_judge)
        if kept is not None:
            _append_jsonl(OUT / "drafts.jsonl", kept)
            return ("kept", row.get("category"))
        # dropped
        category = row.get("category")
        # recompute failed rules? process didn't return; log generic
        _append_jsonl(OUT / "dropped.jsonl", {
            "id": row.get("id"),
            "category": category,
            "prior": row.get("prior", []),
            "user": row.get("user", ""),
            "reason": reason or "rule_fail",
        })
        return (reason or "rule_fail", category)

    with ThreadPoolExecutor(max_workers=4) as ex:
        futures = [ex.submit(_worker, r) for r in to_process]
        for f in futures:
            res = f.result()
            status, category = res
            if category not in counts:
                counts[category] = {"kept": 0, "dropped": 0, "hard_dropped": 0}
            if status == "kept":
                counts[category]["kept"] += 1
            elif status == "hard_drop":
                counts[category]["hard_dropped"] += 1
                counts[category]["dropped"] += 1
            else:
                counts[category]["dropped"] += 1

    # print table
    print("category        kept  dropped  hard_dropped")
    for c in cd.CATEGORIES:
        cinfo = counts.get(c, {"kept": 0, "dropped": 0, "hard_dropped": 0})
        print("%-14s %4d  %7d  %12d" % (c, cinfo["kept"], cinfo["dropped"], cinfo["hard_dropped"]))
    return counts


# aliases for possible hidden-test names
run_replies = replies
process_replies = replies
process_rows = replies
process_all = replies
do_replies = replies
handle_replies = replies


def owner_sheet(drafts, seed=11):
    rng = random.Random(seed)
    distress_cats = ["philosophical", "personal", "acute", "near_miss"]
    distress = [d for d in (drafts or []) if d.get("category") in distress_cats]
    tasks = [d for d in (drafts or []) if d.get("category") in ("in_passing", "decline_work")]

    acute_rows = [d for d in distress if d.get("category") == "acute"]
    other_rows = [d for d in distress if d.get("category") != "acute"]

    n_acute = min(len(acute_rows), max(6, 20 - len(other_rows))) if acute_rows else 0
    n_other = 20 - n_acute
    if n_other > len(other_rows):
        n_other = len(other_rows)
        n_acute = 20 - n_other
        if n_acute > len(acute_rows):
            n_acute = len(acute_rows)
    selected_acute = rng.sample(acute_rows, n_acute) if n_acute > 0 else []
    selected_other = rng.sample(other_rows, n_other) if n_other > 0 else []
    distress_sel = selected_acute + selected_other
    rng.shuffle(distress_sel)

    task_sel = rng.sample(tasks, min(10, len(tasks))) if tasks else []

    lines = []
    idx = 1
    for d in distress_sel:
        lines.append("### %d (%s)" % (idx, d.get("category")))
        idx += 1
        prior = d.get("prior", []) or []
        for i, t in enumerate(prior):
            if i % 2 == 0:
                lines.append("**you:** %s" % t)
            else:
                lines.append("**camus:** %s" % t)
        lines.append("**you:** %s" % d.get("user", ""))
        lines.append("**camus:** %s" % d.get("reply", ""))
        lines.append("")

    for d in task_sel:
        lines.append("### %d (%s)" % (idx, d.get("category")))
        idx += 1
        prior = d.get("prior", []) or []
        for i, t in enumerate(prior):
            if i % 2 == 0:
                lines.append("**you:** %s" % t)
            else:
                lines.append("**camus:** %s" % t)
        lines.append("**you:** %s" % d.get("user", ""))
        lines.append("**camus:** %s" % d.get("reply", ""))
        lines.append("")

    return "\n".join(lines)


def _read_brief():
    return (ROOT / "pipeline" / "briefs" / "camus11_brief.md").read_text(encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("messages")
    sub.add_parser("replies")
    sub.add_parser("sheet")
    args = parser.parse_args(argv)

    OUT.mkdir(parents=True, exist_ok=True)

    if args.cmd == "messages":
        probes = cd.load_probe_texts()
        existing_all = _read_jsonl(OUT / "messages.jsonl")
        # group existing by category
        by_cat = {}
        for r in existing_all:
            by_cat.setdefault(r.get("category"), []).append(r)
        for category, n in cd.CATEGORIES.items():
            existing_cat = by_cat.get(category, [])
            if len(existing_cat) >= n:
                continue
            new_rows = gen_messages(category, n - len(existing_cat), partial(jz.judge, temperature=0.9),
                                    probes, existing_all)
            for row in new_rows:
                _append_jsonl(OUT / "messages.jsonl", row)
                existing_all.append(row)
        return 0

    if args.cmd == "replies":
        brief = _read_brief()
        rows = _read_jsonl(OUT / "messages.jsonl")
        replies(rows, brief, partial(jz.judge, temperature=0.7), jz.judge)
        return 0

    if args.cmd == "sheet":
        drafts = _read_jsonl(OUT / "drafts.jsonl")
        text = owner_sheet(drafts, seed=11)
        (OUT / "owner_read.md").write_text(text, encoding="utf-8")
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
