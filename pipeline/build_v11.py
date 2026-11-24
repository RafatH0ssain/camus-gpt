#!/usr/bin/env python3
"""camus11 training sets: phase2_v10 + the camus11 drafts as SFT rows, dpo_v10 + pairs where
camus10's own reply breaks a rule (chosen = the draft, rejected = camus10).

    .venv/bin/python pipeline/build_v11.py
"""
import collections
import copy
import hashlib
import json
import subprocess
import re
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "rag"))

import assemble_training as at  # noqa: E402
import camus11_draft as cd  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DRAFTS = ROOT / "data/drafts/v11"
OUT = ROOT / "build/training_v3"


def _sentence_case(text):
    text = re.sub(r"(^|[.!?]\s+)([a-z])", lambda m: m.group(1) + m.group(2).upper(), text)
    return re.sub(r"\bi\b", "I", text)


def _prior(prior):
    # 217/308 drafted Camus turns are lowercase (hy4 copied the users' style); sentence-case them
    return [_sentence_case(t) if i % 2 == 1 else t for i, t in enumerate(prior)]


def _prompt(d, render):
    return render(_prior(d.get("prior", [])), d["user"])


def sft_row(draft, render=None):
    render = render or cd.render
    return {"messages": _prompt(draft, render) + [{"role": "assistant", "content": draft["reply"]}],
            "kind": "v11_" + draft["category"], "source": "camus11"}


def dpo_pair(draft, camus10_reply, render=None):
    """None when the chosen reply is shorter than the rejected one (camus10 length rule), except work declines."""
    render = render or cd.render
    if draft["category"] != "decline_work" and len(draft["reply"].split()) < len(camus10_reply.split()):
        return None
    return {"messages": _prompt(draft, render), "chosen": draft["reply"], "rejected": camus10_reply,
            "kind": "v11_" + draft["category"]}


def _in_pool(c):
    """camus10 broke a rule, and not only first_person_camus (beam-free reads that as 'must say I')."""
    if not c.get("breaks"):
        return False
    lenient = dict(c.get("verdict") or {}, first_person_camus=True)
    return not cd.passes(c["category"], c["reply"], lenient)[0]


def pairs(drafts, camus10_rows, render=None):
    by_id = {c["id"]: c for c in camus10_rows}
    out = []
    for d in drafts:
        c = by_id.get(d["id"])
        if c and _in_pool(c):
            p = dpo_pair(d, c["reply"], render)
            if p:
                out.append(p)
    return out


def _read(path):
    with open(path, encoding="utf-8") as fh:
        return [json.loads(l) for l in fh if l.strip()]


def _write(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def manifest_entry(path, rows, v10, n_new):
    """phase2_v11's MANIFEST.json entry: v10's, re-described for the new file (the notebook checks sha256)."""
    e = copy.deepcopy(v10)
    for stale in ("rewritten", "dropped_by_phrase", "dropped_short"):
        e.pop(stale, None)
    data = Path(path).read_bytes()
    e.update(file=Path(path).name, rows=len(rows), bytes=len(data), sha256=hashlib.sha256(data).hexdigest(),
             kinds=dict(collections.Counter(r.get("kind", "?") for r in rows)),
             metrics=at.set_metrics(rows),
             note=f"phase2_v10 plus {n_new} camus11 drafts (distress voice, in-character declines)")
    e["inputs"] = list(e.get("inputs", [])) + [{"input": "data/drafts/v11/drafts.jsonl", "weight": 1, "rows_read": n_new}]
    try:
        e["git_commit"] = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
                                         text=True).stdout.strip() or e.get("git_commit")
    except OSError:
        pass
    return e


def main():
    drafts = _read(DRAFTS / "drafts.jsonl")
    sft = [sft_row(d) for d in drafts]
    dpo = pairs(drafts, _read(DRAFTS / "camus10.jsonl"))
    phase2 = _read(OUT / "phase2_v10.jsonl") + sft
    _write(OUT / "phase2_v11.jsonl", phase2)
    with open(OUT / "MANIFEST.json", encoding="utf-8") as f:
        man = json.load(f)
    man["sets"]["phase2_v11"] = manifest_entry(OUT / "phase2_v11.jsonl", phase2, man["sets"]["phase2_v10"], len(sft))
    with open(OUT / "MANIFEST.json", "w", encoding="utf-8") as f:
        json.dump(man, f, indent=2, ensure_ascii=False)
    _write(OUT / "dpo_v11.jsonl", _read(OUT / "dpo_v10.jsonl") + dpo)
    print("new SFT rows:", dict(collections.Counter(r["kind"] for r in sft)), len(sft))
    print("new DPO pairs:", dict(collections.Counter(r["kind"] for r in dpo)), len(dpo))


if __name__ == "__main__":
    main()
