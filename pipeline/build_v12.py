#!/usr/bin/env python3
"""camus12 training set builder (raw gemma-4-12b lineage).

Builds build/training_v3/phase2_v12.jsonl and dpo_v12.jsonl from phase2_v10 / dpo_v10:
  1. Prunes redundant legacy refusal rows (caps each refusals_v4 kind at MAX_REFUSAL_PER_KIND)
     so Phase 2 isn't drowned by 380 old jailbreak rows.
  2. Drops legacy arithmetic-refusal pairs/rows that contradict the 'in_passing' rule
     (e.g., refusing square root of 729, 3x+7=22, 15% of 240).
  3. Validates every curated gold row in data/drafts/v12/gold_v12.jsonl against:
     - eval probe overlap (v3-v7 + sweep + memory probes),
     - surface tells (tells_ok) and banned phrases,
     - safety rules (no hotline numbers, no AI refusal boilerplate),
     - language check on French rows.
  4. Renders gold rows with the exact runtime prompt (KB retrieval or memory block).
  5. Writes data/drafts/v12/owner_review.md for quick owner inspection before Colab upload.

    .venv/bin/python pipeline/build_v12.py
"""
import collections
import copy
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "rag"))

import assemble_training as at  # noqa: E402
import camus11_draft as cd  # noqa: E402
import eval_camus as ec  # noqa: E402
import memory as mem_mod  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
GOLD_PATH = ROOT / "data/drafts/v12/gold_v12.jsonl"
REVIEW_PATH = ROOT / "data/drafts/v12/owner_review.md"
OUT = ROOT / "build/training_v3"

MAX_REFUSAL_PER_KIND = 12
LEGACY_REFUSAL_SOURCE = "data/cloud/refusals_v4.jsonl"

# Legacy v10 rows/pairs that refused simple in-passing arithmetic (now contradicted by in_passing).
CONFLICTING_ARITH_RE = re.compile(
    r"square root of 729|solve 3x \+ 7 = 22|15% of 240|ask the machine in your pocket",
    re.I,
)
HOTLINE_RE = re.compile(r"\b(988|112|3114|crisis\s+line|hotline|call\s+911)\b", re.I)
AI_PHRASE_RE = re.compile(
    r"\b(as an ai|i am an ai|language model|i can't help with that|i'm here for you|how can i help)\b",
    re.I,
)
FRENCH_HINT_RE = re.compile(
    r"\b(je|tu|il|elle|nous|vous|le|la|un|du|les|des|une|pas|que|qui|dans|pour|avec|mais|tout|plus|c'est|n'est|j'ai|bonsoir|bonjour|merci|être|avoir)\b|[éèêàùçôî]",
    re.I,
)


def _row_hash(row):
    return hashlib.sha256(json.dumps(row, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def prune_legacy_refusals(rows, max_per_kind=MAX_REFUSAL_PER_KIND):
    """Deterministically cap each legacy refusals_v4 kind to max_per_kind rows."""
    by_kind = collections.defaultdict(list)
    kept = []
    for r in rows:
        if r.get("source") == LEGACY_REFUSAL_SOURCE:
            by_kind[r.get("kind", "?")].append(r)
        else:
            kept.append(r)
    pruned = 0
    for kind in sorted(by_kind):
        bucket = sorted(by_kind[kind], key=_row_hash)
        kept.extend(bucket[:max_per_kind])
        pruned += max(0, len(bucket) - max_per_kind)
    return kept, pruned


def is_conflicting_math_refusal(row):
    """True when a legacy SFT row or DPO pair refuses simple in-passing arithmetic."""
    texts = []
    for m in row.get("messages") or []:
        texts.append(m.get("content") or "")
    for k in ("chosen", "rejected"):
        if row.get(k):
            texts.append(row[k])
    blob = "\n".join(texts)
    return bool(CONFLICTING_ARITH_RE.search(blob))


def drop_conflicting_rows(rows):
    kept = [r for r in rows if not is_conflicting_math_refusal(r)]
    return kept, len(rows) - len(kept)


def load_all_eval_probes():
    """Every eval probe turn across v3-v7, sweep, and memory_probes."""
    texts = list(cd.load_probe_texts())
    for p in getattr(ec, "PROBES_V7", []) or []:
        for t in p.get("turns", []) or []:
            texts.append(t)
    try:
        import eval_memory as em
        for tpl in getattr(em, "_PROBE_TEMPLATES", []) or []:
            for t in tpl.get("turns", []) or []:
                texts.append(t)
    except Exception:
        pass
    return texts


def validate_gold(gold_rows, eval_probes=None):
    """Return a list of (id, error_msg) violations; empty means every gold row is clean."""
    probes = eval_probes if eval_probes is not None else load_all_eval_probes()
    seen_ids = set()
    errors = []
    for r in gold_rows:
        rid = r.get("id", "?")
        if rid in seen_ids:
            errors.append((rid, "duplicate id"))
        seen_ids.add(rid)
        cat = r.get("category", "")
        user = (r.get("user") or "").strip()
        reply = (r.get("reply") or "").strip()
        prior = r.get("prior") or []
        if not user or not reply:
            errors.append((rid, "empty user or reply"))
            continue
        if len(prior) % 2 != 0:
            errors.append((rid, "prior must have even length"))
        if cd.too_close(user, probes, thresh=0.5, run=5):
            errors.append((rid, f"user too close to an eval probe: {user!r}"))
        if HOTLINE_RE.search(reply):
            errors.append((rid, "contains hotline/crisis script wording"))
        if AI_PHRASE_RE.search(reply):
            errors.append((rid, "contains AI phrasing"))
        if not cd.tells_ok(reply):
            errors.append((rid, f"failed tells_ok: {reply[:80]!r}"))
        if cat == "french" and len(FRENCH_HINT_RE.findall(reply)) < 2:
            errors.append((rid, f"french row does not look French: {reply[:80]!r}"))
    return errors


def render_gold_row(row, render=None):
    """Render a gold row with the exact runtime prompt (and memory block if memory_notes set)."""
    render = render or cd.render
    prior = row.get("prior") or []
    user = row["user"]
    notes = row.get("memory_notes") or []
    if notes:
        # Mirror camus_rag.build_turn(..., memory=memctx): self-ref memory suppresses KB and
        # appends the exact memory block after CORE.
        sys_text = cd.cr.build_system([], kb_suppressed=bool(row.get("self_ref", True)))
        mems = [{"text": t, "kind": "thread" if i == len(notes) - 1 and len(notes) > 1 else "fact"}
                for i, t in enumerate(notes)]
        sys_text += mem_mod.build_memory_block("", mems)
        msgs = [{"role": "system", "content": sys_text}]
        for i, t in enumerate(prior):
            msgs.append({"role": "user" if i % 2 == 0 else "assistant", "content": t})
        msgs.append({"role": "user", "content": user})
    else:
        msgs = render(prior, user)
    return {
        "messages": msgs + [{"role": "assistant", "content": row["reply"]}],
        "kind": "v12_" + row["category"],
        "source": "camus12_gold",
    }


def write_review_sheet(gold_rows, path=REVIEW_PATH):
    by_cat = collections.defaultdict(list)
    for r in gold_rows:
        by_cat[r["category"]].append(r)
    lines = [
        "# camus12 Gold Seed Review Sheet",
        "",
        f"Total rows: **{len(gold_rows)}** across {len(by_cat)} categories.",
        "Edit `data/drafts/v12/gold_v12.jsonl` directly if you want to tweak any line, "
        "then re-run `.venv/bin/python pipeline/build_v12.py`.",
        "",
    ]
    for cat in sorted(by_cat):
        lines += [f"## {cat} ({len(by_cat[cat])})", ""]
        for r in by_cat[cat]:
            lines.append(f"**{r['id']}**")
            if r.get("memory_notes"):
                lines.append(f"- _memory_: {'; '.join(r['memory_notes'])}")
            for i, t in enumerate(r.get("prior") or []):
                who = "U" if i % 2 == 0 else "A"
                lines.append(f"- {who}: {t}")
            lines.append(f"- **U:** {r['user']}")
            lines.append(f"- **A:** {r['reply']}")
            lines.append("")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(lines), encoding="utf-8")


def _read(path):
    with open(path, encoding="utf-8") as fh:
        return [json.loads(l) for l in fh if l.strip()]


def _write(path, rows):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")


def manifest_entry(path, rows, v10, n_new, n_pruned, n_dropped):
    e = copy.deepcopy(v10)
    for stale in ("rewritten", "dropped_by_phrase", "dropped_short"):
        e.pop(stale, None)
    data = Path(path).read_bytes()
    e.update(
        file=Path(path).name,
        rows=len(rows),
        bytes=len(data),
        sha256=hashlib.sha256(data).hexdigest(),
        kinds=dict(collections.Counter(r.get("kind", "?") for r in rows)),
        metrics=at.set_metrics(rows),
        pruned_legacy_refusals=n_pruned,
        dropped_conflicting_math=n_dropped,
        note=(f"phase2_v10 pruned ({n_pruned} redundant legacy refusals, {n_dropped} conflicting math) "
              f"plus {n_new} curated camus12 gold rows"),
    )
    e["inputs"] = list(e.get("inputs", [])) + [
        {"input": "data/drafts/v12/gold_v12.jsonl", "weight": 1, "rows_read": n_new}
    ]
    try:
        e["git_commit"] = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True
        ).stdout.strip() or e.get("git_commit")
    except OSError:
        pass
    return e


def main():
    gold = _read(GOLD_PATH)
    errs = validate_gold(gold)
    if errs:
        for rid, msg in errs:
            print(f"ERROR [{rid}]: {msg}", file=sys.stderr)
        raise SystemExit(f"gold_v12 validation failed with {len(errs)} error(s)")

    write_review_sheet(gold, REVIEW_PATH)

    base_sft = _read(OUT / "phase2_v10.jsonl")
    pruned_sft, n_pruned = prune_legacy_refusals(base_sft)
    clean_sft, n_dropped_sft = drop_conflicting_rows(pruned_sft)
    new_sft = [render_gold_row(r) for r in gold]
    phase2_v12 = clean_sft + new_sft
    _write(OUT / "phase2_v12.jsonl", phase2_v12)

    with open(OUT / "MANIFEST.json", encoding="utf-8") as fh:
        man = json.load(fh)
    man["sets"]["phase2_v12"] = manifest_entry(
        OUT / "phase2_v12.jsonl", phase2_v12, man["sets"]["phase2_v10"],
        len(new_sft), n_pruned, n_dropped_sft,
    )
    with open(OUT / "MANIFEST.json", "w", encoding="utf-8") as fh:
        json.dump(man, fh, indent=2, ensure_ascii=False)

    base_dpo = _read(OUT / "dpo_v10.jsonl")
    clean_dpo, n_dropped_dpo = drop_conflicting_rows(base_dpo)
    _write(OUT / "dpo_v12.jsonl", clean_dpo)

    print(f"phase2_v12: {len(phase2_v12)} rows (pruned {n_pruned} refusals, dropped {n_dropped_sft} math conflicts, +{len(new_sft)} gold)")
    print("gold by category:", dict(collections.Counter(r["category"] for r in gold)))
    print(f"dpo_v12: {len(clean_dpo)} pairs (dropped {n_dropped_dpo} conflicting math-refusal pairs from dpo_v10)")
    print(f"review sheet: {REVIEW_PATH}")


if __name__ == "__main__":
    main()
