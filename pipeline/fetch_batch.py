#!/usr/bin/env python3
"""
Script B — THE RECEIVER
=======================
Polls a submitted batch until it ends, streams the results, resolves each
custom_id back to its source via the saved map, parses the model output, and
writes the clean dataset file for that stage.

Stage -> output:
    sft            -> ./data/camus_sft.jsonl
    style          -> ./data/camus_dpo_style.jsonl
    adv_paraphrase -> ./data/adv_prompts.jsonl   (intermediate; feeds adv_pairs)
    adv_pairs      -> ./data/camus_dpo_adversarial.jsonl

Usage:
    python fetch_batch.py sft
    python fetch_batch.py style
    python fetch_batch.py adv_paraphrase
    python fetch_batch.py adv_pairs

Batches usually finish in well under an hour (24h max). Results stay retrievable
for 29 days. This script is safe to re-run: if the batch is still processing it
keeps polling; once ended it parses and writes.
"""

import sys
import os
import json
import time
import hashlib

from anthropic import Anthropic
import config as C

client = Anthropic()


def load_manifest():
    if not os.path.exists(C.MANIFEST):
        sys.exit("No manifest found. Submit a batch first with submit_batch.py.")
    return json.load(open(C.MANIFEST, encoding="utf-8"))


def load_map(path):
    return {r["custom_id"]: r for r in C.load_jsonl(path)}


def poll(batch_id):
    while True:
        b = client.messages.batches.retrieve(batch_id)
        rc = b.request_counts
        print(f"  status={b.processing_status}  "
              f"processing={rc.processing} succeeded={rc.succeeded} "
              f"errored={rc.errored} canceled={rc.canceled} expired={rc.expired}")
        if b.processing_status == "ended":
            return b
        time.sleep(C.POLL_SECONDS)


def _err_msg(result):
    """Best-effort human-readable reason for a non-succeeded result."""
    if result.type != "errored":
        return result.type  # canceled / expired
    err = getattr(result, "error", None)
    inner = getattr(err, "error", None) or err
    t = getattr(inner, "type", None)
    m = getattr(inner, "message", None)
    if m:
        return f"{t}: {m}"
    return repr(err)[:200]


def collect(batch_id):
    """Stream results -> list of (custom_id, text). Groups non-successes by reason."""
    out, errors = [], {}
    for entry in client.messages.batches.results(batch_id):
        if entry.result.type == "succeeded":
            text = "".join(b.text for b in entry.result.message.content
                           if getattr(b, "type", None) == "text")
            out.append((entry.custom_id, text))
        else:
            msg = _err_msg(entry.result)
            errors[msg] = errors.get(msg, 0) + 1
    if errors:
        print("  non-succeeded results by reason:")
        for msg, cnt in sorted(errors.items(), key=lambda x: -x[1])[:5]:
            print(f"    [{cnt}] {msg}")
    return out


# ── Per-stage parsers ───────────────────────────────────────────────────────
def parse_sft(results, m):
    rows, seen = [], set()
    mt, mm = 0, 0  # motif throttle counters
    for cid, text in results:
        meta = m.get(cid)
        if not meta:
            continue
        try:
            data = C.extract_json(text)
        except Exception:
            continue
        for p in data.get("prompts", [])[:C.PROMPTS_PER_CHUNK]:
            if isinstance(p, str) and p.strip():
                rows.append({"type": "essayist", "prompt": p.strip(),
                             "response": meta["passage"], "source": meta["source"],
                             "src_id": meta["src_id"]})
        for ex in data.get("conversational", [])[:C.CONVOS_PER_CHUNK]:
            try:
                cp, cr = ex["prompt"].strip(), ex["response"].strip()
            except Exception:
                continue
            if not cp or not cr:
                continue
            h = hashlib.md5(cr[:400].encode()).hexdigest()
            if h in seen:
                continue
            has_motif = any(x in cr.lower() for x in C.SFT_MOTIFS)
            mt += 1
            if has_motif:
                if mm / max(1, mt) > C.SFT_MOTIF_MAX:
                    continue
                mm += 1
            seen.add(h)
            rows.append({"type": "conversational", "prompt": cp, "response": cr,
                         "source": meta["source"], "src_id": meta["src_id"]})
    return rows


def parse_style(results, m):
    rows = []
    for cid, text in results:
        meta = m.get(cid)
        if not meta:
            continue
        try:
            variants = C.extract_json(text).get("rejected", [])
        except Exception:
            continue
        for v in variants:
            txt = (v.get("text") or "").strip()
            if txt and txt != meta["chosen"]:
                rows.append({"category": "style", "axis": v.get("axis", "?"),
                             "prompt": meta["prompt"], "chosen": meta["chosen"],
                             "rejected": txt})
    return rows


def parse_adv_paraphrase(results, m):
    """Emit base prompts + their paraphrases as {category, prompt}, deduped."""
    rows, seen = [], set()
    for cid, text in results:
        meta = m.get(cid)
        if not meta:
            continue
        cat, base = meta["category"], meta["base_prompt"]
        candidates = [base]
        try:
            for s in C.extract_json(text):
                if isinstance(s, str) and s.strip():
                    candidates.append(s.strip())
        except Exception:
            pass
        for p in candidates:
            if p not in seen:
                seen.add(p)
                rows.append({"category": cat, "prompt": p})
    return rows


def parse_adv_pairs(results, m):
    rows, dropped = [], 0
    for cid, text in results:
        meta = m.get(cid)
        if not meta:
            continue
        try:
            d = C.extract_json(text)
            ch, rj = d["chosen"].strip(), d["rejected"].strip()
        except Exception:
            continue
        if not ch or not rj or ch == rj:
            continue
        if C.CONCEDE_AI.search(ch) or C.MODERN_TECH.search(ch):
            dropped += 1                 # chosen admits AI-ness or breaks period voice
            continue
        rows.append({"category": f"adversarial:{meta['category']}",
                     "prompt": meta["prompt"], "chosen": ch, "rejected": rj})
    if dropped:
        print(f"  dropped {dropped} pairs (chosen conceded AI-ness or used anachronistic words)")
    return rows


PARSERS = {"sft": parse_sft, "style": parse_style,
           "adv_paraphrase": parse_adv_paraphrase, "adv_pairs": parse_adv_pairs}
OUTPUTS = {"sft": C.SFT_OUT, "style": C.STYLE_OUT,
           "adv_paraphrase": C.ADV_PROMPTS, "adv_pairs": C.ADV_OUT}


def main():
    if len(sys.argv) != 2 or sys.argv[1] not in PARSERS:
        sys.exit(f"usage: python fetch_batch.py [{'|'.join(PARSERS)}]")
    stage = sys.argv[1]

    manifest = load_manifest()
    if stage not in manifest:
        sys.exit(f"No submitted batch for '{stage}'. Run submit_batch.py {stage} first.")
    entry = manifest[stage]
    batch_id = entry["batch_id"]
    print(f"[{stage}] batch {batch_id} — polling every {C.POLL_SECONDS}s ...")

    poll(batch_id)
    results = collect(batch_id)
    rows = PARSERS[stage](results, load_map(entry["map_path"]))
    C.write_jsonl(OUTPUTS[stage], rows)

    entry["status"] = "done"
    json.dump(manifest, open(C.MANIFEST, "w", encoding="utf-8"), indent=2)
    print(f"[{stage}] ✅ wrote {len(rows)} rows -> {OUTPUTS[stage]}")
    if stage == "sft":
        print("[sft] next: python submit_batch.py style")
    elif stage == "adv_paraphrase":
        print("[adv_paraphrase] next: python submit_batch.py adv_pairs")
    else:
        print(f"[{stage}] inspect: python inspect_data.py {OUTPUTS[stage]}")


if __name__ == "__main__":
    main()
