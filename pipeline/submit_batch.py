#!/usr/bin/env python3
"""
Script A — THE SUBMITTER
========================
Builds Batch API requests for a stage, writes them (and the custom_id map) to
./batches/ as durable local records, then submits the batch inline to Anthropic.

Note on the API shape: Anthropic's Message Batches API takes requests INLINE in
batches.create(requests=[...]) — there is no separate file-upload step (that's
the OpenAI flow). The local .jsonl we write is our own record + custom_id map.

Stages (run in this order, fetching between dependent stages):
    sft               reads ./data/chunks.jsonl
    style             reads ./data/camus_sft.jsonl   (run AFTER fetching sft)
    adv_paraphrase    no input (template expansion)
    adv_pairs         reads ./data/adv_prompts.jsonl (run AFTER fetching adv_paraphrase)

Usage:
    python submit_batch.py sft
    python submit_batch.py style
    python submit_batch.py adv_paraphrase
    python submit_batch.py adv_pairs
"""

import sys
import os
import json
import random
from datetime import datetime, timezone

from anthropic import Anthropic
import config as C

client = Anthropic()   # reads ANTHROPIC_API_KEY from env (config loaded .env)


def _req(cid, system, user, max_tokens):
    """A single Batch request object (plain dict — the SDK accepts this and it
    serializes cleanly to our local .jsonl record)."""
    return {"custom_id": cid,
            "params": {"model": C.MODEL, "max_tokens": max_tokens,
                       "system": system,
                       "messages": [{"role": "user", "content": user}]}}


def build_sft():
    requests, mapping = [], {}
    for c in C.load_jsonl(C.CHUNKS):
        sid = str(c.get("id"))
        passage = C.clean_passage(c.get("text", ""))
        if len(passage) < C.MIN_PASSAGE_CHARS:
            continue
        cid = f"sft-{sid}"[:64]
        user = C.SFT_USER.format(source=c.get("source", "Camus"), passage=passage,
                                 n_prompts=C.PROMPTS_PER_CHUNK, n_convos=C.CONVOS_PER_CHUNK)
        requests.append(_req(cid, C.SFT_SYSTEM, user, C.MAX_TOKENS["sft"]))
        mapping[cid] = {"src_id": sid, "source": c.get("source", "Camus"), "passage": passage}
    return requests, mapping


def build_style():
    rows = C.load_jsonl(C.SFT_OUT)
    pool = [r for r in rows if len(r["response"]) <= C.MAX_CHOSEN_CHARS]
    random.seed(42); random.shuffle(pool)
    pool = pool[: int(len(pool) * C.STYLE_SAMPLE)]
    requests, mapping = [], {}
    for i, r in enumerate(pool):
        axes = random.sample(C.AXES, min(C.VARIANTS_PER_ROW, len(C.AXES)))
        cid = f"style-{i}"
        user = C.STYLE_USER.format(axes=", ".join(axes), prompt=r["prompt"], chosen=r["response"])
        requests.append(_req(cid, C.STYLE_SYSTEM, user, C.MAX_TOKENS["style"]))
        mapping[cid] = {"prompt": r["prompt"], "chosen": r["response"]}
    return requests, mapping


def build_adv_paraphrase():
    base = C.expand_templates(C.ADV_TARGET_PROMPTS, C.PER_CATEGORY_CAP)
    sysmsg = C.PARAPHRASE_SYSTEM.format(n=C.ADV_PARAPHRASES)
    requests, mapping = [], {}
    for i, (cat, p) in enumerate(base):
        cid = f"para-{i}"
        requests.append(_req(cid, sysmsg, p, C.MAX_TOKENS["adv_paraphrase"]))
        mapping[cid] = {"category": cat, "base_prompt": p}
    return requests, mapping


def build_adv_pairs():
    prompts = C.load_jsonl(C.ADV_PROMPTS)
    requests, mapping = [], {}
    for i, row in enumerate(prompts):
        cid = f"pair-{i}"
        requests.append(_req(cid, C.PAIR_SYSTEM, row["prompt"], C.MAX_TOKENS["adv_pairs"]))
        mapping[cid] = {"category": row["category"], "prompt": row["prompt"]}
    return requests, mapping


BUILDERS = {"sft": build_sft, "style": build_style,
            "adv_paraphrase": build_adv_paraphrase, "adv_pairs": build_adv_pairs}


def load_manifest():
    if os.path.exists(C.MANIFEST):
        return json.load(open(C.MANIFEST, encoding="utf-8"))
    return {}


def save_manifest(m):
    json.dump(m, open(C.MANIFEST, "w", encoding="utf-8"), indent=2)


def main():
    if len(sys.argv) != 2 or sys.argv[1] not in BUILDERS:
        sys.exit(f"usage: python submit_batch.py [{'|'.join(BUILDERS)}]")
    stage = sys.argv[1]

    requests, mapping = BUILDERS[stage]()
    if not requests:
        sys.exit(f"No requests built for '{stage}'. Did the input file exist / have rows?")
    if len(requests) > 100_000:
        sys.exit(f"{len(requests)} requests exceeds the 100k per-batch limit; split required.")

    req_path = f"{C.BATCH_DIR}/{stage}_requests.jsonl"
    map_path = f"{C.BATCH_DIR}/{stage}_map.jsonl"
    C.write_jsonl(req_path, requests)
    C.write_jsonl(map_path, [{"custom_id": k, **v} for k, v in mapping.items()])

    print(f"[{stage}] built {len(requests)} requests -> {req_path}")
    print(f"[{stage}] submitting to Anthropic Batch API ...")
    batch = client.messages.batches.create(requests=requests)

    manifest = load_manifest()
    manifest[stage] = {
        "batch_id": batch.id,
        "status": batch.processing_status,
        "n_requests": len(requests),
        "map_path": map_path,
        "submitted_at": datetime.now(timezone.utc).isoformat(),
    }
    save_manifest(manifest)
    print(f"[{stage}] ✅ submitted. batch_id = {batch.id} (status: {batch.processing_status})")
    print(f"[{stage}] next: python fetch_batch.py {stage}")


if __name__ == "__main__":
    main()
