#!/usr/bin/env python3
"""
verify_sft.py — integrity gate for camus_sft.jsonl before the style stage.

Confirms the things that actually matter at scale:
  * every essayist `response` is a VERBATIM copy of its source chunk
  * every chunk produced an essayist row (coverage)
  * no AI-tells / lists / anachronisms leaked into responses
  * conversational rows are sane in length / sentence count
  * prompts are diverse, conversational responses deduped

Reads paths from config.py (./data/chunks.jsonl and ./data/camus_sft.jsonl).

    python verify_sft.py
"""
import json
import re
from collections import Counter
import config as C

AI  = re.compile(r"\b(as an ai|i am an ai|language model|i'm an ai|openai|anthropic|chatbot|as a large)\b", re.I)
LIST = re.compile(r"(^|\n)\s*(\d+[.)]\s|[-*\u2022]\s)")
ANA = re.compile(r"\b(internet|smartphone|email|website|computer|software|app|online|wifi|google|twitter)\b", re.I)


def load(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def main():
    chunks = {c["id"]: c for c in load(C.CHUNKS)}
    sft = load(C.SFT_OUT)
    ess = [r for r in sft if r["type"] == "essayist"]
    conv = [r for r in sft if r["type"] == "conversational"]
    print(f"rows: {len(sft)}  |  {dict(Counter(r['type'] for r in sft))}")
    print(f"source chunks: {len(chunks)}")

    # ── verbatim integrity (the critical one) ──────────────────────────────
    match = mismatch = missing = 0
    examples = []
    for r in ess:
        c = chunks.get(r["src_id"])
        if not c:
            missing += 1
            continue
        if C.clean_passage(c["text"]) == r["response"]:
            match += 1
        else:
            mismatch += 1
            if len(examples) < 5:
                examples.append((r["src_id"], r["response"][:80], C.clean_passage(c["text"])[:80]))
    print(f"\nessayist verbatim: match={match}  mismatch={mismatch}  missing_chunk={missing}")
    for sid, a, b in examples:
        print(f"  MISMATCH {sid}\n    sft  : {a!r}\n    chunk: {b!r}")

    covered = len(set(r["src_id"] for r in ess))
    print(f"coverage: {covered}/{len(chunks)} chunks have an essayist row")

    # ── content flags ──────────────────────────────────────────────────────
    fl = Counter()
    for r in sft:
        if AI.search(r["response"]):  fl["AI-TELL"] += 1
        if LIST.search(r["response"]): fl["LIST"] += 1
        if ANA.search(r["response"]): fl["ANACHRONISM"] += 1
    print(f"flags in responses: {dict(fl) or 'none'}")

    # ── length / sentence sanity ───────────────────────────────────────────
    def stat(xs): xs = sorted(xs); return f"{xs[0]}/{xs[len(xs)//2]}/{xs[-1]}" if xs else "-"
    print(f"essayist words (min/med/max):        {stat([len(r['response'].split()) for r in ess])}")
    print(f"conversational words (min/med/max):  {stat([len(r['response'].split()) for r in conv])}")
    cs = [len(re.findall(r'[.!?]', r['response'])) for r in conv]
    print(f"conversational sentences (min/med/max): {stat(cs)}  (>6 sentences: {sum(1 for s in cs if s > 6)})")

    # ── diversity / dedup ──────────────────────────────────────────────────
    ep = [r["prompt"] for r in ess]
    print(f"essayist prompts: {len(ep)} total, {len(set(ep))} unique")
    print(f"conversational duplicate responses: {len(conv) - len(set(r['response'] for r in conv))}")

    # ── verdict ────────────────────────────────────────────────────────────
    ok = mismatch == 0 and missing == 0 and fl.get("AI-TELL", 0) == 0
    print("\nVERDICT:", "✅ clean — safe to run the style stage"
          if ok else "⚠️  review the issues above before proceeding")


if __name__ == "__main__":
    main()
