#!/usr/bin/env python3
"""
CamusGPT — DATA INSPECTOR
=========================
Stratified random sample + automatic red-flag scan. Auto-detects the schema:
  chunks  -> {id, source, text}      (Step 0 output)
  SFT     -> {type, prompt, response}
  DPO     -> {category|axis, prompt, chosen, rejected}

    python inspect_data.py ./data/chunks.jsonl --n 15
    python inspect_data.py ./data/camus_sft.jsonl --n 12
    python inspect_data.py ./data/camus_dpo_adversarial.jsonl --n 12
"""
import argparse, json, random, re, sys
from collections import defaultdict

# content flags (SFT/DPO)
AI_TELL = re.compile(r"\b(as an ai|i am an ai|language model|i'?m an ai|a\.?i\.? (model|assistant)|openai|anthropic|chatbot|as a large)\b", re.I)
LIST_TELL = re.compile(r"(^|\n)\s*(\d+[.)]\s|[-*\u2022]\s)")
ANACHRONISM = re.compile(r"\b(internet|smartphone|email|website|computer|software|app|online|wifi|google|twitter)\b", re.I)
# chunk-cleanliness flags
HYPHEN_BREAK = re.compile(r"\w-\s")                                  # "bound- less" leftover
CAPS_RUN = re.compile(r"\b[A-Z]{2,}(?:\s+[A-Z]{2,}){2,}\b")          # surviving header

def _flags(t):
    f = []
    if AI_TELL.search(t): f.append("AI-TELL")
    if LIST_TELL.search(t): f.append("LIST")
    if ANACHRONISM.search(t): f.append("ANACHRONISM")
    return f

def _chunk_flags(t):
    f = []
    if HYPHEN_BREAK.search(t): f.append("HYPHEN-BREAK")
    if "\n" in t: f.append("STRAY-NEWLINE")
    if CAPS_RUN.search(t): f.append("CAPS-RUN?")
    return f

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("file")
    ap.add_argument("--n", type=int, default=12)
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.file, encoding="utf-8") if l.strip()]
    if not rows:
        sys.exit("empty file")
    r0 = rows[0]
    if "chosen" in r0:   mode = "dpo"
    elif "response" in r0: mode = "sft"
    elif "text" in r0:   mode = "chunks"
    else: sys.exit(f"unrecognized schema; keys present: {list(r0)}")

    tgt = {"dpo": "chosen", "sft": "response", "chunks": "text"}[mode]
    key = "category" if "category" in r0 else ("type" if "type" in r0 else ("source" if "source" in r0 else None))
    flagfn = _chunk_flags if mode == "chunks" else _flags

    buckets = defaultdict(list)
    for r in rows: buckets[r.get(key, "all")].append(r)
    random.seed(0); sample = []; per = max(1, args.n // max(1, len(buckets)))
    for b in buckets.values(): sample += random.sample(b, min(per, len(b)))

    bad = defaultdict(int); lens = defaultdict(list)
    for r in rows:
        lens[r.get(key, "all")].append(len(r[tgt].split()))
        for fl in flagfn(r[tgt]): bad[fl] += 1

    print(f"\n== {args.file} :: {len(rows)} rows :: {mode.upper()} ==")
    for strat, ws in lens.items():
        ws.sort(); print(f"  {str(strat):24s} n={len(ws):5d}  words[min/median/max]="
                          f"{ws[0]}/{ws[len(ws)//2]}/{ws[-1]}")
    label = "text" if mode == "chunks" else tgt
    print(f"  FLAGS in '{label}': {dict(bad) or 'none'}")
    if mode == "dpo":
        rej = defaultdict(int)
        for r in rows:
            for fl in _flags(r["rejected"]): rej[fl] += 1
        print(f"  FLAGS in 'rejected' (expected high): {dict(rej)}")

    print("\n-- SAMPLE --")
    for r in sample[:args.n]:
        if mode == "chunks":
            print(f"\n[{r.get(key,'?')}] {r.get('id','')} {_chunk_flags(r['text']) or ''}")
            print(f"  {r['text'][:300]}")
        elif mode == "dpo":
            print(f"\n[{r.get(key,'?')}] PROMPT: {r['prompt'][:120]}")
            print(f"  CHOSEN {_flags(r['chosen']) or ''}: {r['chosen'][:240]}")
            print(f"  REJECT {_flags(r['rejected']) or ''}: {r['rejected'][:240]}")
        else:
            print(f"\n[{r.get(key,'?')}] PROMPT: {r['prompt'][:120]}")
            print(f"  RESP   {_flags(r['response']) or ''}: {r['response'][:280]}")

if __name__ == "__main__":
    main()
