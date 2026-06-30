#!/usr/bin/env python3
"""
extract_kb.py — Batch-extract KB entries from source_chunks.jsonl.

TWO MODES, chosen per chunk's stream:
  fact  (biographies/criticism) -> atomic, EXPLICITLY-STATED biographical facts only
  voice (Camus's own words)     -> his paraphrased views/positions (+ optional <15-word quote)

The fact-mode prompt forbids inference and demands the fact be present in THIS passage,
so a wrong entry is the source's error, not the model's invention — and every entry
carries its source + chunk_id for auditing (the lesson from the cat).

Resumable: submits once, saves the batch id + id-map, and on re-run polls / fetches.

SETUP:  pip install anthropic python-dotenv     (.env has ANTHROPIC_API_KEY=...)
RUN:    python extract_kb.py            # submit (first run) then poll
        python extract_kb.py            # re-run to resume polling / fetch results
OUTPUT: ./data/kb_extracted.jsonl   rows {type, text, quote, source, date, chunk_id}
"""
import argparse, json, os, sys, time

MODEL      = "claude-sonnet-4-6"
MAX_TOKENS = 700
BATCH_DIR  = "./batches"
ID_MAP     = os.path.join(BATCH_DIR, "extract_idmap.json")
MANIFEST   = os.path.join(BATCH_DIR, "extract_manifest.json")

SYS_FACT = (
"You extract atomic, verifiable biographical facts about Albert Camus from a passage of a "
"biography or critical study. STRICT RULES: include ONLY facts explicitly stated in THIS "
"passage; never infer, never add outside knowledge; each fact is ONE self-contained sentence "
"in the third person ('Camus ...'); skip the author's interpretation, opinion, or commentary; "
"skip vague or hedged claims. If the passage contains no concrete biographical fact, return []. "
"Return ONLY a JSON array of objects: [{\"text\": \"...\"}]. No prose."
)
SYS_VOICE = (
"You extract Albert Camus's own views and characteristic positions from a passage of HIS writing "
"(notebooks, essays, journalism). RULES: paraphrase each view as ONE self-contained third-person "
"sentence ('Camus held that ...', 'Camus argued ...') capturing what he thought, valued, or claimed "
"— not plot or events; at most 4 items; you MAY attach ONE short verbatim phrase under 15 words if "
"it is especially characteristic. If the passage is purely narrative or states no view, return []. "
"Return ONLY a JSON array of objects: [{\"text\": \"...\", \"quote\": \"...\" or null}]. No prose."
)

def build_requests(chunks):
    reqs = []
    for c in chunks:
        sysmsg = SYS_FACT if c["stream"] == "fact" else SYS_VOICE
        reqs.append({
            "custom_id": c["id"],
            "params": {
                "model": MODEL, "max_tokens": MAX_TOKENS, "system": sysmsg,
                "messages": [{"role":"user","content": c["text"]}],
            },
        })
    return reqs

def load_chunks(path):
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]

def parse_entries(text, meta):
    """meta = {source, stream, date}. Returns list of normalized KB rows."""
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        text = text[text.find("["):]
    try:
        arr = json.loads(text)
    except Exception:
        return []
    out = []
    for item in arr if isinstance(arr, list) else []:
        if isinstance(item, str):
            t, q = item, None
        elif isinstance(item, dict):
            t, q = item.get("text","").strip(), (item.get("quote") or None)
        else:
            continue
        if not t: continue
        out.append({
            "type": "fact" if meta["stream"]=="fact" else "view",
            "text": t, "quote": q, "source": meta["source"], "date": meta.get("date",""),
        })
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks", default="./data/source_chunks.jsonl")
    ap.add_argument("--out", default="./data/kb_extracted.jsonl")
    ap.add_argument("--poll", type=int, default=60)
    args = ap.parse_args()
    os.makedirs(BATCH_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    from dotenv import load_dotenv; load_dotenv()
    import anthropic
    client = anthropic.Anthropic()

    chunks = load_chunks(args.chunks)
    idmeta = {c["id"]: {"source":c["source"],"stream":c["stream"],"date":c.get("date","")} for c in chunks}

    # submit once
    if not os.path.exists(MANIFEST):
        reqs = build_requests(chunks)
        print(f"Submitting {len(reqs)} extraction requests ({MODEL}, batch rates)...")
        batch = client.messages.batches.create(requests=reqs)
        json.dump({"batch_id": batch.id, "n": len(reqs)}, open(MANIFEST,"w"))
        json.dump(idmeta, open(ID_MAP,"w"), ensure_ascii=False)
        print("batch_id =", batch.id)
    else:
        idmeta = json.load(open(ID_MAP, encoding="utf-8"))

    batch_id = json.load(open(MANIFEST))["batch_id"]

    # poll
    while True:
        b = client.messages.batches.retrieve(batch_id)
        c = b.request_counts
        print(f"status={b.processing_status}  done={c.succeeded} err={c.errored} proc={c.processing}")
        if b.processing_status == "ended": break
        time.sleep(args.poll)

    # fetch + parse
    n_rows = 0
    with open(args.out, "w", encoding="utf-8") as out:
        for r in client.messages.batches.results(batch_id):
            if r.result.type != "succeeded": continue
            meta = idmeta.get(r.custom_id)
            if not meta: continue
            text = "".join(blk.text for blk in r.result.message.content if blk.type=="text")
            for row in parse_entries(text, meta):
                row["chunk_id"] = r.custom_id
                out.write(json.dumps(row, ensure_ascii=False) + "\n"); n_rows += 1
    print(f"\n✅ extracted {n_rows} KB entries -> {args.out}")
    print("   Spot-check a sample against the sources before merging (extraction can still err).")

if __name__ == "__main__":
    main()
