"""Build the camus6 RAFT rows: biography questions paired with the exact system prompt and
passages the runtime retrieves for them, so the model learns to answer from what it is shown
and to ignore the decoys next to it (RAFT, arXiv:2403.10131). The closed-book versions of
these rows are dropped from Phase 2 (Gekhman et al., arXiv:2405.05904).

Needs the local knowledge base and Ollama, so it runs on a workstation, never in the cloud.
Everything it writes contains KB text and lives under the gitignored data/drafts/.

  python pipeline/build_raft_v6.py contexts    # retrieve, write data/drafts/raft/batch_*.jsonl
  (judge each batch: verdict + grounded answer -> data/drafts/raft/judged_*.jsonl)
  python pipeline/build_raft_v6.py assemble    # write data/drafts/raft_v6.jsonl
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data/drafts/raft"
FINAL = ROOT / "data/drafts/raft_v6.jsonl"
BATCH = 40

# Closed-book biography rows that become grounded rows.
SOURCES = {
    "data/cloud/epistemic_v4.jsonl": {"true_bio_answer", "false_bio_correct", "false_bio_hedge",
                                      "true_attr_affirm", "false_attr_named", "false_attr_hedge"},
    "data/cloud/defects_v4.jsonl": {"bio_precision", "true_quote", "false_quote"},
    "data/cloud/defects_v6.jsonl": {"bio_family", "bio_pets"},
}
QUESTIONS = ROOT / "pipeline/raft_questions_v6.txt"


def single_turn(row):
    """(prompt, response) for a single-turn row in either format, else None."""
    if "prompt" in row:
        return row["prompt"], row["response"]
    m = row.get("messages", [])
    if len(m) == 2 and m[0]["role"] == "user" and m[1]["role"] == "assistant":
        return m[0]["content"], m[1]["content"]
    return None


def collect():
    items = []
    for rel, kinds in SOURCES.items():
        for line in open(ROOT / rel, encoding="utf-8"):
            row = json.loads(line)
            pr = single_turn(row) if row.get("kind") in kinds else None
            if pr:
                items.append({"kind": row["kind"], "prompt": pr[0], "old_response": pr[1]})
    for q in QUESTIONS.read_text(encoding="utf-8").splitlines():
        if q.strip() and not q.startswith("#"):
            items.append({"kind": "decoy", "prompt": q.strip(), "old_response": ""})
    seen, out = set(), []
    for it in items:                       # the same prompt can sit in two sources
        if it["prompt"] not in seen:
            seen.add(it["prompt"])
            out.append(it)
    return out


def contexts():
    sys.path.insert(0, str(ROOT / "rag"))
    import camus_rag as cr
    facts, vecs = cr.load_kb()
    bm25, ce = cr.build_bm25(facts), cr.load_reranker()
    items = collect()
    OUT.mkdir(parents=True, exist_ok=True)
    for i, it in enumerate(items):
        turn = cr.build_turn(it["prompt"], [], facts, vecs, bm25=bm25, ce=ce)
        it["id"] = i
        it["system"] = turn.messages[0]["content"]
        it["passages"] = [h["text"] + (f' (your words: "{h["quote"]}")' if h.get("quote") else "")
                          for h in turn.hits]
    for b in range(0, len(items), BATCH):
        with open(OUT / f"batch_{b // BATCH:02d}.jsonl", "w", encoding="utf-8") as fh:
            for it in items[b:b + BATCH]:
                fh.write(json.dumps({k: it[k] for k in ("id", "kind", "prompt", "passages",
                                                         "old_response")}, ensure_ascii=False) + "\n")
    with open(OUT / "contexts.jsonl", "w", encoding="utf-8") as fh:
        for it in items:
            fh.write(json.dumps(it, ensure_ascii=False) + "\n")
    print(f"{len(items)} items, {(len(items) + BATCH - 1) // BATCH} batches in {OUT}")


def assemble():
    ctx = {json.loads(l)["id"]: json.loads(l) for l in open(OUT / "contexts.jsonl", encoding="utf-8")}
    rows, dropped = [], 0
    for f in sorted(OUT.glob("judged_*.jsonl")):
        for line in open(f, encoding="utf-8"):
            j = json.loads(line)
            c = ctx[j["id"]]
            if j.get("verdict") == "drop" or not j.get("response", "").strip():
                dropped += 1
                continue
            rows.append({"messages": [{"role": "system", "content": c["system"]},
                                      {"role": "user", "content": c["prompt"]},
                                      {"role": "assistant", "content": j["response"].strip()}],
                         "kind": "raft_" + c["kind"]})
    with open(FINAL, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"{len(rows)} rows -> {FINAL} ({dropped} dropped, {len(ctx) - len(rows) - dropped} unjudged)")


if __name__ == "__main__":
    {"contexts": contexts, "assemble": assemble}[sys.argv[1]]()
