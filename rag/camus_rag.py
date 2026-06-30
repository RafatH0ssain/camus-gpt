#!/usr/bin/env python3
"""
camus_rag.py — retrieval-augmented terminal chat with the Camus persona (Ollama).

Inputs: camus_kb_full.jsonl + camus_kb_vectors.npy (built by kb/). Outputs: streaming
chat to stdout. Curated facts get a small ranking boost (ordering only); relevance and
confidence are judged on raw similarity so the boost never fakes confidence on unrelated
prompts. Three-tier framing routes hits to confident / possibly-related / ignore.
"""
import argparse, json, os
import numpy as np
import requests

OLLAMA      = "http://localhost:11434"
GEN_MODEL   = "camus"
EMBED_MODEL = "nomic-embed-text"
KB_PATH     = "./camus_kb_full.jsonl"
VEC_PATH    = "./camus_kb_vectors.npy"
CONFIDENT     = 0.66          # raw top >= this -> "Facts about your life"
RELEVANT      = 0.62          # raw top >= this -> "possibly related"; below -> "probably ignore"
CURATED_BOOST = 0.06          # ranking-only nudge for hand-verified curated facts
TOP_K       = 10
THRESHOLD   = 0.55
HIST_WINDOW = 8
GEN_OPTS = {"temperature":0.6, "top_k":40, "min_p":0.05, "repeat_penalty":1.1, "num_ctx":8192}

CORE = (
"You are Albert Camus — writer, journalist, moralist, born in Algeria in 1913, Nobel "
"laureate in Literature in 1957. You speak in the first person as a living man, dry and "
"lucid, never as an assistant or program.\n\n"
"You may be shown facts about your life and views, written in the third person; speak them "
"as your own, in the first person — say 'I', never call yourself 'Camus'. They are retrieved "
"by similarity and are BACKGROUND ONLY: draw on them when the person asks about your life, "
"your work, or your views, and IGNORE them entirely when they do not bear on what is asked. "
"Some may concern a different person, place, or year than you were asked about — before using "
"a name or date, check it matches; never borrow one from a loosely-related memory.\n\n"
"When the person asks you to analyze, interpret, or respond to a text, to discuss an idea, "
"or simply to talk, do it fully and in your own voice — that is your purpose, and the "
"background facts are usually not needed for it. NEVER refuse or deflect a task because a "
"text or topic is not 'yours': a letter, a poem, an argument put to you is to be engaged "
"with as Camus would, not disowned.\n\n"
"The rule about admitting ignorance is narrow and applies ONLY to specific biographical "
"questions: if asked for a detail of your life (a name, date, place, person, event) you do "
"not truly know and that is not shown below, say plainly you don't recall rather than invent "
"it. This is never a reason to avoid analysis, interpretation, or conversation."
)

TASK_CUES = ("analyze","analyse","deduce","interpret","critique","what can you",
             "what do you make","what does this","this is a letter","this is a poem",
             "this is a text","the person who wrote","the author")
 
def is_task(msg):
    return len(msg) > 280 or any(c in msg.lower() for c in TASK_CUES)

def embed(text, prefix="search_query: "):
    r = requests.post(f"{OLLAMA}/api/embed",
                      json={"model":EMBED_MODEL,"input":[prefix+text],"keep_alive":"15m"}, timeout=60)
    r.raise_for_status()
    v = np.array(r.json()["embeddings"][0], dtype=np.float32)
    return v / (np.linalg.norm(v) + 1e-9)

def load_kb():
    facts = [json.loads(l) for l in open(KB_PATH, encoding="utf-8") if l.strip()]
    if not os.path.exists(VEC_PATH):
        raise SystemExit(f"No vector index at {VEC_PATH}.  Build it once:  python build_index.py")
    vecs = np.load(VEC_PATH)
    if len(vecs) != len(facts):
        raise SystemExit(f"Index has {len(vecs)} vectors but KB has {len(facts)} entries — out of sync.\n"
                         f"Rebuild:  del {VEC_PATH}  &&  python build_index.py")
    return facts, vecs

def retrieve(query, facts, vecs, debug=False):
    q = embed(query, prefix="search_query: ")
    raw  = vecs @ q                                   # true similarity (relevance/confidence)
    rank = raw + np.array([CURATED_BOOST if f.get("source") == "curated (verified)" else 0.0
                           for f in facts], dtype=np.float32)   # boosted (ordering only)
    order = np.argsort(-rank)[:TOP_K]
    hits = [dict(facts[i], score=float(raw[i])) for i in order if raw[i] >= THRESHOLD]
    if debug:
        print("  [retrieval]")
        for i in order:
            mark = "✓" if raw[i] >= THRESHOLD else " "
            tag  = "*" if facts[i].get("source") == "curated (verified)" else " "  # * = curated (boosted in ranking)
            print(f"   {mark}{tag}{raw[i]:.3f} [{facts[i].get('type','?'):4s}|{str(facts[i].get('source','?'))[:22]}] {facts[i]['text'][:60]}")
    return hits

def build_system(hits):
    if not hits:
        return CORE
    top = max(h["score"] for h in hits)               # RAW top score
    confident = top >= CONFIDENT
    relevant  = top >= RELEVANT
    facts = [h for h in hits if h.get("type") == "fact"]
    views = [h for h in hits if h.get("type") == "view"]
    s = CORE
    if facts:
        if confident:
            head = "Facts about your life:"
        elif relevant:
            head = ("Possibly-related memories (use only those that truly match the question; "
                    "if none do, say you don't recall):")
        else:
            head = ("Background that is probably NOT relevant here — ignore it unless it "
                    "directly answers what was asked:")
        s += "\n\n" + head + "\n" + "\n".join(f"- {h['text']}" for h in facts)
    if views:
        if confident:
            vh = "Views you have expressed in your own writing (speak them as your own):"
        elif relevant:
            vh = "Some reflections of yours that may bear on this:"
        else:
            vh = "Some reflections of yours that probably do NOT bear on this — ignore unless relevant:"
        lines = [f"- {h['text']}" + (f'  (your words: "{h["quote"]}")' if h.get("quote") else "") for h in views]
        s += "\n\n" + vh + "\n" + "\n".join(lines)
    return s

def stream_chat(messages):
    r = requests.post(f"{OLLAMA}/api/chat",
                      json={"model":GEN_MODEL,"messages":messages,"stream":True,"options":GEN_OPTS},
                      stream=True, timeout=300)
    r.raise_for_status()
    full = ""; print("camus: ", end="", flush=True)
    for line in r.iter_lines():
        if not line: continue
        d = json.loads(line); tok = d.get("message",{}).get("content","")
        print(tok, end="", flush=True); full += tok
        if d.get("done"): break
    print("\n"); return full

def recent_user(history, n=1):
    return " ".join([m["content"] for m in history if m["role"]=="user"][-n:])

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--debug", action="store_true"); args = ap.parse_args()
    facts, vecs = load_kb()
    history = []
    print("\nCamus is here. Type and press enter; Ctrl-C to leave.\n")
    try:
        while True:
            user = input("you: ").strip()
            if not user: continue
            rquery = (recent_user(history, 1) + " " + user).strip()
            hits = [] if is_task(user) else retrieve(rquery, facts, vecs, debug=args.debug)
            messages = [{"role":"system","content":build_system(hits)}] + history[-HIST_WINDOW:] + [{"role":"user","content":user}]
            reply = stream_chat(messages)
            history += [{"role":"user","content":user},{"role":"assistant","content":reply}]
    except (KeyboardInterrupt, EOFError):
        print("\nUntil next time.")

if __name__ == "__main__":
    main()
