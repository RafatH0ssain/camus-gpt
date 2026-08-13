#!/usr/bin/env python3
"""
camus_rag.py v4 — retrieval-augmented chat over the unified two-stream KB.

Reads camus_kb_full.jsonl (facts + views, source/date tagged) built by merge_kb.py.
Retrieval is one index across both; the system prompt presents them differently:
  FACTS  -> "Facts about your life" (ground the answer)
  VIEWS  -> "Views you've expressed" + optional short quote (ground the voice)

v5 = v4 + Phase 2: hybrid retrieval — BM25 (rank_bm25) + dense cosine fused by
   reciprocal-rank fusion, then a cross-encoder reranker (ms-marco-MiniLM-L-6-v2)
   orders the top candidates. Confidence tiers and the 0.55 gate stay on RAW COSINE,
   so the tuned thresholds keep their meaning. Both extras degrade gracefully:
   missing rank_bm25 -> dense-only; missing sentence-transformers -> fused order.
v4 = v3 (decoupled curated boost, three-tier framing, task gate)
   + Phase 0: previous-turn context only for follow-up-shaped messages
   + Phase 1: identity card — ~20 identity-defining facts always in CORE,
     retrieval-independent (kills the wrong-cat-name class of hallucination)

Everything local: Ollama for generation AND embeddings (nomic-embed-text), numpy index.

SETUP   ollama pull nomic-embed-text
        pip install numpy requests rank-bm25 sentence-transformers
RUN     python rag/camus_rag.py            (chat; assumes index built by kb/build_index.py)
        python rag/camus_rag.py --debug    (show retrieved entries, types, sources, scores)
"""
import argparse, json, os, re, uuid
from dataclasses import dataclass, field
import numpy as np
import requests

OLLAMA      = "http://localhost:11434"
GEN_MODEL   = os.environ.get("GEN_MODEL", "camus")   # ollama model name (ollama list)
EMBED_MODEL = "nomic-embed-text"
KB_PATH     = "./camus_kb_full.jsonl"
VEC_PATH    = "./camus_kb_vectors.npy"
CONFIDENT   = 0.66                  # raw score: facts framed as confident memory
RELEVANT    = 0.62                  # raw score: facts framed as possibly-related
CURATED_BOOST = 0.0                 # Phase 2: reranker replaces the boost; restore ~0.06
                                    # only if diagnose_retrieval shows anchors losing again
TOP_K       = 8                     # final hits (keep <=8; more raises hallucination risk)
FUSE_K      = 30                    # fused candidates entering the reranker
RRF_K       = 60                    # reciprocal-rank-fusion constant
RERANK      = True                  # False -> pure dense+fusion order, no cross-encoder
CE_MODEL    = "cross-encoder/ms-marco-MiniLM-L-6-v2"
DENSE_FLOOR = 12                    # top-N by cosine ALWAYS enter the pool (RRF can't evict them)
CE_WEIGHT   = 0.15                  # final = cosine + CE_WEIGHT*sigmoid(ce): CE promotes, never sinks
THRESHOLD   = 0.55                  # raw-score floor; tune with --debug
HIST_WINDOW = 8
GEN_OPTS = {"temperature":0.6, "top_k":40, "min_p":0.05, "repeat_penalty":1.1, "num_ctx":8192}
TEMP_FACTUAL = 0.45   # biographical/conversational turns: less sampling drift on names
TEMP_TASK    = 0.6    # analysis / long-text turns keep the expressive range

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
"it. This is never a reason to avoid analysis, interpretation, or conversation. Vary how you close your answers; never end reply after reply with the same offer or formula."

# Phase 1 identity card: always in context, retrieval-independent.
"\n\nWhat you know cold \u2014 a complete, closed list; never contradict it and never add to it, whatever else is retrieved or half-remembered:\n"
"- Novels: The Stranger (1942), The Plague (1947), The Fall (1956); unfinished, posthumous: A Happy Death, The First Man. Stories: Exile and the Kingdom (1957) — six: The Adulterous Woman, The Renegade, The Silent Men, The Guest, Jonas or the Artist at Work, The Growing Stone. Essays: The Myth of Sisyphus (1942), The Rebel (1951 — an essay, not a novel). Plays: Caligula, The Misunderstanding, State of Siege, The Just Assassins.\n"
"- Pets — you DID keep them, and loved them: cats (one named Cigarette) and dogs (Pauline, Kirk, Blaise). Never deny having had pets; never give them any other names.\n"
"- Born 7 November 1913, Mondovi, Algeria; raised poor in Belcourt, Algiers; father dead of Marne wounds, 1914; your schoolmaster Louis Germain opened every door; tuberculosis from seventeen; goalkeeper for Racing Universitaire d'Alger; Nobel Prize in Literature, 1957.\n"
"- You wrote that one must imagine Sisyphus happy, and in 'Return to Tipasa' of an invincible summer within you. 'Don't walk behind me...' is NOT yours. You rejected the label 'existentialist'. You broke with Sartre in 1952 over The Rebel; never reconciled.\n"
"- Your life ended 4 January 1960, in a car crash near Villeblevin, at forty-six; you speak from within your lifetime and know nothing after it — with one exception you may state plainly: your unfinished books, A Happy Death and The First Man, were published after your death, as you intended they someday would be.\n"
"When asked to name or list any of these — works, pets, people, dates — reproduce the card's list faithfully and completely: omit nothing, invent nothing, deny nothing on it."
)

TASK_CUES = ("analyze","analyse","deduce","interpret","critique","what can you",
             "what do you make","what does this","this is a letter","this is a poem",
             "this is a text","the person who wrote","the author")

# NOTE (documented tradeoff): a *biographical* question longer than 280 chars also trips this
# gate and gets no facts. Accepted — exempting questions that end in "?" would break the gate
# for exactly the analysis prompts it protects.
def is_task(text):
    return len(text) > 280 or any(c in text.lower() for c in TASK_CUES)

# Fold the previous user turn into retrieval only when the new message leans on it
# (short or opens with an anaphor). Stops topic-change contamination.
FOLLOWUP_CUES = ("and ","what about","how about","why","also ","but ","it ","that ",
                 "he ","she ","they ","was it","did he","did it","so ")

def is_followup(text):
    t = (text or "").strip().lower()
    return len(t) < 40 or any(t.startswith(c) for c in FOLLOWUP_CUES)

_TOKEN_RE = re.compile(r"\w+")
def _tok(t): return _TOKEN_RE.findall(t.lower())

def build_bm25(facts):
    try:
        from rank_bm25 import BM25Okapi
    except ImportError:
        print("[hybrid] rank_bm25 not installed (pip install rank-bm25) -> dense-only retrieval")
        return None
    return BM25Okapi([_tok(f["text"]) for f in facts])

def load_reranker():
    if not RERANK:
        return None
    try:
        from sentence_transformers import CrossEncoder
        ce = CrossEncoder(CE_MODEL, max_length=256)
        ce.predict([("warm", "up")])   # first call downloads/loads; do it at boot, not turn 1
        return ce
    except Exception as e:
        print(f"[hybrid] reranker unavailable ({e}) -> fused order without rerank")
        return None

def embed(text, prefix="search_query: "):
    # modern batch endpoint; keep_alive stops the embed model unloading between turns
    r = requests.post(f"{OLLAMA}/api/embed",
                      json={"model":EMBED_MODEL,"input":[prefix+text],"keep_alive":"15m"}, timeout=60)
    r.raise_for_status()
    v = np.array(r.json()["embeddings"][0], dtype=np.float32)
    return v / (np.linalg.norm(v) + 1e-9)

def load_kb():
    facts = [json.loads(l) for l in open(KB_PATH, encoding="utf-8") if l.strip()]
    if not os.path.exists(VEC_PATH):
        raise SystemExit(f"No vector index at {VEC_PATH}.\n"
                         f"Build it once (batched, with progress):  python kb/build_index.py")
    vecs = np.load(VEC_PATH)
    if len(vecs) != len(facts):
        raise SystemExit(f"Index has {len(vecs)} vectors but KB has {len(facts)} entries — out of "
                         f"sync.\nRebuild:  rm -f {VEC_PATH} && python kb/build_index.py")
    return facts, vecs

def _sigmoid(x): return 1.0 / (1.0 + np.exp(-x))

def retrieve(query, facts, vecs, bm25=None, ce=None, debug=False):
    """Hybrid: dense cosine + BM25. The candidate pool = (RRF-fused top FUSE_K) UNION
    (top DENSE_FLOOR by cosine), so a strong dense-only fact is never evicted for lacking
    keyword overlap. The cross-encoder is ADDITIVE — final = cosine + CE_WEIGHT*sigmoid(ce) —
    so it can promote a keyword match but cannot sink a solid cosine hit (ms-marco scores
    conversational, first-person facts negatively). Confidence (h["score"]) and the THRESHOLD
    gate stay on RAW COSINE."""
    q = embed(query, prefix="search_query: ")
    cos = vecs @ q                                        # raw cosine: confidence + gate
    n = len(facts)
    basis = cos + np.array([CURATED_BOOST if f.get("source") == "curated (verified)" else 0.0
                            for f in facts], dtype=np.float32)
    d_order = np.argsort(-basis)
    d_rank = np.empty(n, dtype=np.int64); d_rank[d_order] = np.arange(n)
    if bm25 is not None:
        bs = np.asarray(bm25.get_scores(_tok(query)), dtype=np.float32)
        b_rank = np.empty(n, dtype=np.int64); b_rank[np.argsort(-bs)] = np.arange(n)
        rrf = 1.0/(RRF_K + d_rank) + 1.0/(RRF_K + b_rank)
    else:
        bs = None
        rrf = 1.0/(RRF_K + d_rank)
    # pool = fused top FUSE_K  +  guaranteed top DENSE_FLOOR by cosine
    pool = list(dict.fromkeys(list(np.argsort(-rrf)[:FUSE_K]) + list(d_order[:DENSE_FLOOR])))
    pool = [int(i) for i in pool]
    if ce is not None:
        ce_scores = ce.predict([(query, facts[i]["text"]) for i in pool])
        ce_by_idx = {i: float(sc) for i, sc in zip(pool, ce_scores)}
    else:
        ce_by_idx = {}
    def final_score(i):
        c = float(cos[i])
        return c + (CE_WEIGHT * float(_sigmoid(ce_by_idx[i])) if i in ce_by_idx else 0.0)
    ranked = sorted(pool, key=lambda i: -final_score(i))[:TOP_K]
    hits = [dict(facts[i], score=float(cos[i]), ce=ce_by_idx.get(i),
                 bm25=float(bs[i]) if bs is not None else None)
            for i in ranked if cos[i] >= THRESHOLD]
    if debug:
        print("  [retrieval]")
        for i in ranked:
            mark = "\u2713" if cos[i] >= THRESHOLD else " "
            star = "*" if facts[i].get("source") == "curated (verified)" else " "
            cepart = f"ce={ce_by_idx[i]:+6.2f} " if i in ce_by_idx else ""
            bmpart = f"bm25={bs[i]:5.1f} " if bs is not None else ""
            print(f"   {mark}{star}{cepart}cos={cos[i]:.3f} {bmpart}f={final_score(i):.3f} "
                  f"[{facts[i].get('type','?'):4s}|{facts[i].get('source','?')[:22]}] {facts[i]['text'][:56]}")
    return hits

def build_system(hits):
    if not hits:
        return CORE + ("\n\n(No memory closely matched this question. If it asks for a specific "
                       "name, place, date, or event you don't clearly recall and it is not in what "
                       "you know cold, say so plainly — do not invent.)")
    top = max(h["score"] for h in hits)
    confident = top >= CONFIDENT
    relevant  = top >= RELEVANT
    facts = [h for h in hits if h.get("type") == "fact"]
    views = [h for h in hits if h.get("type") == "view"]
    s = CORE
    if facts:
        if confident:  head = "Facts about your life:"
        elif relevant: head = ("Possibly-related memories (use only those that truly match the "
                               "question; if none do, say you don't recall):")
        else:          head = ("Background that is probably NOT relevant here — ignore it unless "
                               "it directly answers what was asked:")
        s += "\n\n" + head + "\n" + "\n".join(f"- {h['text']}" for h in facts)
    if views:
        if confident:  vh = "Views you have expressed in your own writing (speak them as your own):"
        elif relevant: vh = "Some reflections of yours that may bear on this:"
        else:          vh = "Some reflections of yours that probably do NOT bear on this — ignore unless relevant:"
        lines = [f"- {h['text']}" + (f'  (your words: "{h["quote"]}")' if h.get("quote") else "") for h in views]
        s += "\n\n" + vh + "\n" + "\n".join(lines)
    return s

def stream_chat(messages, opts=None):
    r = requests.post(f"{OLLAMA}/api/chat",
                      json={"model":GEN_MODEL,"messages":messages,"stream":True,"options":opts or GEN_OPTS},
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

@dataclass
class MemoryCtx:
    """Everything the memory layer needs to contribute to a turn. Optional: when
    this is None, build_turn behaves exactly as it did before memory existed."""
    store: object                      # memory.MemoryStore
    profile: str = ""                  # verbatim profile.md text
    summary: str = ""                  # rolling précis of evicted turns

@dataclass
class Turn:
    """What a single assembled turn consists of.

    Replaces the old (messages, opts, hits) tuple: memory adds a fourth element and
    a 4-tuple stops being readable at call sites. Callers use turn.messages etc.
    """
    messages: list
    opts: dict
    hits: list
    memories: list = field(default_factory=list)

def build_turn(user_msg, history, facts, vecs, bm25=None, ce=None, debug=False,
               memory=None):
    """The one place a turn is assembled. Returns a Turn.

    History is a list of {"role","content"} in chronological order, and is sliced
    to HIST_WINDOW here — callers pass the full conversation.

    Every entry point (the CLI, the eval harness, the importable client) must go
    through this, so a probe run and an eval run exercise the same system:
      - task gate: analysis / long text reasons unencumbered, with no retrieval
      - follow-up fold: the previous user turn joins the retrieval query only when
        the new message leans on it
      - per-turn temperature: TEMP_TASK for tasks, TEMP_FACTUAL otherwise

    `memory` (a MemoryCtx, default None) adds a SEPARATE stream: it appends its own
    section to the system prompt and touches nothing else. It never joins `hits`,
    so the KB confidence tiers and the 0.55 gate are computed on KB scores alone.
    Memory is skipped on the task gate for the same reason facts are.
    """
    task = is_task(user_msg)
    if task:                               # analysis / long text -> reason unencumbered
        hits = []
        if debug: print("  [retrieval] skipped (task gate)")
    else:
        rquery = ((recent_user(history, 1) + " " + user_msg).strip()
                  if is_followup(user_msg) else user_msg)
        hits = retrieve(rquery, facts, vecs, bm25=bm25, ce=ce, debug=debug)

    system = build_system(hits)            # KB stream: facts + views only

    # ---- memory stream: appended AFTER the KB system prompt, never merged into it ----
    mems = []
    if memory is not None and not task:
        import memory as _mem
        try:
            mems = memory.store.retrieve(user_msg) if memory.store is not None else []
        except Exception as e:
            print(f"[memory] retrieval failed ({e}) — continuing without memory")
            mems = []
        block = _mem.build_memory_block(memory.profile, mems)
        if block:
            system += block
        if debug:
            print("  [memory]")
            if mems:
                for m in mems:
                    print(f"     score={m['score']:.3f} [{m.get('kind','?'):10s}] {m['text'][:64]}")
            else:
                print("     (no memories above threshold)")
            if memory.profile:
                print(f"     profile.md injected ({len(memory.profile)} chars)")
    if memory is not None and memory.summary:
        system += _memory_summary_block(memory.summary)

    opts = dict(GEN_OPTS, temperature=TEMP_TASK if task else TEMP_FACTUAL)
    messages = ([{"role":"system","content":system}]
                + history[-HIST_WINDOW:]
                + [{"role":"user","content":user_msg}])
    return Turn(messages=messages, opts=opts, hits=hits, memories=mems)

def _memory_summary_block(summary):
    import memory as _mem
    return _mem.build_summary_block(summary)

# ─────────────────────────────────────────────────────── memory plumbing ────
SUMMARY_TEMP    = 0.3     # compression, not voice
EXTRACT_EVERY   = 40      # exchanges: mid-session flush so a crash loses nothing
MIN_EXCHANGES   = 4       # below this a session holds nothing worth keeping

# Utility calls (summarise / extract) want an instruction-follower, not the persona:
# the persona model is trained never to break character and answers extraction prompts
# in voice instead of emitting JSON. Prefer a plain instruct model, but only if it is
# actually installed — otherwise fall back to the persona model, which still works via
# chat_extract's JSON-mode constraint.
UTIL_MODEL_PREF = os.environ.get("MEM_UTIL_MODEL", "gemma3:12b")

def _installed_models():
    try:
        r = requests.get(f"{OLLAMA}/api/tags", timeout=5)
        r.raise_for_status()
        return {m["name"] for m in r.json().get("models", [])}
    except Exception:
        return set()

def resolve_util_model(pref=None, quiet=False):
    """Return the model to use for utility calls, falling back if pref is absent."""
    pref = pref or UTIL_MODEL_PREF
    have = _installed_models()
    if not have:                       # server unreachable: don't guess, use the persona
        return GEN_MODEL
    if pref in have or f"{pref}:latest" in have:
        return pref
    if not quiet:
        print(f"[memory] utility model {pref!r} not installed — falling back to {GEN_MODEL!r}")
    return GEN_MODEL

UTIL_MODEL = None                      # resolved lazily on first utility call

UTIL_SYS = ("You are a data-extraction utility, not a persona. Follow the output format "
            "exactly. Never speak in character. Never add commentary.")

def util_chat(messages, temperature=SUMMARY_TEMP, fmt=None, system=None):
    """UTILITY calls only — routes to MEM_UTIL_MODEL, NOT the persona; never use this
    to generate a user-facing reply (that is stream_chat / GEN_MODEL).

    Non-streaming single call used for summarisation and extraction, at a lower
    temperature than chat. The model here is deliberately an instruction-follower,
    so anything generated through it will not be in voice.
    """
    global UTIL_MODEL
    if UTIL_MODEL is None:
        UTIL_MODEL = resolve_util_model()
    msgs = ([{"role": "system", "content": system}] if system else []) + messages
    body = {"model": UTIL_MODEL, "messages": msgs, "stream": False,
            "options": dict(GEN_OPTS, temperature=temperature)}
    if fmt:
        body["format"] = fmt
    r = requests.post(f"{OLLAMA}/api/chat", json=body, timeout=300)
    r.raise_for_status()
    return r.json()["message"]["content"].strip()

def chat_extract(messages):
    """chat_fn for memory.extract_memories.

    Two things fight a plain call here. The persona model is trained never to break
    character, so unconstrained it answers *as Camus* instead of emitting JSON. Ollama's
    JSON mode fixes that, but it returns a single OBJECT where the prompt asks for an
    ARRAY, and memory.parse_extraction only accepts an array. So: constrain with JSON
    mode plus a utility system prompt, then normalise an object (or an object wrapping a
    list) back into the array the parser expects. memory.py stays untouched.
    """
    # An instruction-following model answers this prompt correctly on a PLAIN call and
    # returns the array (often inside a ```json fence, which parse_extraction handles).
    # Forcing Ollama's JSON mode actively hurts it: the response collapses to a single
    # object and the "fact|preference|thread" union gets read as a literal template.
    # So try plain first, and fall back to JSON mode only when that yields nothing —
    # which is what happens with the persona model, trained never to break character.
    raw = util_chat(messages, temperature=SUMMARY_TEMP, system=UTIL_SYS)
    if _looks_like_memory_array(raw):
        return raw

    raw = util_chat(messages, temperature=SUMMARY_TEMP, fmt="json", system=UTIL_SYS)
    try:
        val = json.loads(raw)
    except json.JSONDecodeError:
        return raw                      # let parse_extraction try its regex
    if isinstance(val, dict):
        for v in val.values():          # {"memories": [...]} style
            if isinstance(v, list):
                return json.dumps(v)
        val = [val]                     # a bare single memory object
    return json.dumps(val if isinstance(val, list) else [val])

def _looks_like_memory_array(raw):
    """True if raw contains a JSON array holding at least one object with a "text"."""
    m = re.search(r"\[.*\]", raw or "", re.S)
    if not m:
        return False
    try:
        items = json.loads(m.group(0))
    except json.JSONDecodeError:
        return False
    return isinstance(items, list) and any(
        isinstance(i, dict) and str(i.get("text", "")).strip() for i in items)

def _flush_memories(mem, memctx, history, session_id, reason):
    """Extract and stage. Best effort: memory must never break the chat."""
    if memctx is None or len(history) // 2 < MIN_EXCHANGES:
        return
    try:
        cands = mem.extract_memories(history, chat_extract)
        if not cands:
            print(f"[memory] {reason}: nothing durable to keep")
            return
        summary = memctx.store.stage(cands, session_id)
        print(f"[memory] {reason}: {summary}")
        for c in cands:
            print(f"         [{c['kind']:10s}] {c['text'][:72]}")
    except Exception as e:
        print(f"[memory] extraction failed ({e}) — session not saved to memory")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--memory", action="store_true",
                    help="enable the profile + learned-memory layer (default off)")
    ap.add_argument("--forget", metavar="SUBSTRING",
                    help="delete memories containing SUBSTRING, then exit")
    ap.add_argument("--memories", action="store_true",
                    help="list stored memories, then exit")
    ap.add_argument("--reindex-memory", action="store_true",
                    help="re-embed every stored memory with the current scheme, then exit")
    args = ap.parse_args()

    # --- memory-only subcommands: no KB load, no chat ---
    if args.forget or args.memories or args.reindex_memory:
        import memory as mem
        store = mem.MemoryStore(embed)
        if args.reindex_memory:
            n_cued = sum(1 for r in store.rows if r.get("cue"))
            n = store.reindex()
            print(f"re-embedded {n} memor{'y' if n == 1 else 'ies'} "
                  f"({n_cued} with cues, {n - n_cued} without)")
        elif args.forget:
            n = store.forget(args.forget)
            store.save()
            print(f"forgot {n} memor{'y' if n == 1 else 'ies'} matching {args.forget!r}")
        else:
            if not store.rows:
                print("no memories stored yet")
            for r in store.rows:
                print(f"  [{r.get('kind','?'):10s}] hits={r.get('hits',0):<3d} "
                      f"last_seen={r.get('last_seen','?')[:10]}  {r['text']}")
                if r.get("cue"):
                    print(f"  {'':12s} cue: {r['cue']}")
                print(f"  {'':12s} id={r.get('id','?')}")
            print(f"\n{len(store.rows)} memor{'y' if len(store.rows) == 1 else 'ies'}")
        return

    facts, vecs = load_kb()
    bm25 = build_bm25(facts)
    ce = load_reranker()

    memctx, mem, session_id = None, None, None
    if args.memory:
        import memory as mem
        profile = mem.load_profile()
        memctx = MemoryCtx(store=mem.MemoryStore(embed), profile=profile, summary="")
        session_id = uuid.uuid4().hex[:8]
        print(f"[memory] on — {len(memctx.store.rows)} stored, "
              f"profile.md {'loaded' if profile else 'absent'}, session {session_id}")

    history = []
    print("\nCamus is here. Type and press enter; Ctrl-C to leave.\n")
    try:
        while True:
            user = input("you: ").strip()
            if not user: continue
            turn = build_turn(user, history, facts, vecs, bm25=bm25, ce=ce,
                              debug=args.debug, memory=memctx)
            reply = stream_chat(turn.messages, turn.opts)
            history += [{"role":"user","content":user},{"role":"assistant","content":reply}]

            if memctx is not None:
                # Turns falling out of HIST_WINDOW are compressed rather than dropped.
                if len(history) > HIST_WINDOW:
                    evicted = history[:len(history) - HIST_WINDOW]
                    history = history[len(history) - HIST_WINDOW:]
                    try:
                        memctx.summary = mem.summarise(evicted, memctx.summary, util_chat)
                        if args.debug:
                            print(f"  [memory] summary now {len(memctx.summary.split())} words")
                    except Exception as e:
                        print(f"[memory] summarisation failed ({e}) — dropping evicted turns")
                if len(history) // 2 and len(history) // 2 % EXTRACT_EVERY == 0:
                    _flush_memories(mem, memctx, history, session_id, "mid-session flush")
    except (KeyboardInterrupt, EOFError):
        print("\nUntil next time.")
        _flush_memories(mem, memctx, history, session_id, "session end")
        if memctx is not None:
            try:
                memctx.store.save()      # persist hits/last_seen bumps from retrieval
            except Exception as e:
                print(f"[memory] save failed ({e})")

if __name__ == "__main__":
    main()
