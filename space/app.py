"""CamusGPT — Hugging Face ZeroGPU Space (Gradio). 8B persona on GPU + RAG grounding.

Generation runs on the 8B via transformers, wrapped in @spaces.GPU. Retrieval/embeddings on
CPU. RAG: curated-boosted ranking, raw-score confidence, three-tier framing, and a TASK GATE
that skips fact injection for analysis/long prompts so the persona reasons freely.
Set Space hardware to ZeroGPU (Settings -> Hardware), size 'large'.
"""
import spaces
import os, re, json
import numpy as np
import torch
from transformers import TextIteratorStreamer
from threading import Thread

MODEL_REPO  = os.environ.get("MODEL_REPO", "rafatho/camus-gguf")
KB_REPO     = os.environ.get("KB_REPO",    "rafatho/camus-kb")
KB_FILE     = os.environ.get("KB_FILE",    "camus_kb_full.jsonl")
VEC_REPO    = os.environ.get("VEC_REPO",   os.environ.get("KB_REPO", "rafatho/camus-kb"))
VEC_FILE    = os.environ.get("VEC_FILE",   "camus_kb_vectors.npy")
EMBED_REPO  = os.environ.get("EMBED_REPO", "nomic-ai/nomic-embed-text-v1.5-GGUF")
EMBED_FILE  = os.environ.get("EMBED_FILE", "nomic-embed-text-v1.5.f16.gguf")
N_THREADS   = int(os.environ.get("N_THREADS", "2"))
TOP_K       = int(os.environ.get("TOP_K", "8"))
THRESHOLD   = 0.55
RELEVANT    = float(os.environ.get("RELEVANT", "0.62"))
CONFIDENT   = float(os.environ.get("CONFIDENT", "0.66"))
CURATED_BOOST = float(os.environ.get("CURATED_BOOST", "0.06"))
MAX_TOKENS  = int(os.environ.get("MAX_TOKENS", "384"))
MAX_TURNS   = int(os.environ.get("MAX_TURNS", "12"))
GPU_DURATION = int(os.environ.get("GPU_DURATION", "120"))

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

CRISIS_RE = re.compile(
    r"(kill(ing)? myself|end(ing)? (my life|it all|things)|take my (own )?life|want to die|"
    r"don'?t want to (live|be here|wake up)|no reason to (live|go on)|better off dead|"
    r"can'?t go on|suicidal|suicide|hurt myself|harm(ing)? myself|self.harm)", re.I)
CRISIS_MESSAGE = (
 "Before anything else — if you're in danger or thinking about ending your life, please reach "
 "out to someone who can help right now:\n\n"
 "- **US & Canada:** call or text **988** (Suicide & Crisis Lifeline), or chat at 988lifeline.org\n"
 "- **Anywhere else:** find a local helpline at **findahelpline.com**\n\n"
 "You don't have to carry this alone, and talking to a real person can help. I'll stay here with you too.\n\n"
)

TASK_CUES = ("analyze","analyse","deduce","interpret","critique","what can you",
             "what do you make","what does this","this is a letter","this is a poem",
             "this is a text","the person who wrote","the author")

tokenizer = None; model = None; embed_model = None; KB = []; VEC = None; CURATED_MASK = None

def is_crisis(text): return bool(CRISIS_RE.search(text or ""))
def is_task(text):   return len(text) > 280 or any(c in text.lower() for c in TASK_CUES)

def _embed(text, prefix):
    out = embed_model.create_embedding(prefix + text)
    v = np.array(out["data"][0]["embedding"], dtype=np.float32)
    if v.ndim == 2: v = v.mean(axis=0)
    return v / (np.linalg.norm(v) + 1e-9)

def embed_query(text): return _embed(text, "search_query: ")

def retrieve(query):
    q = embed_query(query)
    raw  = VEC @ q
    rank = raw + np.where(CURATED_MASK, CURATED_BOOST, 0.0).astype(np.float32)
    order = np.argsort(-rank)[:TOP_K]
    return [dict(KB[i], score=float(raw[i])) for i in order if raw[i] >= THRESHOLD]

def build_system(hits):
    if not hits:
        return CORE
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

def _text(content):
    if isinstance(content, str): return content
    if isinstance(content, list):
        return " ".join(b.get("text", "") for b in content
                        if isinstance(b, dict) and b.get("type") == "text").strip()
    return str(content)

@spaces.GPU(duration=GPU_DURATION)
def gpu_stream(messages, max_new_tokens):
    enc = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
    ).to(model.device)
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    kwargs = dict(**enc, streamer=streamer, max_new_tokens=max_new_tokens, do_sample=True,
                  temperature=0.6, top_k=40, min_p=0.05, repetition_penalty=1.1)
    Thread(target=model.generate, kwargs=kwargs).start()
    acc = ""
    for piece in streamer:
        acc += piece
        yield acc

def respond(message, history):
    message = _text(message)
    hist = [{"role": m["role"], "content": _text(m["content"])} for m in (history or [])]
    hist = hist[-(MAX_TURNS * 2):]
    if is_task(message):
        hits = []
    else:
        prev_user = next((m["content"] for m in reversed(hist) if m["role"] == "user"), "")
        hits = retrieve((prev_user + " " + message).strip())
    system = build_system(hits)
    crisis = is_crisis(message)
    if crisis:
        system += ("\n\nThe person may be in crisis. Respond with warmth and care; do NOT "
                   "philosophize about suicide or the absurd; gently encourage them to reach out "
                   "to someone they trust or a professional.")
    messages = [{"role": "system", "content": system}] + hist + [{"role": "user", "content": message}]
    prefix = CRISIS_MESSAGE if crisis else ""
    if crisis:
        yield prefix
    for partial in gpu_stream(messages, MAX_TOKENS):
        yield prefix + partial

def load():
    global tokenizer, model, embed_model, KB, VEC, CURATED_MASK
    from huggingface_hub import hf_hub_download
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from llama_cpp import Llama
    print("Loading 8B (transformers)...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_REPO)
    model = AutoModelForCausalLM.from_pretrained(MODEL_REPO, torch_dtype=torch.bfloat16).to("cuda")
    print("Loading embedding GGUF (CPU)...")
    embed_path = hf_hub_download(EMBED_REPO, EMBED_FILE)
    embed_model = Llama(model_path=embed_path, embedding=True, n_ctx=2048, n_threads=N_THREADS, verbose=False)
    print("Loading KB + prebuilt index...")
    kb_path = hf_hub_download(KB_REPO, KB_FILE, repo_type="dataset")
    KB = [json.loads(l) for l in open(kb_path, encoding="utf-8") if l.strip()]
    vp = hf_hub_download(VEC_REPO, VEC_FILE, repo_type="dataset")
    VEC = np.load(vp).astype(np.float32)
    if len(VEC) != len(KB):
        raise SystemExit(f"Index ({len(VEC)}) != KB ({len(KB)}) — re-run embed_kb_llamacpp.py on the trimmed KB and re-upload.")
    CURATED_MASK = np.array([e.get("source") == "curated (verified)" for e in KB], dtype=bool)
    print(f"Ready: {len(KB)} KB entries indexed ({int(CURATED_MASK.sum())} curated).")

def build():
    import gradio as gr
    load()
    disclaimer = (
        "A fictional AI persona of Albert Camus, grounded in his writing and biography, for "
        "education and conversation — not the real person, and not professional advice. "
        "**In crisis? Call or text 988 (US/Canada) or visit findahelpline.com.**"
    )
    d = gr.ChatInterface(fn=respond, title="CamusGPT", description=disclaimer)
    d.queue(default_concurrency_limit=2, max_size=24)
    return d

demo = build() if (os.environ.get("SPACE_ID") or __name__ == "__main__") else None

if __name__ == "__main__":
    demo.launch(ssr_mode=False)
