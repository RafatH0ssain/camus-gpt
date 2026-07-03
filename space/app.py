"""CamusGPT — Hugging Face ZeroGPU Space (Gradio). 8B persona on GPU + RAG grounding.

Generation runs on the 8B via transformers, wrapped in @spaces.GPU; retrieval/embeddings on
CPU. RAG: curated-boosted ranking, raw-score confidence, three-tier framing, and a TASK GATE
that skips fact injection for analysis/long prompts so the persona reasons freely.

Phase 0 hardening: streamer timeout + thread-exception propagation (no more silent hangs),
restored GPU duration budget, non-blocking logging, crisis banner stripped from history,
anaphora-gated retrieval context, tightened quota detection, per-IP daily message cap.
Set Space hardware to ZeroGPU, size 'large'.
"""
import os, re, json, hashlib
from datetime import datetime, timezone
from threading import Thread, Lock
from queue import Empty

import spaces
import numpy as np
import torch
import gradio as gr
from transformers import TextIteratorStreamer

# ---- config (override via Space Variables) ----
MODEL_REPO   = os.environ.get("MODEL_REPO", "rafatho/camus-gguf")
KB_REPO      = os.environ.get("KB_REPO",    "rafatho/camus-kb")
KB_FILE      = os.environ.get("KB_FILE",    "camus_kb_full.jsonl")
VEC_REPO     = os.environ.get("VEC_REPO",   os.environ.get("KB_REPO", "rafatho/camus-kb"))
VEC_FILE     = os.environ.get("VEC_FILE",   "camus_kb_vectors.npy")
EMBED_REPO   = os.environ.get("EMBED_REPO", "nomic-ai/nomic-embed-text-v1.5-GGUF")
EMBED_FILE   = os.environ.get("EMBED_FILE", "nomic-embed-text-v1.5.f16.gguf")
N_THREADS    = int(os.environ.get("N_THREADS", "2"))
TOP_K        = int(os.environ.get("TOP_K", "8"))
THRESHOLD    = 0.55
RELEVANT     = float(os.environ.get("RELEVANT", "0.62"))
CONFIDENT    = float(os.environ.get("CONFIDENT", "0.66"))
CURATED_BOOST = float(os.environ.get("CURATED_BOOST", "0.06"))
MAX_TOKENS   = int(os.environ.get("MAX_TOKENS", "384"))
MAX_TURNS    = int(os.environ.get("MAX_TURNS", "12"))
GPU_DURATION = int(os.environ.get("GPU_DURATION", "90"))
STREAM_TIMEOUT = float(os.environ.get("STREAM_TIMEOUT", "60"))
RATE_LIMIT_PER_DAY = int(os.environ.get("RATE_LIMIT_PER_DAY", "60"))

QUOTA_IN  = os.environ.get("QUOTA_MSG_IN",  "about 5 minutes")
QUOTA_OUT = os.environ.get("QUOTA_MSG_OUT", "about 2 minutes")

LOG_SHEET_ID = os.environ.get("LOG_SHEET_ID", "")
LOG_HASH_IP  = os.environ.get("LOG_HASH_IP", "0") == "1"
LOG_IP_SALT  = os.environ.get("LOG_IP_SALT", "camusgpt")

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

FOLLOWUP_CUES = ("and ","what about","how about","why","also ","but ","it ","that ",
                 "he ","she ","they ","was it","did he","did it","so ")

RATE_LIMIT_MESSAGE = (
 "You've reached today's message limit for this demo — it's a small project on a daily GPU "
 "budget. The counter resets at midnight UTC; please come back then. Thank you for trying it."
)

tokenizer = None; model = None; embed_model = None; KB = []; VEC = None; CURATED_MASK = None
_LOG_WS = None
_RATE = {}                 # ip -> [YYYY-MM-DD, count]
_RATE_LOCK = Lock()

def is_crisis(text): return bool(CRISIS_RE.search(text or ""))

def is_task(text):   return len(text) > 280 or any(c in text.lower() for c in TASK_CUES)

def is_followup(text):
    t = (text or "").strip().lower()
    return len(t) < 40 or any(t.startswith(c) for c in FOLLOWUP_CUES)

# ---------------------------------------------------------------- retrieval ----
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

def _hist_text(role, content):
    """History content, with the crisis banner stripped from past assistant turns so the
    model never reads helpline text as its own words (Phase 0)."""
    t = _text(content)
    if role == "assistant" and t.startswith(CRISIS_MESSAGE):
        t = t[len(CRISIS_MESSAGE):]
    return t

# ---------------------------------------------------------------- generation ----
@spaces.GPU(duration=GPU_DURATION)
def gpu_stream(messages, max_new_tokens):
    enc = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
    ).to(model.device)
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True,
                                    timeout=STREAM_TIMEOUT)
    state = {"error": None}

    def _worker():
        try:
            model.generate(**enc, streamer=streamer, max_new_tokens=max_new_tokens,
                           do_sample=True, temperature=0.6, top_k=40, min_p=0.05,
                           repetition_penalty=1.1)
        except Exception as e:            
            state["error"] = e
            try: streamer.end()          
            except Exception: pass

    Thread(target=_worker, daemon=True).start()
    acc = ""
    try:
        for piece in streamer:
            acc += piece
            yield acc
    except Empty:                          
        raise RuntimeError("generation stalled: no tokens within stream timeout") from None
    if state["error"] is not None:        
        raise state["error"]

def friendly_error(exc, logged_in):
    """Raw exception (usually ZeroGPU quota/abort) -> calm, login-aware note."""
    blob = str(exc).lower()
    is_quota = any(k in blob for k in
                   ("quota", "zerogpu", "exceeded", "gpu task aborted", "queue is full"))
    if is_quota:
        if logged_in:
            return ("⏳ That's the GPU time for this session — signed-in visitors get "
                    f"{QUOTA_IN} of generation per day, and the quota resets daily. "
                    "Please come back a little later and we'll continue.")
        return ("⏳ This session has used up the free GPU time — anonymous visitors get "
                f"{QUOTA_OUT} per day. **Sign in with Hugging Face** (top of the page) and "
                "reload for a larger allowance, or come back later.")
    return ("Something went wrong on my end while I was answering. Please try again in a "
            "moment; if it keeps happening, the Space may be waking up or restarting.")

# ---------------------------------------------------------------- rate limit ----
def _allow(ip):
    """True if this ip may send another message today (UTC). No-op when ip is unknown."""
    if not ip or RATE_LIMIT_PER_DAY <= 0:
        return True
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    with _RATE_LOCK:
        day, n = _RATE.get(ip, (today, 0))
        if day != today:
            day, n = today, 0
        if n >= RATE_LIMIT_PER_DAY:
            _RATE[ip] = (day, n)
            return False
        _RATE[ip] = (day, n + 1)
        return True

# ---------------------------------------------------------------- logging -------
def _client_ip(request):
    if request is None:
        return ""
    headers = getattr(request, "headers", {}) or {}
    fwd = headers.get("x-forwarded-for") or headers.get("X-Forwarded-For")
    ip = fwd.split(",")[0].strip() if fwd else (
        getattr(getattr(request, "client", None), "host", "") or "")
    if ip and LOG_HASH_IP:
        return "sha256:" + hashlib.sha256((LOG_IP_SALT + ip).encode()).hexdigest()[:16]
    return ip

def _get_log_ws():
    global _LOG_WS
    if _LOG_WS is not None:
        return _LOG_WS or None
    creds_json = os.environ.get("GCP_SERVICE_ACCOUNT", "")
    if not (LOG_SHEET_ID and creds_json):
        _LOG_WS = False
        return None
    try:
        import gspread
        from google.oauth2.service_account import Credentials
        creds = Credentials.from_service_account_info(
            json.loads(creds_json),
            scopes=["https://www.googleapis.com/auth/spreadsheets"])
        ws = gspread.authorize(creds).open_by_key(LOG_SHEET_ID).sheet1
        if not ws.row_values(1):
            ws.append_row(["utc_timestamp", "logged_in", "ip", "accept_language",
                           "user_agent", "user_message", "camus_response"],
                          value_input_option="RAW")
        _LOG_WS = ws
        return ws
    except Exception as e:
        print("[log] disabled — could not open sheet:", e)
        _LOG_WS = False
        return None

def log_interaction(request, logged_in, user_msg, bot_msg):
    """Best-effort append of one turn. Never raises into the chat."""
    try:
        ws = _get_log_ws()
        if ws is None:
            return
        headers = getattr(request, "headers", {}) or {}
        ws.append_row(
            [datetime.now(timezone.utc).isoformat(timespec="seconds"),
             "yes" if logged_in else "no",
             _client_ip(request),
             headers.get("accept-language", "")[:120],
             headers.get("user-agent", "")[:300],
             user_msg, bot_msg],
            value_input_option="RAW")
    except Exception as e:
        print("[log] skipped one row:", e)

def log_async(request, logged_in, user_msg, bot_msg):
    """Phase 0: logging off the reply path — the Sheets HTTP call no longer adds latency."""
    Thread(target=log_interaction, args=(request, logged_in, user_msg, bot_msg),
           daemon=True).start()

# ---------------------------------------------------------------- chat ----------
def respond(message, history, request: gr.Request = None, profile: gr.OAuthProfile = None):
    message = _text(message)
    logged_in = profile is not None
    ip = _client_ip(request)
    if not _allow(ip):                                 # cap BEFORE any GPU is touched
        log_async(request, logged_in, message, "[rate-limited]")
        yield RATE_LIMIT_MESSAGE
        return
    hist = [{"role": m["role"], "content": _hist_text(m["role"], m["content"])}
            for m in (history or [])]
    hist = hist[-(MAX_TURNS * 2):]
    if is_task(message):                               # analysis / long text -> reason unencumbered
        hits = []
    else:
        q = message
        if is_followup(message):
            prev_user = next((m["content"] for m in reversed(hist) if m["role"] == "user"), "")
            q = (prev_user + " " + message).strip()
        hits = retrieve(q)
    crisis = is_crisis(message)
    system = build_system(hits)
    if crisis:
        system += ("\n\nThe person may be in crisis. Respond with warmth and care; do NOT "
                   "philosophize about suicide or the absurd; gently encourage them to reach out "
                   "to someone they trust or a professional.")
    messages = [{"role": "system", "content": system}] + hist + [{"role": "user", "content": message}]
    prefix = CRISIS_MESSAGE if crisis else ""
    answer = ""
    try:
        if crisis:
            yield prefix
        for partial in gpu_stream(messages, MAX_TOKENS):
            answer = partial
            yield prefix + partial
    except Exception as e:
        answer = friendly_error(e, logged_in)
        yield prefix + answer
    finally:
        log_async(request, logged_in, message, prefix + answer)

# ---------------------------------------------------------------- load + UI -----
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
        raise SystemExit(f"Index ({len(VEC)}) != KB ({len(KB)}) — re-run embed_kb_llamacpp.py and re-upload.")
    CURATED_MASK = np.array([e.get("source") == "curated (verified)" for e in KB], dtype=bool)
    print(f"Ready: {len(KB)} KB entries indexed ({int(CURATED_MASK.sum())} curated).")

HEADER_HTML = """
<div class="camus-header">
  <h1>CamusGPT</h1>
  <p class="disclaimer">
    A fictional AI persona of Albert Camus, grounded in his writing and biography — not the
    real person, and not professional advice.
    Conversations and basic metadata may be logged for safety and to improve the project.
  </p>
</div>
"""

CUSTOM_CSS = """
.gradio-container { max-width: 850px !important; margin: auto; }
.camus-header { text-align: center; max-width: 620px; margin: 18px auto 4px; padding-bottom: 14px;
                border-bottom: 1px solid #dcd9d1; }
.camus-header h1 { font-family: 'Crimson Text', serif; font-size: 3em; margin: 0 0 6px;
                   font-weight: 600; color: #1a1a1a; letter-spacing: .5px; }
.camus-header .quote { font-style: italic; color: #555; font-size: 1.05em; margin: 0 0 12px; }
.camus-header .disclaimer { font-size: .85em; color: #666; line-height: 1.5; margin: 0; }
.camus-header .crisis-warn { color: #8b0000; }
.dark .camus-header { border-bottom-color: #444; }
.dark .camus-header h1 { color: #f4f3ee; }
.dark .camus-header .quote { color: #b6b2a8; }
.dark .camus-header .disclaimer { color: #9a958c; }
.dark .camus-header .crisis-warn { color: #ff7b7b; }
"""

def build():
    load()
    user_svg = ('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="#888"><path '
                'd="M12 12c2.21 0 4-1.79 4-4s-1.79-4-4-4-4 1.79-4 4 1.79 4 4 4zm0 2c-2.67 0-8 '
                '1.34-8 4v2h16v-2c0-2.66-5.33-4-8-4z"/></svg>')
    camus_svg = ('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="#888"><path '
                 'd="M3 17.25V21h3.75L17.81 9.94l-3.75-3.75L3 17.25zM20.71 7.04a.996.996 0 0 0 '
                 '0-1.41l-2.34-2.34a.996.996 0 0 0-1.41 0l-1.83 1.83 3.75 3.75 1.83-1.83z"/></svg>')
    with open("user_avatar.svg", "w", encoding="utf-8") as f:  f.write(user_svg)
    with open("camus_avatar.svg", "w", encoding="utf-8") as f: f.write(camus_svg)

    theme = gr.themes.Soft(
        primary_hue="zinc", secondary_hue="stone", neutral_hue="stone",
        font=[gr.themes.GoogleFont("Crimson Text"), "ui-sans-serif", "system-ui", "sans-serif"],
        font_mono=[gr.themes.GoogleFont("Fira Code"), "monospace"],
    ).set(
        body_background_fill="#f4f3ee", block_background_fill="#ffffff",
        block_border_color="#dcd9d1", block_border_width="1px", block_radius="4px",
        button_primary_background_fill="#2b2b2b",
        button_primary_background_fill_hover="#1a1a1a",
        button_primary_text_color="#ffffff",
    )

    with gr.Blocks(theme=theme, css=CUSTOM_CSS, title="CamusGPT") as d:
        gr.HTML(HEADER_HTML)
        if os.environ.get("SPACE_ID"):
            try:
                gr.LoginButton(size="sm")
            except Exception as e:
                print("[oauth] login button unavailable:", e)
        gr.ChatInterface(
            fn=respond,
            chatbot=gr.Chatbot(show_label=False, height=550,
                               avatar_images=("user_avatar.svg", "camus_avatar.svg")),
            textbox=gr.Textbox(placeholder="Speak to Camus...", container=False, scale=7),
        )
    d.queue(default_concurrency_limit=2, max_size=24)
    return d

demo = build() if (os.environ.get("SPACE_ID") or __name__ == "__main__") else None

if __name__ == "__main__":
    demo.launch(ssr_mode=False)
