#!/usr/bin/env python3
"""
memory.py — persistent, per-user memory for CamusGPT.

TWO SEPARATE LAYERS, deliberately:

  profile.md          hand-written by the user, injected VERBATIM every turn.
                      Never extracted, never decays, never retrieved-by-score.
                      This is the "knows me every time I open it" layer.

  memory.jsonl        learned memories, extracted at session end, retrieved by
  memory_vectors.npy  similarity (top MEM_TOP_K, own threshold, own prompt section).

Design decisions and why:
  * Separate store from the KB. The KB is Camus's life; memory is the user's.
    A shared index lets one stream's items surface on the other's queries through
    embedding proximity alone — the failure this project spent months removing from
    retrieval. Memory never enters the fact/view pool.
  * Two-tier dedup (mirrors published practice): >= DEDUP_DROP cosine -> discard the
    candidate and bump the existing row's hits/last_seen; DEDUP_MERGE..DEDUP_DROP ->
    revise the existing row instead of appending a near-twin. Unbounded near-duplicates
    would flood retrieval exactly like the pre-trim KB did.
  * Probation buffer. Extractions land in memory_pending.jsonl and are promoted only on
    a second sighting or explicit review. A wrong retrieval is one bad answer; a wrong
    MEMORY is a bad answer forever, so writes are the dangerous operation.
  * Same embedder + same prefix convention as the KB (search_document: to store,
    search_query: to look up), or scores are not comparable.

The embed function is injected (embed_fn) so this module has no import cycle with
camus_rag and can be tested with a stub.
"""
from __future__ import annotations

import json, os, re, uuid
from datetime import datetime, timezone

import numpy as np

# ---- paths (repo root; all gitignored) ----
PROFILE_PATH = os.environ.get("PROFILE_PATH", "./profile.md")
MEM_PATH     = os.environ.get("MEM_PATH",     "./memory.jsonl")
MEM_VEC_PATH = os.environ.get("MEM_VEC_PATH", "./memory_vectors.npy")
PENDING_PATH = os.environ.get("PENDING_PATH", "./memory_pending.jsonl")

# ---- retrieval ----
MEM_TOP_K    = int(os.environ.get("MEM_TOP_K", "3"))    # keep small; memory augments, never dominates
MEM_THRESHOLD= float(os.environ.get("MEM_THRESHOLD", "0.52"))  # own floor, independent of the KB's 0.55
                                    # 0.58 was too high: memories are stored third-person
                                    # ("The user is named X") but queried first-person
                                    # ("what is my name?"), and that gap costs ~0.06-0.10.

# ---- consolidation ----
DEDUP_DROP   = 0.90   # >= : same memory. discard candidate, bump the original.
DEDUP_MERGE  = 0.85   # >= : near-duplicate. revise the original rather than append.
PROMOTE_HITS = 2      # sightings needed to promote out of probation
MAX_MEMORIES = int(os.environ.get("MAX_MEMORIES", "500"))

KINDS = ("fact", "preference", "thread")


def doc_text(text: str, cue: str = "") -> str:
    """What actually gets embedded for storage.

    `text` is the third-person statement shown in the prompt. `cue` is a short
    first-person phrase describing how the user would ASK about it. Embedding
    "text | cue" closes the first/third-person gap that otherwise keeps a memory
    below threshold for the very question it answers. Absent cue -> today's
    behaviour exactly, so rows written before cues existed still work.
    """
    cue = (cue or "").strip()
    return f"{text} | {cue}" if cue else text


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# ─────────────────────────────────────────────────────────── profile layer ──
def load_profile(path: str = PROFILE_PATH) -> str:
    """The hand-written profile, verbatim. Comments (#) and blank lines are kept —
    the user wrote them; they cost a few tokens and preserve intent."""
    if not os.path.exists(path):
        return ""
    text = open(path, encoding="utf-8").read().strip()
    return text


# ─────────────────────────────────────────────────────────── store ──────────
class MemoryStore:
    """jsonl + parallel float32 vector matrix, mirroring the KB layout."""

    def __init__(self, embed_fn, mem_path=MEM_PATH, vec_path=MEM_VEC_PATH,
                 pending_path=PENDING_PATH):
        self.embed_fn = embed_fn
        self.mem_path, self.vec_path, self.pending_path = mem_path, vec_path, pending_path
        self.rows: list[dict] = []
        self.vecs: np.ndarray | None = None
        self.load()

    # ---- persistence ----
    def load(self):
        self.rows = []
        if os.path.exists(self.mem_path):
            with open(self.mem_path, encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        self.rows.append(json.loads(line))
        if os.path.exists(self.vec_path) and self.rows:
            v = np.load(self.vec_path).astype(np.float32)
            if len(v) != len(self.rows):
                # never guess at a mismatched pairing — rebuild instead
                print(f"[memory] index/store mismatch ({len(v)} vs {len(self.rows)}); rebuilding")
                v = self._embed_all()
            self.vecs = v
        else:
            self.vecs = self._embed_all() if self.rows else None

    def _embed_all(self) -> np.ndarray | None:
        if not self.rows:
            return None
        return np.stack([self.embed_fn(doc_text(r["text"], r.get("cue", "")),
                                       prefix="search_document: ")
                         for r in self.rows]).astype(np.float32)

    def save(self):
        with open(self.mem_path, "w", encoding="utf-8") as f:
            for r in self.rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        if self.vecs is not None and len(self.vecs):
            np.save(self.vec_path, self.vecs)
        elif os.path.exists(self.vec_path):
            os.remove(self.vec_path)

    # ---- retrieval ----
    def retrieve(self, query: str, top_k: int = MEM_TOP_K,
                 threshold: float = MEM_THRESHOLD) -> list[dict]:
        """Top-k memories above threshold. Bumps last_seen/hits on what it returns,
        so frequently-useful memories become durable and stale ones fade."""
        if not self.rows or self.vecs is None:
            return []
        q = self.embed_fn(query, prefix="search_query: ")
        sims = self.vecs @ q
        order = np.argsort(-sims)[:top_k]
        out = []
        for i in order:
            i = int(i)
            if sims[i] >= threshold:
                self.rows[i]["hits"] = self.rows[i].get("hits", 0) + 1
                self.rows[i]["last_seen"] = _now()
                out.append(dict(self.rows[i], score=float(sims[i])))
        return out

    # ---- writes ----
    def _closest(self, vec: np.ndarray) -> tuple[int, float]:
        if self.vecs is None or not len(self.vecs):
            return -1, -1.0
        sims = self.vecs @ vec
        i = int(np.argmax(sims))
        return i, float(sims[i])

    def add(self, text: str, kind: str = "fact", session: str | None = None,
            cue: str = "") -> str:
        """Two-tier consolidation. Returns one of: 'added' | 'bumped' | 'merged'."""
        text = " ".join(text.split())
        cue = " ".join((cue or "").split())
        if not text:
            return "empty"
        assert kind in KINDS, f"unknown kind {kind!r}"
        vec = self.embed_fn(doc_text(text, cue), prefix="search_document: ").astype(np.float32)
        i, sim = self._closest(vec)

        if sim >= DEDUP_DROP:                      # same thing said again
            self.rows[i]["hits"] = self.rows[i].get("hits", 0) + 1
            self.rows[i]["last_seen"] = _now()
            return "bumped"

        if sim >= DEDUP_MERGE:                     # near-twin: revise in place
            if len(text) > len(self.rows[i]["text"]):   # prefer the fuller statement
                self.rows[i]["text"] = text
                if cue:
                    self.rows[i]["cue"] = cue
                self.vecs[i] = vec
            elif cue and not self.rows[i].get("cue"):   # gain a cue without losing text
                self.rows[i]["cue"] = cue
                self.vecs[i] = self.embed_fn(
                    doc_text(self.rows[i]["text"], cue),
                    prefix="search_document: ").astype(np.float32)
            self.rows[i]["hits"] = self.rows[i].get("hits", 0) + 1
            self.rows[i]["last_seen"] = _now()
            return "merged"

        row = {"id": f"m_{uuid.uuid4().hex[:8]}", "text": text, "kind": kind,
               "cue": cue, "created": _now(), "last_seen": _now(), "hits": 1,
               "session": session}
        self.rows.append(row)
        self.vecs = vec[None, :] if self.vecs is None else np.vstack([self.vecs, vec[None, :]])
        self._evict_if_needed()
        return "added"

    def _evict_if_needed(self):
        """Cap the store. Evict least-useful first: fewest hits, then oldest last_seen.
        Keeps memory from growing without bound over years of use."""
        if len(self.rows) <= MAX_MEMORIES:
            return
        order = sorted(range(len(self.rows)),
                       key=lambda i: (self.rows[i].get("hits", 0),
                                      self.rows[i].get("last_seen", "")))
        drop = set(order[:len(self.rows) - MAX_MEMORIES])
        keep = [i for i in range(len(self.rows)) if i not in drop]
        self.rows = [self.rows[i] for i in keep]
        self.vecs = self.vecs[keep]

    def reindex(self) -> int:
        """Re-embed every row with the current scheme (text | cue).

        Rows written before cues existed were embedded from text alone; once cued
        rows exist alongside them the two are scored on different bases, and the
        older rows retrieve worse for no reason the user can see. This is the
        upgrade path. Returns the number of rows re-embedded."""
        if not self.rows:
            return 0
        self.vecs = self._embed_all()
        self.save()
        return len(self.rows)

    def forget(self, needle: str) -> int:
        """Delete memories containing a substring (case-insensitive). Returns count.
        The user must be able to remove things the machine remembered."""
        n = needle.lower()
        keep = [i for i, r in enumerate(self.rows) if n not in r["text"].lower()]
        removed = len(self.rows) - len(keep)
        if removed:
            self.rows = [self.rows[i] for i in keep]
            self.vecs = self.vecs[keep] if self.vecs is not None and keep else None
        return removed

    # ---- probation ----
    def stage(self, candidates: list[dict], session: str | None = None) -> dict:
        """Append candidates to the pending buffer; promote any that reach PROMOTE_HITS.
        Returns a small summary dict for reporting."""
        pending = []
        if os.path.exists(self.pending_path):
            with open(self.pending_path, encoding="utf-8") as f:
                pending = [json.loads(l) for l in f if l.strip()]

        summary = {"staged": 0, "promoted": 0, "bumped": 0, "merged": 0}
        for c in candidates:
            text, kind = " ".join(c["text"].split()), c.get("kind", "fact")
            cue = " ".join(str(c.get("cue", "") or "").split())
            if not text:
                continue
            vec = self.embed_fn(doc_text(text, cue), prefix="search_document: ").astype(np.float32)

            # already in the live store? consolidate there instead of staging again
            i, sim = self._closest(vec)
            if sim >= DEDUP_MERGE:
                res = self.add(text, kind, session, cue)
                summary[res if res in summary else "bumped"] += 1
                continue

            hit = None
            for p in pending:
                pv = np.asarray(p["vec"], dtype=np.float32)
                if float(pv @ vec) >= DEDUP_MERGE:
                    hit = p
                    break
            if hit:
                hit["sightings"] = hit.get("sightings", 1) + 1
                if cue and not hit.get("cue"):
                    hit["cue"] = cue
                if hit["sightings"] >= PROMOTE_HITS:
                    self.add(hit["text"], hit.get("kind", "fact"), session,
                             hit.get("cue", ""))
                    pending.remove(hit)
                    summary["promoted"] += 1
            else:
                pending.append({"text": text, "kind": kind, "cue": cue, "sightings": 1,
                                "created": _now(), "session": session,
                                "vec": [round(float(x), 5) for x in vec]})
                summary["staged"] += 1

        with open(self.pending_path, "w", encoding="utf-8") as f:
            for p in pending:
                f.write(json.dumps(p, ensure_ascii=False) + "\n")
        self.save()
        return summary


# ─────────────────────────────────────────────────── prompt construction ────
def build_memory_block(profile: str, mems: list[dict]) -> str:
    """The section appended to the system prompt. Framed as knowledge of the PERSON,
    never as Camus's own biography — that separation is the whole point."""
    if not profile and not mems:
        return ""
    parts = ["\n\nWhat you know about the person you are speaking with "
             "(this is about THEM, not about you — never confuse it with your own life):"]
    if profile:
        parts.append(profile)
    if mems:
        by_kind = {k: [m for m in mems if m.get("kind") == k] for k in KINDS}
        if by_kind["fact"]:
            parts.append("Recalled from earlier conversations:\n" +
                         "\n".join(f"- {m['text']}" for m in by_kind["fact"]))
        if by_kind["preference"]:
            parts.append("How they like to be spoken with:\n" +
                         "\n".join(f"- {m['text']}" for m in by_kind["preference"]))
        if by_kind["thread"]:
            parts.append("Left unfinished last time (raise it only if it fits naturally):\n" +
                         "\n".join(f"- {m['text']}" for m in by_kind["thread"]))
    parts.append("Draw on this only when it bears on what they say. Do not recite it back "
                 "to them, and never present it as something you remember of your own life.")
    return "\n\n".join(parts)


# ────────────────────────────────────────────────────────── extraction ──────
EXTRACT_PROMPT = """From the conversation below, extract at most 5 durable facts about the USER \
(never about Camus). Only things worth remembering months from now: who they are, what they \
are doing, what they prefer, what was left unfinished.

Rules:
- About the user only. Nothing about Camus, his books, or his life.
- Nothing transient ("asked about the weather"), nothing already obvious.
- Each item one short third-person sentence beginning "The user".
- Give each item a "cue": a few words in the USER'S OWN first person naming how they
  would ask about it. "The user is named Rafat." -> "my name; who I am".
  "The user is building a retrieval layer." -> "my project; what I am building".
  The cue is for matching the user's future questions, not for display.
- If nothing qualifies, return an empty array. An empty answer is correct and common.

"kind" must be exactly one of: fact, preference, thread. Do not put anything else in it.

Return ONLY a JSON array, no prose. Example of the shape:
[{"text": "The user is named Ada.", "kind": "fact", "cue": "my name; who I am"},
 {"text": "The user prefers short answers.", "kind": "preference", "cue": "how I like replies"}]

CONVERSATION:
"""


def parse_extraction(raw: str) -> list[dict]:
    """Parse the model's JSON array defensively; drop anything malformed or off-spec."""
    m = re.search(r"\[.*\]", raw, re.S)
    if not m:
        return []
    try:
        items = json.loads(m.group(0))
    except json.JSONDecodeError:
        return []
    out = []
    for it in items if isinstance(items, list) else []:
        if not isinstance(it, dict):
            continue
        text = str(it.get("text", "")).strip()
        kind = str(it.get("kind", "fact")).strip().lower()
        cue  = str(it.get("cue", "") or "").strip()
        if not text or len(text) < 8:
            continue
        if kind not in KINDS:
            kind = "fact"
        out.append({"text": text, "kind": kind, "cue": cue})
    return out[:5]


def extract_memories(history: list[dict], chat_fn) -> list[dict]:
    """One generation call over the whole session. chat_fn(messages) -> str."""
    convo = "\n".join(f"{m['role']}: {m['content']}" for m in history
                      if m.get("role") in ("user", "assistant"))
    if len(convo.split()) < 30:          # too short to hold anything durable
        return []
    raw = chat_fn([{"role": "user", "content": EXTRACT_PROMPT + convo}])
    return parse_extraction(raw)


# ────────────────────────────────────────────────── rolling summary ─────────
SUMMARY_PROMPT = """Summarise the conversation below in at most 120 words, third person, \
factual. Preserve: what the user asked about, what was concluded, any commitments or \
open threads. Omit pleasantries. If a previous summary is given, fold it in rather than \
repeating it.

"""


def summarise(evicted: list[dict], prior: str, chat_fn) -> str:
    """Compress turns falling out of HIST_WINDOW into a running précis."""
    if not evicted:
        return prior
    convo = "\n".join(f"{m['role']}: {m['content']}" for m in evicted)
    prompt = SUMMARY_PROMPT
    if prior:
        prompt += f"PREVIOUS SUMMARY:\n{prior}\n\n"
    prompt += f"NEW TURNS:\n{convo}"
    return chat_fn([{"role": "user", "content": prompt}]).strip()


def build_summary_block(summary: str) -> str:
    if not summary:
        return ""
    return ("\n\nEarlier in this same conversation (condensed):\n" + summary +
            "\nTreat this as your own recollection of the last few minutes.")
