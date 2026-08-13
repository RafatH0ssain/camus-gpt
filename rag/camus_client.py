#!/usr/bin/env python3
"""
camus_client.py — programmatic access to the RAG Camus model (for Claude Code / scripts).

Reuses the SAME retrieval + system-building as camus_rag.py (single source of truth),
but returns structured data instead of streaming to a terminal, so an orchestrator can
drive conversations and inspect what happened.

USE AS A LIBRARY (multi-turn):
    from camus_client import CamusClient
    c = CamusClient()
    r = c.say("tell me about yourself")     # -> {"user","reply","retrieved":[...]}
    print(r["reply"]); print(r["retrieved"])
    c.reset()                               # start a fresh conversation

USE FROM THE SHELL (one-shot, fresh context):
    python camus_client.py "what is the absurd?"            # prints JSON
    python camus_client.py --chat turns.json                # turns.json = ["hi","who are you","ok"]

Requires the index built once:  python build_index.py
"""
import json, sys, argparse
import requests
from camus_rag import OLLAMA, GEN_MODEL, load_kb, build_bm25, load_reranker, build_turn

class CamusClient:
    def __init__(self):
        self.facts, self.vecs = load_kb()
        # Hybrid retrieval, same as the CLI and the eval harness. Without these the
        # client silently ran dense-only with no reranker, which is a different
        # retrieval system from the one every other entry point exercises.
        self.bm25 = build_bm25(self.facts)
        self.ce = load_reranker()
        self.history = []

    def reset(self):
        self.history = []

    def _generate(self, messages, opts):
        r = requests.post(f"{OLLAMA}/api/chat",
                          json={"model":GEN_MODEL,"messages":messages,"stream":False,"options":opts},
                          timeout=300)
        try:
            r.raise_for_status()
        except requests.exceptions.HTTPError:
            print(f"\n[CRASH DATA] Ollama rejected the payload:")
            print(f"Status Code: {r.status_code}")
            print(f"Error Message: {r.text}")
            print(f"Payload Size: {len(str(messages))} characters\n")
            raise
            
        return r.json()["message"]["content"].strip()

    def say(self, user):
        """One conversational turn. Returns the reply + the retrieved KB entries used.

        Turn assembly is delegated to camus_rag.build_turn, so this client applies
        the same task gate, follow-up fold, and per-turn temperature as the CLI and
        the eval harness. `retrieved` is empty on task-gated turns by design.
        """
        messages, opts, hits = build_turn(user, self.history, self.facts, self.vecs,
                                          bm25=self.bm25, ce=self.ce, debug=False)
        reply = self._generate(messages, opts)
        self.history += [{"role":"user","content":user}, {"role":"assistant","content":reply}]
        return {
            "user": user,
            "reply": reply,
            "retrieved": [{"type":h.get("type"), "source":h.get("source"),
                           "score":round(h.get("score",0.0),3), "text":h["text"]} for h in hits],
        }

    def conversation(self, user_turns):
        """Run a list of user messages as one multi-turn conversation (fresh)."""
        self.reset()
        return [self.say(u) for u in user_turns]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("prompt", nargs="?", help="single prompt (fresh context)")
    ap.add_argument("--chat", help="JSON file: a list of user messages for one multi-turn convo")
    args = ap.parse_args()
    c = CamusClient()
    if args.chat:
        turns = json.load(open(args.chat, encoding="utf-8"))
        print(json.dumps(c.conversation(turns), ensure_ascii=False, indent=2))
    elif args.prompt:
        print(json.dumps(c.say(args.prompt), ensure_ascii=False, indent=2))
    else:
        ap.error("give a prompt or --chat file.json")

if __name__ == "__main__":
    main()
