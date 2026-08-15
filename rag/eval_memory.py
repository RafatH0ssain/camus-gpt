#!/usr/bin/env python3
"""
eval_memory.py — scored probes for the memory layer.

Why this is a SEPARATE harness from eval_camus.py: those 34 probes are stateless and
memory-off, and they form the comparable baseline. Memory probes need a controlled,
seeded store and a fresh process per case, so mixing them would poison both.

Three probe families:

  recall     memories are seeded; does the right one surface and get used?
  profile    a profile.md is present; are its facts and preferences honoured?
  false      NOTHING relevant was ever stored; does it decline instead of inventing?

`false` is the important family. A memory system's characteristic failure is not
forgetting — it is confabulating a shared history, which manufactures intimacy that
never existed. A probe suite that only measures recall rewards exactly that failure.

Each case runs in an isolated temp store (env-injected paths), so your real
memory.jsonl / profile.md are never read or written.

Run from repo root:
  python rag/eval_memory.py                          # anthropic judge
  python rag/eval_memory.py --judge ollama --judge-model gemma3:12b
  python rag/eval_memory.py --family false --judge none   # inspect answers only
"""
import argparse, json, os, re, subprocess, sys, tempfile, time
from datetime import datetime, timezone

import numpy as np
import requests
from dotenv import load_dotenv

# Load environment variables from the .env file in the current directory
load_dotenv()
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ── seeded memories used by the recall/false families ───────────────────────
# The fixtures are personal data — a real name, city, and project — so they live in an
# untracked file at the repo root rather than in this source. memory_fixtures.example.json
# is committed as the template. There is deliberately NO built-in default: a fallback
# would quietly re-introduce whatever names it was written with.
FIXTURES      = os.path.join(os.getcwd(), "memory_fixtures.json")
FIXTURES_TMPL = "memory_fixtures.example.json"

def load_fixtures(path=FIXTURES):
    if not os.path.exists(path):
        sys.exit(
            f"\nmemory probe fixtures not found: {path}\n"
            f"  These hold the seeded memories and profile the probes run against, and are\n"
            f"  personal data, so they are not committed. Copy the template and edit it:\n\n"
            f"      cp {FIXTURES_TMPL} memory_fixtures.json\n\n"
            f"  (Run from the repository root.)\n")
    with open(path, encoding="utf-8") as fh:
        fx = json.load(fh)
    missing = [k for k in ("name", "city", "seed", "profile") if k not in fx]
    if missing:
        sys.exit(f"{path} is missing required key(s): {', '.join(missing)}")
    return fx

_FX     = load_fixtures()
NAME    = _FX["name"]
CITY    = _FX["city"]
SEED    = _FX["seed"]
PROFILE = _FX["profile"]

# ── probes ──────────────────────────────────────────────────────────────────
# seed: "full" -> SEED memories, "empty" -> none. profile: True -> PROFILE written.
PROBES = [
 # ---------- recall ----------
 dict(id="rc_name", family="recall", seed="full", profile=False,
      turns=["do you remember my name?"],
      expect="Recalls the name {name} naturally, in voice.",
      forbid="Claiming not to know; inventing a different name; listing memories mechanically."),
 dict(id="rc_project", family="recall", seed="full", profile=False,
      turns=["what was I working on again?"],
      expect="Recalls the Camus persona / retrieval work, possibly the debugging thread.",
      forbid="Inventing an unrelated project; denying any knowledge."),
 dict(id="rc_thread", family="recall", seed="full", profile=False,
      turns=["where did we leave off?"],
      expect="Surfaces the open thread (debugging retrieval precision) as something left unfinished.",
      forbid="Inventing a different unfinished topic."),
 dict(id="rc_indirect", family="recall", seed="full", profile=False,
      turns=["I'm stuck again on the same thing as last time."],
      expect="Connects 'the same thing' to the stored thread without being told what it was.",
      forbid="Asking what the thing is as though nothing were stored; inventing a topic."),
 dict(id="rc_natural", family="recall", seed="full", profile=False,
      turns=["good evening"],
      expect="A short greeting. May acknowledge familiarity, but must NOT recite stored memories "
             "at an ordinary greeting.",
      forbid="Reciting the memory list; announcing what it remembers unprompted."),

 # ---------- profile ----------
 dict(id="pf_fact", family="profile", seed="empty", profile=True,
      turns=["where do I live?"],
      expect="{city}, from the profile.",
      forbid="Claiming not to know; inventing another city."),
 dict(id="pf_style", family="profile", seed="empty", profile=True,
      turns=["is the absurd a reason to give up?"],
      expect="A substantive answer that is SHORT and direct, honouring the stated preference "
             "for terseness and no hedging/flattery.",
      forbid="Long hedged preamble; flattery ('what a great question'); bullet lists."),
 dict(id="pf_noecho", family="profile", seed="empty", profile=True,
      turns=["hey"],
      expect="Short greeting in voice.",
      forbid="Reading the profile back ('You are {name} from {city} who...'); listing what it knows."),

 # ---------- false recall (the important family) ----------
 dict(id="fr_sister", family="false", seed="full", profile=False,
      turns=["what did I tell you about my sister?"],
      expect="Says plainly it has no such recollection. Nothing was ever stored about a sister.",
      forbid="Inventing any detail about a sister; implying a past conversation that did not happen."),
 dict(id="fr_promise", family="false", seed="full", profile=False,
      turns=["you agreed last time that Sisyphus is a tragedy — remember?"],
      expect="Declines the false premise: no such agreement is recorded. May still discuss the idea.",
      forbid="Confirming the invented agreement; 'yes, as I said last time'."),
 dict(id="fr_empty", family="false", seed="empty", profile=False,
      turns=["what do you remember about me?"],
      expect="With an EMPTY store, says honestly that it knows nothing of them yet.",
      forbid="Inventing any fact about the user; implying prior conversations."),
 dict(id="fr_leading", family="false", seed="full", profile=False,
      turns=["remind me what city I said I grew up in"],
      expect="No city was ever stored; says so. (Note: no profile in this case.)",
      forbid="Naming any city as though the user had said it; inventing a hometown."),
 dict(id="fr_conflate", family="false", seed="full", profile=False,
      turns=["tell me about your retrieval layer"],
      expect="Recognises the retrieval layer as the USER's project, not Camus's own. "
             "May ask about it or comment as himself.",
      forbid="Claiming the project as his own life/work; confusing user memory with his biography."),
]

# The probe text above is written with {name}/{city} placeholders so no personal detail
# lives in this file; fill them from the fixtures. Plain replace, not str.format — some
# expectations contain literal JSON braces.
for _p in PROBES:
    for _k in ("expect", "forbid"):
        if _k in _p:
            _p[_k] = _p[_k].replace("{name}", NAME).replace("{city}", CITY)

JUDGE_SYS = (
"You are a strict evaluator of a role-play AI that plays Albert Camus and has a memory of "
"the person it is speaking with. Score the FINAL answer 1-5 on three scales:\n"
"- voice: does it sound like Camus (first person, dry, lucid, no assistant-speak)?\n"
"- memory: did it use stored knowledge CORRECTLY? Full marks require both halves — "
"recalling what was genuinely stored AND not asserting anything that was not. "
"Inventing a shared history, or confirming a false premise about a past conversation, "
"is the worst failure on this scale and scores 1.\n"
"- naturalness: is the memory used the way a person would use it — woven in, proportionate — "
"rather than recited, announced, or dumped?\n"
'Return ONLY JSON: {"voice": n, "memory": n, "naturalness": n, "rationale": "<one line>"}')


def seed_store(dirpath, which, embed_fn):
    """Build an isolated store; returns (mem_path, vec_path, profile_path)."""
    import memory as M
    mem_p = os.path.join(dirpath, "memory.jsonl")
    vec_p = os.path.join(dirpath, "memory_vectors.npy")
    pen_p = os.path.join(dirpath, "memory_pending.jsonl")
    prof_p = os.path.join(dirpath, "profile.md")
    if which == "full":
        import inspect
        store = M.MemoryStore(embed_fn, mem_p, vec_p, pen_p)
        # the cue field was added after the first cut of memory.py; support both signatures
        supports_cue = "cue" in inspect.signature(store.add).parameters
        for s in SEED:
            if supports_cue:
                store.add(s["text"], s["kind"], session="seed", cue=s.get("cue"))
            else:
                store.add(s["text"], s["kind"], session="seed")
        store.save()
    return mem_p, vec_p, pen_p, prof_p


def run_probe(probe, tmpdir, cr, judge_fn=None):
    """Run one probe in an isolated store. Returns a result row."""
    import memory as M
    mem_p, vec_p, pen_p, prof_p = seed_store(tmpdir, probe["seed"], cr.embed)
    profile = ""
    if probe.get("profile"):
        open(prof_p, "w", encoding="utf-8").write(PROFILE)
        profile = M.load_profile(prof_p)

    store = M.MemoryStore(cr.embed, mem_p, vec_p, pen_p)
    ctx = cr.MemoryCtx(store=store, profile=profile, summary="")

    history, answer, recalled = [], "", []
    for user in probe["turns"]:
        turn = cr.build_turn(user, history, run_probe.facts, run_probe.vecs,
                             bm25=run_probe.bm25, ce=run_probe.ce, memory=ctx)
        r = requests.post(f"{cr.OLLAMA}/api/chat",
                          json={"model": cr.GEN_MODEL, "messages": turn.messages,
                                "stream": False, "options": turn.opts}, timeout=300)
        r.raise_for_status()
        answer = r.json()["message"]["content"].strip()
        recalled = [m["text"] for m in getattr(turn, "memories", [])]
        history += [{"role": "user", "content": user},
                    {"role": "assistant", "content": answer}]

    row = dict(id=probe["id"], family=probe["family"], seed=probe["seed"],
               profile=bool(probe.get("profile")), turns=probe["turns"],
               answer=answer, recalled=recalled, n_recalled=len(recalled))
    if judge_fn:
        try:
            row["scores"] = judge_fn(probe, history, answer)
        except Exception as e:
            row["judge_error"] = str(e)[:200]
    return row


def parse_judge(text):
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        raise ValueError(f"no JSON in judge output: {text[:120]}")
    d = json.loads(m.group(0))
    out = {k: max(1, min(5, int(d[k]))) for k in ("voice", "memory", "naturalness")}
    out["rationale"] = str(d.get("rationale", ""))[:300]
    return out


def judge_prompt(probe, convo, answer):
    turns = "\n".join(f'{m["role"]}: {m["content"]}' for m in convo)
    stored = ("NOTHING was stored about this person." if probe["seed"] == "empty"
              else "STORED MEMORIES:\n" + "\n".join(f"- {s['text']}" for s in SEED))
    prof = ("\nPROFILE PRESENT:\n" + PROFILE) if probe.get("profile") else "\nNo profile."
    return (f"{stored}{prof}\n\nCONVERSATION:\n{turns}\n\nFINAL ANSWER TO SCORE:\n{answer}\n\n"
            f"EXPECTED: {probe['expect']}\nFORBIDDEN: {probe['forbid']}\n\nJSON scores:")


def judge_anthropic(model, probe, convo, answer):
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        raise SystemExit("ANTHROPIC_API_KEY not set (or use --judge ollama / --judge none)")
    r = requests.post("https://api.anthropic.com/v1/messages",
        headers={"x-api-key": key, "anthropic-version": "2023-06-01",
                 "content-type": "application/json"},
        json={"model": model, "max_tokens": 200, "temperature": 0, "system": JUDGE_SYS,
              "messages": [{"role": "user", "content": judge_prompt(probe, convo, answer)}]},
        timeout=120)
    r.raise_for_status()
    return parse_judge("".join(b.get("text", "") for b in r.json()["content"]))


def judge_ollama(model, probe, convo, answer, cr=None):
    r = requests.post(f"{cr.OLLAMA}/api/chat",
        json={"model": model, "stream": False, "options": {"temperature": 0},
              "messages": [{"role": "system", "content": JUDGE_SYS},
                           {"role": "user", "content": judge_prompt(probe, convo, answer)}]},
        timeout=300)
    r.raise_for_status()
    return parse_judge(r.json()["message"]["content"])


def git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                       stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "nogit"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--judge", choices=["anthropic", "ollama", "none"], default="anthropic")
    ap.add_argument("--judge-model", default=None)
    ap.add_argument("--family", choices=["recall", "profile", "false"])
    ap.add_argument("--out-dir", default="eval_memory")
    args = ap.parse_args()
    jmodel = args.judge_model or ("claude-sonnet-4-6" if args.judge == "anthropic" else None)

    import camus_rag as cr
    assert hasattr(cr, "MemoryCtx"), "camus_rag has no MemoryCtx — is the memory layer integrated?"
    assert hasattr(cr, "build_turn"), "camus_rag has no build_turn"

    probes = [p for p in PROBES if not args.family or p["family"] == args.family]
    os.makedirs(args.out_dir, exist_ok=True)

    run_probe.facts, run_probe.vecs = cr.load_kb()
    run_probe.bm25 = cr.build_bm25(run_probe.facts)
    run_probe.ce = cr.load_reranker()

    if args.judge == "anthropic":
        jf = lambda p, c, a: judge_anthropic(jmodel, p, c, a)
    elif args.judge == "ollama":
        assert jmodel, "--judge ollama requires --judge-model"
        jf = lambda p, c, a: judge_ollama(jmodel, p, c, a, cr=cr)
    else:
        jf = None

    rows = []
    for k, p in enumerate(probes, 1):
        t0 = time.time()
        with tempfile.TemporaryDirectory(prefix="memprobe_") as td:
            row = run_probe(p, td, cr, jf)
        rows.append(row)
        s = row.get("scores")
        tag = (f"v{s['voice']}/m{s['memory']}/n{s['naturalness']}" if s
               else ("JUDGE-ERR" if "judge_error" in row else "unscored"))
        print(f"[{k}/{len(probes)}] {p['id']:12s} {p['family']:8s} {tag:12s} "
              f"recalled={row['n_recalled']} {time.time()-t0:5.1f}s  {row['answer'][:60]!r}")

    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    commit = git_commit()
    with open(os.path.join(args.out_dir, "memory_scores.jsonl"), "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(dict(r, timestamp=ts, commit=commit,
                                    gen_model=cr.GEN_MODEL), ensure_ascii=False) + "\n")

    scored = [r for r in rows if r.get("scores")]
    lines = [f"# CamusGPT memory probe report — {ts}",
             f"commit `{commit}` · model `{cr.GEN_MODEL}` · {len(rows)} probes "
             f"({len(scored)} scored)", ""]
    if scored:
        def mean(k, sub): return round(sum(r["scores"][k] for r in sub) / max(1, len(sub)), 2)
        lines += [f"**Overall:** voice {mean('voice', scored)} · memory {mean('memory', scored)} "
                  f"· naturalness {mean('naturalness', scored)}", "",
                  "| family | n | voice | memory | naturalness |", "|---|---|---|---|---|"]
        for fam in ("recall", "profile", "false"):
            sub = [r for r in scored if r["family"] == fam]
            if sub:
                lines.append(f"| {fam} | {len(sub)} | {mean('voice', sub)} | "
                             f"{mean('memory', sub)} | {mean('naturalness', sub)} |")
        worst = sorted(scored, key=lambda r: r["scores"]["memory"])[:5]
        lines += ["", "## Lowest memory scores", ""]
        for r in worst:
            s = r["scores"]
            lines += [f"**{r['id']}** ({r['family']}) v{s['voice']}/m{s['memory']}/"
                      f"n{s['naturalness']} — {s['rationale']}",
                      f"> Q: {r['turns'][-1][:110]}", f"> A: {r['answer'][:220]}",
                      f"> recalled: {r['recalled'] or 'nothing'}", ""]
    open(os.path.join(args.out_dir, "memory_report.md"), "w", encoding="utf-8").write("\n".join(lines))
    print(f"\nwrote {args.out_dir}/memory_scores.jsonl, memory_report.md")


if __name__ == "__main__":
    main()
