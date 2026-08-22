#!/usr/bin/env python3
"""
ab_session.py — blind A/B chat sessions for human naturalness judging.

The LLM judge saturates on voice (4.57/5), so the human is the instrument. This runs the
two configurations under test back to back behind opaque labels, logs both halves, and
withholds which was which until asked.

    python rag/ab_session.py --compare core:current core:lean
    python rag/ab_session.py --compare temp:0.45 temp:0.75
    python rag/ab_session.py --reveal 2026-08-15_141230

Config specs:
    core:current | core:lean      the CORE prose variant (identity card is identical in both)
    temp:<float>                  TEMP_FACTUAL for the session
    model:<name>                  an Ollama model name, e.g. model:camus

Everything lands in eval_human/<timestamp>/ — key.json, session_A.md, session_B.md,
scores.md. That directory is gitignored: it holds real conversations.

Memory is OFF unless --memory is passed, so the only difference between halves is the
variable under test. The topic list and the score sheet are read from
docs/HUMAN_EVAL_PROTOCOL.md rather than duplicated here — edit the protocol, not this file.
"""
import argparse, datetime, json, os, random, sys, re

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

ROOT      = os.getcwd()
OUT_ROOT  = os.path.join(ROOT, "eval_human")
PROTOCOL  = os.path.join(ROOT, "docs", "HUMAN_EVAL_PROTOCOL.md")


# ───────────────────────────────────────────────────────────── protocol ─────
def _section(text, keyword):
    """Body of the first '## ...' heading containing `keyword`, up to the next '## '.
    Matched loosely on purpose: the protocol is an authored document, and its headings
    ('The eight topics', 'Score sheet (fill for each configuration...)') are prose, not
    parser input. Do not tighten this into an exact match."""
    for m in re.finditer(r"^##\s+(.+)$", text, re.M):
        if keyword.lower() in m.group(1).lower():
            rest = text[m.end():]
            nxt = re.search(r"^##\s+", rest, re.M)
            return (rest[:nxt.start()] if nxt else rest).strip("\n")
    return ""


def _parse_topics(body):
    """Topics come either as a markdown table (| # | Topic | Open with... |) or as a
    numbered list. Handle both; the authored protocol uses the table."""
    topics = []
    for ln in body.splitlines():
        ln = ln.strip()
        if ln.startswith("|"):
            cells = [c.strip() for c in ln.strip("|").split("|")]
            if len(cells) >= 2 and re.fullmatch(r"\d+", cells[0]):
                name = re.sub(r"\*\*|\*|`", "", cells[1]).strip()
                hint = re.sub(r"\*\*|\*|`", "", cells[2]).strip() if len(cells) > 2 else ""
                topics.append(f"{name} — {hint}" if hint else name)
        elif re.match(r"^\d+[.)]\s+", ln):
            topics.append(re.sub(r"^\d+[.)]\s*", "", ln).strip())
    return topics


def load_protocol():
    """Topics and score sheet come from the protocol doc so there is one source of truth."""
    if not os.path.exists(PROTOCOL):
        sys.exit(f"protocol not found: {PROTOCOL}\n"
                 f"  ab_session reads its topic list and score sheet from that file.")
    text = open(PROTOCOL, encoding="utf-8").read()
    topics = _parse_topics(_section(text, "topics"))
    sheet  = _section(text, "score sheet")
    if not topics:
        sys.exit(f"no topics found under a '## ...topics...' heading in {PROTOCOL}")
    if not sheet:
        sys.exit(f"no '## ...score sheet...' section found in {PROTOCOL}")
    return topics, sheet


# ───────────────────────────────────────────────────────────── configs ──────
def parse_spec(spec):
    if ":" not in spec:
        sys.exit(f"bad config spec {spec!r} — expected kind:value, e.g. core:lean or temp:0.7")
    kind, value = spec.split(":", 1)
    kind = kind.strip().lower()
    if kind == "core":
        if value not in ("current", "lean"):
            sys.exit(f"core variant must be 'current' or 'lean', got {value!r}")
    elif kind == "temp":
        try:
            float(value)
        except ValueError:
            sys.exit(f"temp must be a number, got {value!r}")
    elif kind != "model":
        sys.exit(f"unknown config kind {kind!r} — use core:, temp:, or model:")
    return dict(spec=spec, kind=kind, value=value)


def apply_config(cr, cfg):
    """Mutate the imported module in place. Both halves run in one process, so whatever
    this sets must be fully overwritten by the other config — never left half-applied."""
    if cfg["kind"] == "core":
        prose = cr._CORE_PROSE_LEAN if cfg["value"] == "lean" else cr._CORE_PROSE_CURRENT
        cr.CORE = prose + cr._IDENTITY_CARD
        cr.CORE_VARIANT = cfg["value"]
    elif cfg["kind"] == "temp":
        cr.TEMP_FACTUAL = float(cfg["value"])
    elif cfg["kind"] == "model":
        cr.GEN_MODEL = cfg["value"]


# ───────────────────────────────────────────────────────────── session ──────
BANNER = """
────────────────────────────────────────────────────────────────────────
  Session {label}   ({idx} of 2)
────────────────────────────────────────────────────────────────────────
  Cover all eight. 4-6 turns each. Use the SAME opening line in both halves.
  Don't be polite to it — interrupt, disagree, change subject, be boring.

{topics}

  /done   finish this half        /skip   note a topic as skipped
────────────────────────────────────────────────────────────────────────
"""


def run_session(cr, label, idx, res, topics, outdir, use_memory):
    print(BANNER.format(label=label, idx=idx,
                        topics="\n".join(f"   {i}. {t}" for i, t in enumerate(topics, 1))))
    facts, vecs, bm25, ce = res

    memctx = None
    if use_memory:
        import memory as M
        store = M.MemoryStore(cr.embed, "memory.jsonl", "memory_vectors.npy",
                              "memory_pending.jsonl")
        memctx = cr.MemoryCtx(store=store, profile="", summary="")

    path = os.path.join(outdir, f"session_{label}.md")
    log = open(path, "w", encoding="utf-8")
    log.write(f"# Session {label}\n\n"
              f"Started {datetime.datetime.now().isoformat(timespec='seconds')}. "
              f"Memory {'ON' if use_memory else 'off'}.\n\n"
              f"<!-- configuration deliberately not recorded here; see key.json -->\n\n")
    log.flush()

    history, turns = [], 0
    while True:
        try:
            user = input("\nyou › ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n[ending this half]")
            break
        if not user:
            continue
        if user in ("/done", "/quit", "/exit"):
            break
        if user == "/skip":
            log.write("_(topic skipped)_\n\n"); log.flush()
            continue

        turn = cr.build_turn(user, history, facts, vecs, bm25, ce, memory=memctx)
        print()                                  # stream_chat prints its own "camus: " prefix
        answer = "".join(cr.stream_chat(turn.messages, turn.opts)).strip()

        history += [{"role": "user", "content": user},
                    {"role": "assistant", "content": answer}]
        turns += 1
        log.write(f"**you:** {user}\n\n**camus:** {answer}\n\n---\n\n")
        log.flush()

    log.write(f"\n_{turns} exchanges._\n")
    log.close()
    print(f"\n[session {label} logged to {os.path.relpath(path, ROOT)}]")
    return turns


# ────────────────────────────────────────────────────────────── reveal ──────
def reveal(stamp):
    outdir = os.path.join(OUT_ROOT, stamp)
    keyfile = os.path.join(outdir, "key.json")
    if not os.path.exists(keyfile):
        sys.exit(f"no key at {keyfile}")
    key = json.load(open(keyfile, encoding="utf-8"))
    print(f"\n  session {stamp}")
    for label in ("A", "B"):
        print(f"    {label} = {key['assignment'][label]}")
    print()

    sheet = os.path.join(outdir, "scores.md")
    if os.path.exists(sheet):
        text = open(sheet, encoding="utf-8").read()
        if "## Key (revealed)" in text:
            print("  (already appended to scores.md)")
            return
        with open(sheet, "a", encoding="utf-8") as fh:
            fh.write("\n## Key (revealed)\n\n")
            for label in ("A", "B"):
                fh.write(f"- **{label}** = `{key['assignment'][label]}`\n")
            fh.write(f"\nRevealed {datetime.datetime.now().isoformat(timespec='seconds')}.\n")
        print(f"  appended to {os.path.relpath(sheet, ROOT)}")


# ──────────────────────────────────────────────────────────────── main ──────
def main():
    ap = argparse.ArgumentParser(description="Blind A/B chat sessions for naturalness judging.")
    ap.add_argument("--compare", nargs=2, metavar=("CONFIG1", "CONFIG2"),
                    help="two config specs, e.g. core:current core:lean")
    ap.add_argument("--reveal", metavar="TIMESTAMP",
                    help="print which config was A and which was B, and append it to scores.md")
    ap.add_argument("--memory", action="store_true",
                    help="enable the memory layer in both halves (off by default)")
    ap.add_argument("--seed", type=int, help="fix the A/B coin flip (testing only)")
    args = ap.parse_args()

    if args.reveal:
        reveal(args.reveal); return
    if not args.compare:
        ap.error("one of --compare or --reveal is required")

    topics, sheet = load_protocol()
    cfgs = [parse_spec(s) for s in args.compare]
    if cfgs[0]["spec"] == cfgs[1]["spec"]:
        sys.exit("the two configurations are identical — nothing to compare")

    rng = random.Random(args.seed)
    order = [0, 1]
    rng.shuffle(order)                       # which config gets label A
    assignment = {"A": cfgs[order[0]]["spec"], "B": cfgs[order[1]]["spec"]}

    stamp  = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")
    outdir = os.path.join(OUT_ROOT, stamp)
    os.makedirs(outdir, exist_ok=True)
    json.dump({"created": stamp, "assignment": assignment,
               "memory": bool(args.memory)},
              open(os.path.join(outdir, "key.json"), "w", encoding="utf-8"), indent=2)

    print(f"\n  comparing {cfgs[0]['spec']} vs {cfgs[1]['spec']} — assignment hidden")
    print(f"  logging to eval_human/{stamp}/")

    import camus_rag as cr
    baseline = dict(CORE=cr.CORE, TEMP_FACTUAL=cr.TEMP_FACTUAL, GEN_MODEL=cr.GEN_MODEL)

    # Loaded once and shared: the KB, BM25 index and cross-encoder are identical for both
    # halves, and re-loading the reranker between them would add a slow pause mid-session.
    facts, vecs = cr.load_kb()
    res = (facts, vecs, cr.build_bm25(facts), cr.load_reranker())

    for idx, label in enumerate(("A", "B"), 1):
        for k, v in baseline.items():        # reset, so config 2 never inherits config 1
            setattr(cr, k, v)
        apply_config(cr, parse_spec(assignment[label]))
        run_session(cr, label, idx, res, topics, outdir, args.memory)

    # The protocol says to fill the sheet FOR EACH configuration, then reveal — so emit it
    # twice, once per label, rather than a single side-by-side grid.
    sheet_path = os.path.join(outdir, "scores.md")
    with open(sheet_path, "w", encoding="utf-8") as fh:
        fh.write(f"# Score sheet — session {stamp}\n\n"
                 f"Score **both** halves, then reveal:\n\n"
                 f"    python rag/ab_session.py --reveal {stamp}\n\n")
        for label in ("A", "B"):
            fh.write(f"## Half {label}\n\n{sheet}\n\n")

    print(f"\n  both halves done.")
    print(f"  score sheet: eval_human/{stamp}/scores.md")
    print(f"  reveal with: python rag/ab_session.py --reveal {stamp}\n")


if __name__ == "__main__":
    main()
