#!/usr/bin/env python3
"""
ingest_sources.py — PDFs -> cleaned, source-tagged chunks for the two-stream KB.

Reads sources_manifest.json, extracts each PDF with PyMuPDF, cleans it (de-hyphenate
across line breaks, strip running headers/footers and bare page numbers, reflow soft
wraps, repair encoding), chunks it, and writes ONE file:

    ./data/source_chunks.jsonl   rows: {id, source, stream, date, text}

stream='fact'  chunks -> later mined for atomic biographical FACTS (biographies/criticism)
stream='voice' chunks -> later mined for Camus's VIEWS in his own words (primary sources)

SETUP:  pip install pymupdf ftfy
RUN:    python ingest_sources.py            (PDFs in ./sources/, names per manifest)
        python ingest_sources.py --src ./sources --fact-words 220 --voice-words 320
"""
import argparse, json, os, re, glob, hashlib
try:
    import fitz  # PyMuPDF
except ImportError:
    fitz = None
try:
    import ftfy
except ImportError:
    ftfy = None

def slug(s):
    return re.sub(r"[^a-z0-9]+","_", s.lower()).strip("_")[:40]

def extract_pages(path):
    doc = fitz.open(path)
    return [doc[i].get_text("text") for i in range(len(doc))]

def find_running_lines(pages):
    """Lines that repeat at top/bottom of many pages = headers/footers."""
    from collections import Counter
    top, bot = Counter(), Counter()
    for p in pages:
        lines = [l.strip() for l in p.splitlines() if l.strip()]
        if not lines: continue
        top[re.sub(r"\d+","#",lines[0])] += 1
        bot[re.sub(r"\d+","#",lines[-1])] += 1
    n = max(len(pages), 1)
    drop = set()
    for c in (top, bot):
        for line, ct in c.items():
            if ct >= 4 and ct / n > 0.25 and len(line) < 80:   # repeats on many pages
                drop.add(line)
    return drop

def clean_page(text, drop):
    out = []
    for line in text.splitlines():
        s = line.strip()
        if not s: 
            out.append("")  # keep paragraph breaks
            continue
        if re.sub(r"\d+","#",s) in drop: continue        # running header/footer
        if re.fullmatch(r"[\divxlcdm]+", s, re.I): continue   # bare page number / roman numeral
        if re.fullmatch(r"\W{0,3}\d{1,4}\W{0,3}", s): continue
        out.append(line)
    return "\n".join(out)

def normalize(text):
    if ftfy: text = ftfy.fix_text(text)
    text = text.replace("\r","\n")
    # de-hyphenate across line/page breaks: "pover-\nty" or "pover-\n\nty" -> "poverty"
    # (lowercase continuation = soft split; rejoin without the hyphen)
    text = re.sub(r"(\w+)-\s*\n+\s*([a-z]\w*)", r"\1\2", text)
    # paragraph break = blank line; single newline = soft wrap -> space
    text = re.sub(r"\n[ \t]*\n", "\n\n", text)
    text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()

def chunk(text, target_words, max_words):
    """Paragraph-aware: pack whole paragraphs up to target; never split mid-sentence
    unless a single paragraph exceeds max_words."""
    paras = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks, buf, n = [], [], 0
    def flush():
        nonlocal buf, n
        if buf: chunks.append(" ".join(buf)); buf, n = [], 0
    for p in paras:
        w = len(p.split())
        if w > max_words:                       # giant paragraph -> sentence split
            flush()
            sents = re.split(r"(?<=[.!?])\s+", p)
            sb, sn = [], 0
            for s in sents:
                sb.append(s); sn += len(s.split())
                if sn >= target_words: chunks.append(" ".join(sb)); sb, sn = [], 0
            if sb: chunks.append(" ".join(sb))
        elif n + w > target_words:
            flush(); buf, n = [p], w
        else:
            buf.append(p); n += w
    flush()
    return [c for c in chunks if len(c.split()) >= 25]   # drop scraps

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="sources_manifest.json")
    ap.add_argument("--src", default="./sources")
    ap.add_argument("--out", default="./data/source_chunks.jsonl")
    ap.add_argument("--fact-words", type=int, default=220)   # tight -> focused fact extraction
    ap.add_argument("--voice-words", type=int, default=320)  # roomier -> preserve view context
    args = ap.parse_args()
    if fitz is None:
        raise SystemExit("PyMuPDF missing: pip install pymupdf ftfy")
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    man = json.load(open(args.manifest, encoding="utf-8"))["sources"]
    total = 0
    with open(args.out, "w", encoding="utf-8") as out:
        for s in man:
            path = os.path.join(args.src, s["file"])
            if not os.path.exists(path):
                print(f"  !! missing: {path}  (skipping)"); continue
            pages = extract_pages(path)
            drop = find_running_lines(pages)
            body = normalize("\n\n".join(clean_page(p, drop) for p in pages))
            tw = args.fact_words if s["stream"]=="fact" else args.voice_words
            chunks = chunk(body, tw, tw*2)
            sl = slug(s["source"])
            for i, c in enumerate(chunks):
                out.write(json.dumps({
                    "id": f"{sl}-{i:05d}", "source": s["source"],
                    "stream": s["stream"], "date": s.get("date",""), "text": c
                }, ensure_ascii=False) + "\n")
            total += len(chunks)
            print(f"  OK {s['file']:32s} {s['stream']:5s} -> {len(chunks)} chunks  (dropped {len(drop)} header/footer patterns)")
    print(f"\n✅ {total} chunks -> {args.out}")

if __name__ == "__main__":
    main()
