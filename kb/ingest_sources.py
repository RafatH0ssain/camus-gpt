#!/usr/bin/env python3
"""
ingest_sources.py — PDFs -> cleaned, source-tagged chunks for the two-stream KB.

Reads sources_manifest.json, extracts each PDF with PyMuPDF, cleans it (de-hyphenate
across line breaks, strip running headers/footers and bare page numbers, reflow soft
wraps, repair encoding), chunks it, and writes ONE file:

    ./data/source_chunks.jsonl   rows: {id, source, stream, date, text}

With --entries, files whose name contains 'notebooks' are cut into notebook
entries instead: rows {id, source, stream, date, text, entry}. With --only, every
row also carries its manifest 'file'.

stream='fact'  chunks -> later mined for atomic biographical FACTS (biographies/criticism)
stream='voice' chunks -> later mined for Camus's VIEWS in his own words (primary sources)

SETUP:  pip install pymupdf ftfy
RUN:    python ingest_sources.py            (PDFs in ./sources/, names per manifest)
        python ingest_sources.py --src ./sources --fact-words 220 --voice-words 320
        python ingest_sources.py --dir ../sources --dir ../raw_books   (books read in place)
        python ingest_sources.py --only 1935_1942_notebooks.pdf --entries --out data/primary_chunks.jsonl
"""
import argparse, json, os, re, glob, hashlib, sys
from collections import Counter
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

def extract_lines(path):
    """[(x0, text)] per page: the same text extract_pages() returns, plus the
    left offset of every line, so a notebook entry's indent survives."""
    doc = fitz.open(path)
    out = []
    for i in range(len(doc)):
        page = []
        for block in doc[i].get_text("dict")["blocks"]:
            if block.get("type") != 0: continue
            for line in block["lines"]:
                page.append((line["bbox"][0], "".join(s["text"] for s in line["spans"])))
        out.append(page)
    return out

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

def clean_lines(lines, drop):
    """clean_page's rules over a line list, so the left offsets stay attached."""
    out = []
    for x0, line in lines:
        s = line.strip()
        if not s:
            out.append((x0, ""))  # keep paragraph breaks
            continue
        if re.sub(r"\d+","#",s) in drop: continue        # running header/footer
        if re.fullmatch(r"[\divxlcdm]+", s, re.I): continue   # bare page number / roman numeral
        if re.fullmatch(r"\W{0,3}\d{1,4}\W{0,3}", s): continue
        out.append((x0, line))
    return out

def clean_page(text, drop):
    return "\n".join(t for _, t in clean_lines([(0, l) for l in text.split("\n")], drop))

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

# ── notebook entries ───────────────────────────────────────────────────────────
# A notebook volume is thousands of dated, self-contained entries. chunk() melts
# them into 320-word blocks, which is right for a book and wrong for a notebook,
# so with --entries those files are cut into entries instead: one row per entry,
# never merged, never split here. See split_entries() for the boundary rules.

PAGE_BREAK = "\f"                      # marks the end of a page in the raw text
ENTRY_HINT = "notebooks"               # --entries applies to these files
INDENT_MIN, INDENT_MAX = 7.0, 23.0     # entry indent from the body margin, in points
SEP_CHARS = r"*—–=\u2026~\u2022_\-"
SEPARATOR = re.compile(rf"^[{SEP_CHARS}]{{1,12}}$")
# A rule that the extraction glued to the text it divides:
# "————A test of daily concentration..." is still a boundary, not an opener.
# Two or more rule chars, so the single dash that opens a real sentence
# ("—Goes to the balcony and pours his whole being") is left alone.
GLUED_RULE = re.compile(rf"^([{SEP_CHARS}]{{2,12}})(?=[^{SEP_CHARS}])")
# An editor's note opens with its number and a capital: "11 In \"The Nearby Sea...",
# "35 Almost all the elements of this entry...". The capital after the number is
# what keeps prose out: "10 years ago", "3 o'clock" and ". . . Run away." do not
# match, so Camus's own sentences and ellipses stay inside their entry.
NOTE_START = re.compile(r"^\d{1,3}\s+[\"'\u201c\u2018A-Z(\u00c0-\u00dd]")
SENTENCE_END = re.compile(r"[.!?\u2026][\"'\u2019\u201d\)\]]*$")
SENTENCE_START = re.compile(r"[\"'“‘(\[]*[A-Z0-9\u00c0-\u00ff]")
MONTH = (r"jan|feb|mar|apr|may|jun|jul|aug|sept?|oct|nov|dec|"
         r"janvier|fevrier|février|mars|avril|juin|juillet|aout|août|"
         r"septembre|octobre|novembre|decembre|décembre")
MONTH_TOKEN = re.compile(rf"^(?:{MONTH})[a-z]*\.?$", re.I)
YEAR_TOKEN = re.compile(r"^\d{3,4}$")
NUM_TOKEN = re.compile(r"^\d{1,2}$")
DATE_WORDS = ("spring", "summer", "autumn", "winter", "printemps", "été", "ete")

def is_blank(line):
    return not line.strip()

def is_separator(line):
    return bool(SEPARATOR.match(line.strip()))

def is_note_start(line):
    """A line that opens an editor's footnote: '11 In "The Nearby Sea...'."""
    return NOTE_START.match(line.strip()) is not None

def is_date_heading(line):
    """A line that is only a date, a month/year, or a span of them:
    'APRIL', 'DECEMBER 15', 'April 1948', 'September 1937 - April 1939'."""
    s = line.strip().strip(" .\t")
    if not s or len(s) > 48: return False
    toks = [t for t in re.split(r"[\s,./\-–—]+", s) if t]
    if not any(MONTH_TOKEN.match(t) or t.lower() in DATE_WORDS for t in toks):
        return False
    for t in toks:
        if (MONTH_TOKEN.match(t) or YEAR_TOKEN.match(t) or NUM_TOKEN.match(t)
                or t.lower() in DATE_WORDS):
            continue
        return False
    return True

def split_entries(text):
    """Split raw extracted text into notebook entries, before any reflow.

    An entry ends at: a blank line; a date or month/year heading line (which
    then heads the next entry); a separator line (*, ***, ---); a footnote
    opening (which starts the next unit); or a page break that lands on a
    paragraph end. A page break mid-paragraph is not a boundary, so an entry
    continues onto the next page. Pages are separated by \\f.
    """
    entries, buf, heading_only = [], [], True

    def flush():
        nonlocal buf, heading_only
        if buf: entries.append("\n".join(buf).strip())
        buf, heading_only = [], True

    for page in text.split(PAGE_BREAK):
        lines = page.split("\n")
        # page break: a boundary only if the page ended on a paragraph end
        if buf and lines and not is_blank(lines[0]) and not is_blank(buf[-1]) \
                and SENTENCE_END.search(buf[-1].strip()) \
                and SENTENCE_START.match(lines[0].strip()):
            flush()
        for line in lines:
            glued = GLUED_RULE.match(line.strip())
            if glued:                         # the rule closes the entry above
                flush()
                line = line.strip()[glued.end():]
            if is_blank(line) or is_separator(line):
                flush(); continue
            if is_date_heading(line):
                if not heading_only: flush()      # the heading heads what follows
                buf.append(line); heading_only = True; continue
            # The combined volume sets the editor's notes in the text flow with no
            # blank line between them, so without this a run of notes fuses into a
            # single entry. The number opens the note, never the entry above it.
            if buf and is_note_start(line): flush()
            buf.append(line); heading_only = False
    flush()
    return [e for e in entries if e]

def body_margin(lines):
    """The page's body-text left offset = the most common one among its long
    lines; running heads, page numbers and indents are all outliers."""
    counts = Counter(round(x) for x, t in lines if len(t.strip()) >= 12)
    if counts: return counts.most_common(1)[0][0]
    return min((x for x, _ in lines), default=0.0)

def entry_start_indices(lines):
    """Indexes of the lines that begin an entry. Every notebook edition sets each
    entry a step in from the body margin, so the indent is the boundary that the
    plain text stream does not carry."""
    base = body_margin(lines)
    return {i for i, (x, t) in enumerate(lines)
            if t.strip() and INDENT_MIN <= x - base <= INDENT_MAX}

def entry_page_text(lines):
    """Raw page text with a blank line inserted before every entry start, so
    that the plain-text splitter sees the edition's entries as paragraphs."""
    starts = entry_start_indices(lines)
    out = []
    for i, (_, t) in enumerate(lines):
        if i in starts and (out or i):
            out.append("")
        out.append(t)
    return "\n".join(out)

def notebook_entries(path):
    """(raw entries, number of running header/footer patterns dropped)."""
    pages = extract_lines(path)
    drop = find_running_lines(["\n".join(t for _, t in p) for p in pages])
    raw = PAGE_BREAK.join(entry_page_text(clean_lines(p, drop)) for p in pages)
    return split_entries(raw), len(drop)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="sources_manifest.json")
    ap.add_argument("--src", default="./sources")
    ap.add_argument("--dir", action="append", dest="dirs", default=None, metavar="D",
                    help="extra directory to search for manifest files (repeatable); "
                         "default: --src only. First hit wins.")
    ap.add_argument("--out", default="./data/source_chunks.jsonl")
    ap.add_argument("--only", nargs="+", default=None, metavar="FILE",
                    help="ingest only these manifest files (repeatable names, no "
                         "paths) and tag every row with its 'file'; a name that is "
                         "not in the manifest is an error")
    ap.add_argument("--entries", action="store_true",
                    help=f"for files whose name contains '{ENTRY_HINT}', cut the "
                         "raw text into notebook entries (one row per entry) "
                         "instead of chunking; other files are unaffected")
    ap.add_argument("--fact-words", type=int, default=220)   # tight -> focused fact extraction
    ap.add_argument("--voice-words", type=int, default=320)  # roomier -> preserve view context
    args = ap.parse_args()
    if fitz is None:
        raise SystemExit("PyMuPDF missing: pip install pymupdf ftfy")
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    man = json.load(open(args.manifest, encoding="utf-8"))["sources"]
    if args.only:
        wanted = set(args.only)
        unknown = sorted(wanted - {s["file"] for s in man})
        if unknown:                       # a typo'd --only silently drops sources
            for name in unknown:
                print(f"  !! --only {name}: not in the manifest", file=sys.stderr)
            raise SystemExit(
                f"error: {len(unknown)} --only file(s) not in the manifest: "
                + ", ".join(unknown))
        man = [s for s in man if s["file"] in wanted]
    dirs = args.dirs or [args.src]
    total, kind = 0, "chunks"
    with open(args.out, "w", encoding="utf-8") as out:
        for s in man:
            path = next((os.path.join(d, s["file"]) for d in dirs
                         if os.path.exists(os.path.join(d, s["file"]))), None)
            if path is None:
                print(f"  !! missing: {s['file']} in {dirs}  (skipping)"); continue
            entry_mode = args.entries and ENTRY_HINT in s["file"].lower()
            if entry_mode:
                found, n_drop = notebook_entries(path)
                units = [normalize(e) for e in found]
                units = [u for u in units if u]
                label = "entries"
            else:
                pages = extract_pages(path)
                drop = find_running_lines(pages)
                body = normalize("\n\n".join(clean_page(p, drop) for p in pages))
                tw = args.fact_words if s["stream"]=="fact" else args.voice_words
                units, n_drop = chunk(body, tw, tw*2), len(drop)
                label = "chunks"
            if entry_mode: kind = "entries"
            sl = slug(s["source"])
            for i, c in enumerate(units):
                row = {
                    "id": f"{sl}-{i:05d}", "source": s["source"],
                    "stream": s["stream"], "date": s.get("date",""), "text": c
                }
                if entry_mode: row["entry"] = True
                if args.only:   row["file"] = s["file"]
                out.write(json.dumps(row, ensure_ascii=False) + "\n")
            total += len(units)
            print(f"  OK {s['file']:32s} {s['stream']:5s} -> {len(units)} {label}  (dropped {n_drop} header/footer patterns)")
    print(f"\n✅ {total} {kind} -> {args.out}")

if __name__ == "__main__":
    main()
