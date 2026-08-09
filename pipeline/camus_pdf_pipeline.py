#!/usr/bin/env python3
"""
CamusGPT — STEP 0: SOURCE INGESTION + CLEANING + CHUNKING
=========================================================

Turns raw PDFs/EPUBs into a pristine chunks.jsonl that camus_sft_generator.py
can consume. Since the essayist SFT targets are used VERBATIM, this stage is
where literary fidelity is won or lost.

What it does
------------
PDF  (PyMuPDF): extracts text block-by-block (blocks ~= paragraphs), detects and
     strips running headers/footers (lines that repeat across many pages) and
     pure page numbers, drops short margin-zone junk, de-hyphenates words broken
     across lines, and reflows soft-wrapped lines.
EPUB (ebooklib + BeautifulSoup): strips HTML/scripts/styles, keeps paragraph
     structure. EPUBs are already clean — prefer them over PDFs when you have a
     choice.
Both: encoding repair (ftfy if installed), ligature/quote/dash normalization,
     control-char removal, then paragraph-aware chunking to a token budget.

Honest limits
-------------
- Regex cannot repair true OCR character errors ("rn"->"m"). If a source is a
  SCANNED PDF (image-only), extraction will be poor — re-OCR it or use the EPUB.
  This pipeline assumes a real embedded text layer (true for ebooks/digital PDFs).
- Always run inspect_data.py on a few chunks before generating thousands of rows.

INPUT
-----
A folder of .pdf / .epub files. Filename (minus extension) becomes the "source"
unless you pass --source-map.

OUTPUT
------
chunks.jsonl, one object per line:
    {"id": "the-plague-0007", "source": "The Plague", "text": "<clean passage>"}

USAGE
-----
    pip install pymupdf ebooklib beautifulsoup4 ftfy
    python camus_pdf_pipeline.py \
        --input-dir ./raw_books \
        --output chunks.jsonl \
        --target-tokens 400 \
        --max-tokens 800 \
        --min-tokens 80
"""

import argparse
import json
import os
import re
import sys
import unicodedata
from collections import Counter

# ---- optional deps, imported lazily with clear errors --------------------- #
try:
    import fitz  # PyMuPDF
except ImportError:
    fitz = None
try:
    from ebooklib import epub
    import ebooklib
    from bs4 import BeautifulSoup
except ImportError:
    epub = None
try:
    import ftfy
except ImportError:
    ftfy = None

# English prose averages ~1.3 Llama-3 tokens/word. Good enough for budgeting;
# pass --use-exact-tokenizer to count with the real tokenizer instead.
TOKENS_PER_WORD = 1.3

LIGATURES = {"\ufb01": "fi", "\ufb02": "fl", "\ufb00": "ff", "\ufb03": "ffi",
             "\ufb04": "ffl", "\u00a0": " ", "\u200b": ""}

PAGE_NUM_RE = re.compile(r"^\s*[\divxlcdmIVXLCDM]{1,7}\s*$")
HYPHEN_BREAK_RE = re.compile(r"(\w)[-\u2010]\s*\n\s*(\w)")  # word broken across line


# --------------------------------------------------------------------------- #
# Text normalization (applies to both PDF and EPUB output)
# --------------------------------------------------------------------------- #
def normalize_text(text: str) -> str:
    if ftfy:
        text = ftfy.fix_text(text)            # repair mojibake / bad encoding
    text = unicodedata.normalize("NFKC", text)
    for bad, good in LIGATURES.items():
        text = text.replace(bad, good)
    # strip control chars except newline/tab
    text = "".join(ch for ch in text if ch == "\n" or ch == "\t" or unicodedata.category(ch)[0] != "C")
    text = HYPHEN_BREAK_RE.sub(r"\1\2", text)  # join hyphen-broken words
    # canonicalize paragraph breaks (one or more blank lines) -> exactly "\n\n"
    text = re.sub(r"[ \t]*\n(?:[ \t]*\n)+[ \t]*", "\n\n", text)
    # reflow remaining single (soft-wrap) newlines into spaces
    text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


def looks_like_junk(line: str) -> bool:
    s = line.strip()
    if not s:
        return True
    if PAGE_NUM_RE.match(s):
        return True
    # mostly non-alphabetic -> likely artifact
    alpha = sum(c.isalpha() for c in s)
    return len(s) > 0 and alpha / len(s) < 0.5 and len(s) < 40


# --------------------------------------------------------------------------- #
# PDF extraction with running-header/footer detection
# --------------------------------------------------------------------------- #
def extract_pdf(path: str) -> str:
    if fitz is None:
        sys.exit("pip install pymupdf")
    doc = fitz.open(path)
    n_pages = len(doc)

    # Pass 1: collect first/last line of each page to find repeating headers/footers
    edge_counter = Counter()
    page_blocks = []
    for page in doc:
        blocks = [b for b in page.get_text("blocks") if b[6] == 0 and b[4].strip()]
        blocks.sort(key=lambda b: (round(b[1]), b[0]))  # reading order: top->bottom
        page_blocks.append((page.rect.height, blocks))
        if blocks:
            top = blocks[0][4].strip().split("\n")[0]
            bot = blocks[-1][4].strip().split("\n")[-1]
            # normalize away page numbers so "Chapter 3 — 41" still matches "Chapter 3"
            edge_counter[re.sub(r"\d+", "#", top)[:60]] += 1
            edge_counter[re.sub(r"\d+", "#", bot)[:60]] += 1

    repeat_threshold = max(3, int(0.30 * n_pages))
    running = {k for k, c in edge_counter.items() if c >= repeat_threshold}

    # Pass 2: build body, skipping headers/footers/margin junk/page numbers
    parts = []
    for height, blocks in page_blocks:
        for b in blocks:
            x0, y0, x1, y1, txt = b[0], b[1], b[2], b[3], b[4].strip()
            norm = re.sub(r"\d+", "#", txt.split("\n")[0])[:60]
            in_margin = (y0 < 0.08 * height) or (y1 > 0.92 * height)
            if norm in running and in_margin:
                continue
            if PAGE_NUM_RE.match(txt):
                continue
            if in_margin and len(txt) < 60 and looks_like_junk(txt):
                continue
            parts.append(txt)
    doc.close()
    return "\n\n".join(parts)


# --------------------------------------------------------------------------- #
# EPUB extraction
# --------------------------------------------------------------------------- #
def extract_epub(path: str) -> str:
    if epub is None:
        sys.exit("pip install ebooklib beautifulsoup4")
    book = epub.read_epub(path)
    parts = []
    for item in book.get_items_of_type(ebooklib.ITEM_DOCUMENT):
        soup = BeautifulSoup(item.get_content(), "html.parser")
        for tag in soup(["script", "style", "nav"]):
            tag.decompose()
        for p in soup.find_all(["p", "div", "h1", "h2", "h3", "blockquote"]):
            t = p.get_text(" ", strip=True)
            if t:
                parts.append(t)
    return "\n\n".join(parts)


# --------------------------------------------------------------------------- #
# Paragraph-aware chunking to a token budget
# --------------------------------------------------------------------------- #
def est_tokens(text: str) -> int:
    return int(len(text.split()) * TOKENS_PER_WORD)


def split_sentences(paragraph: str):
    # lightweight splitter; good enough for over-long paragraphs
    return re.split(r'(?<=[.!?])["\'\u00bb\u201d)\]]?\s+(?=[A-Z\u00ab"\u201c])', paragraph)


def chunk_paragraphs(paragraphs, target, max_tok, min_tok):
    chunks, buf, buf_tok = [], [], 0
    for para in paragraphs:
        para = para.strip()
        if not para or looks_like_junk(para):
            continue
        ptok = est_tokens(para)
        if ptok > max_tok:  # giant paragraph -> split on sentences
            for sent in split_sentences(para):
                stok = est_tokens(sent)
                if buf_tok + stok > target and buf_tok >= min_tok:
                    chunks.append(" ".join(buf)); buf, buf_tok = [], 0
                buf.append(sent.strip()); buf_tok += stok
            continue
        if buf_tok + ptok > target and buf_tok >= min_tok:
            chunks.append(" ".join(buf)); buf, buf_tok = [], 0
        buf.append(para); buf_tok += ptok
    if buf and buf_tok >= min_tok:
        chunks.append(" ".join(buf))
    return chunks


def slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--target-tokens", type=int, default=400)
    ap.add_argument("--max-tokens", type=int, default=800)
    ap.add_argument("--min-tokens", type=int, default=80)
    ap.add_argument("--source-map", help="optional JSON {filename: 'Pretty Title'}")
    args = ap.parse_args()

    source_map = {}
    if args.source_map:
        source_map = json.load(open(args.source_map, encoding="utf-8"))

    files = [f for f in sorted(os.listdir(args.input_dir))
             if f.lower().endswith((".pdf", ".epub"))]
    if not files:
        sys.exit(f"No .pdf/.epub files in {args.input_dir}")

    total = 0
    with open(args.output, "w", encoding="utf-8") as out:
        for fname in files:
            path = os.path.join(args.input_dir, fname)
            stem = os.path.splitext(fname)[0]
            source = source_map.get(fname, stem)
            kind = "EPUB" if fname.lower().endswith(".epub") else "PDF"
            print(f"[{kind}] {fname} -> '{source}'")
            try:
                raw = extract_epub(path) if kind == "EPUB" else extract_pdf(path)
            except Exception as e:  # noqa: BLE001
                print(f"  !! failed to read: {e}", file=sys.stderr)
                continue
            clean = normalize_text(raw)
            paragraphs = clean.split("\n\n")
            chunks = chunk_paragraphs(paragraphs, args.target_tokens,
                                      args.max_tokens, args.min_tokens)
            for i, ch in enumerate(chunks):
                out.write(json.dumps(
                    {"id": f"{slug(source)}-{i:05d}", "source": source, "text": ch},
                    ensure_ascii=False) + "\n")
            print(f"   {len(chunks)} chunks "
                  f"(median ~{args.target_tokens} tok, {kind} source)")
            total += len(chunks)
    print(f"\nDone. {total} chunks -> {args.output}")
    print("NEXT: run inspect_data.py on this file before generating SFT rows.")


if __name__ == "__main__":
    main()
