#!/usr/bin/env python3
"""
merge_dpo.py — assemble the final DPO training file.

Combines style + adversarial pairs into one shuffled file, capping adversarial
at FINAL_ADV_FRACTION so guardrails sharpen behavior without dominating it (and
turning the model into a refusal machine). Prints a composition + quality report.

    python merge_dpo.py

Output: ./data/camus_dpo_final.jsonl  (this is what the DPO trainer will read)
"""
import json
import re
import random
from collections import Counter
import config as C

DPO_FINAL = f"{C.DATA_DIR}/camus_dpo_final.jsonl"
FINAL_ADV_FRACTION = 0.08

AI  = re.compile(r"\b(as an ai|i am an ai|language model|i'm an ai|openai|anthropic|chatbot|as a large)\b", re.I)
LIST = re.compile(r"(^|\n)\s*(\d+[.)]\s|[-*\u2022]\s)")


def load(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def main():
    style = load(C.STYLE_OUT)
    adv = load(C.ADV_OUT)
    random.seed(42)
    random.shuffle(style); random.shuffle(adv)

    f = FINAL_ADV_FRACTION
    cap = int(len(style) * f / (1 - f)) if f < 1 else len(adv)
    adv_keep = adv[:cap]
    final = style + adv_keep
    random.shuffle(final)
    C.write_jsonl(DPO_FINAL, final)

    # ── report ──────────────────────────────────────────────────────────────
    ratio = len(adv_keep) / max(1, len(final)) * 100
    same = sum(1 for r in final if r["chosen"].strip() == r["rejected"].strip())
    cf = sum(1 for r in final if AI.search(r["chosen"]))      # AI-tells in chosen (want ~0)
    cl = sum(1 for r in final if LIST.search(r["chosen"]))
    fam = Counter("style" if r["category"] == "style" else "adversarial" for r in final)

    print(f"final DPO: {len(final)} rows -> {DPO_FINAL}")
    print(f"  composition: style={fam['style']}  adversarial={fam['adversarial']} ({ratio:.1f}%)")
    if len(adv) > cap:
        print(f"  held back {len(adv) - cap} adversarial pairs to keep the {f*100:.0f}% cap")
    print(f"  chosen==rejected (must be 0): {same}")
    print(f"  AI-tells in chosen (want ~0): {cf}   lists in chosen: {cl}")
    print("  next: this file feeds Stage-2 DPO training.")


if __name__ == "__main__":
    main()
