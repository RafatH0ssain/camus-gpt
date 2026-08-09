import json, re
ai = re.compile(r"\b(as an ai|i am an ai|language model|i'm an ai|openai|anthropic|chatbot|as a large)\b", re.I)
for r in (json.loads(l) for l in open("./data/camus_dpo_adversarial.jsonl", encoding="utf-8")):
    if ai.search(r["chosen"]): print("\n---", r["category"], "\n", r["chosen"])