"""Sample a served model on fixed contexts: the rejected side of DPO pairs (on-policy data,
Tajwar et al., arXiv:2404.14367). Input lines carry `id` and either `messages` or
`system` + `prompt`; output lines add `reply`. Resumes: ids already in the output are skipped.

  GEN_MODEL=camus4 python pipeline/sample_policy.py IN.jsonl OUT.jsonl
"""
import json
import os
import sys

import requests

OLLAMA = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
MODEL = os.environ.get("GEN_MODEL", "camus4")
OPTS = {"temperature": 0.6, "top_k": 40, "min_p": 0.05, "repeat_penalty": 1.1, "num_ctx": 8192}


def messages_of(item):
    if "messages" in item:
        return item["messages"]
    return [{"role": "system", "content": item["system"]}, {"role": "user", "content": item["prompt"]}]


def main(src, dst):
    done = set()
    if os.path.exists(dst):
        done = {json.loads(l)["id"] for l in open(dst, encoding="utf-8")}
    with open(dst, "a", encoding="utf-8") as out:
        for line in open(src, encoding="utf-8"):
            item = json.loads(line)
            if item["id"] in done:
                continue
            r = requests.post(f"{OLLAMA}/api/chat", timeout=300, json={
                "model": MODEL, "messages": messages_of(item), "stream": False,
                "options": OPTS, "think": False})
            r.raise_for_status()
            out.write(json.dumps({"id": item["id"], "model": MODEL,
                                  "reply": r.json()["message"]["content"].strip()},
                                 ensure_ascii=False) + "\n")
            out.flush()


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
