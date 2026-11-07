# CamusGPT

I wanted to talk to Albert Camus, so I built him. CamusGPT is Gemma 4 12B fine-tuned to speak
like Camus: dry, plain, first person, a man of his time who doesn't know what a smartphone is.
It runs locally on Ollama.

The model only learns how he talks and behaves. What he knows about his own life comes from a
knowledge base I built from biographies and his writing, retrieved fresh for every message. So
when he gets a fact wrong, I fix the knowledge base or the retrieval, not the weights.

> A fictional AI persona for education and conversation, not the real Albert Camus.

![Starting CamusGPT in a terminal and asking it four questions, from the absurd to his cat Cigarette; answers stream back in his voice (sped up 2.5x)](docs/camus-demo.gif)

## Running it

You need [Ollama](https://ollama.com) and Python 3.10+. The model weights and the knowledge base
are too big for git, so [docs/WORKSPACE.md](docs/WORKSPACE.md) covers building them and where they
go. Then, from the repo root:

```bash
python rag/camus_rag.py
```

`--debug` shows what retrieval pulled up for each message, and `--memory` turns on a memory
layer so he remembers you between sessions (off by default, see
[docs/MEMORY.md](docs/MEMORY.md)). Long chats don't fall off a cliff either: once a turn scrolls
out of his window it gets folded into a running summary he can still see.

## How it's made

Training is two LoRA passes on Colab. The first teaches the voice, from text mined out of his
books plus a set of short conversational replies. The second teaches behaviour: correcting false
premises about his life, refusing to pretend he wrote things he didn't, staying himself when
someone tries to jailbreak him, and using retrieved facts instead of making them up. A short DPO
pass at the end trains against his own worst answers. The export notebook merges both passes and
quantizes to a q4_k_m GGUF, which takes about 8 GB of memory while it's loaded.

| Folder | What's in it |
|---|---|
| `rag/` | the chat itself: retrieval, the persona prompt, memory, and the eval scripts |
| `kb/` | builds the knowledge base (ingest sources, extract facts, merge, embed, trim) |
| `training/` | the Colab notebooks for both passes and the GGUF export |
| `pipeline/` | assembles the training sets and runs the voice corpus batch jobs |

The training data isn't in the repo. [data/README.md](data/README.md) describes its format, and
[docs/PIPELINE.md](docs/PIPELINE.md) walks through the whole thing end to end.

## Testing it

I score every build instead of eyeballing it. `rag/eval_camus.py` runs a set of probes through
the exact same chat pipeline and has a separate model judge the answers:

```bash
python rag/eval_camus.py --out-dir /tmp/camus_eval
```

`rag/eval_memory.py` does the same for the memory layer.
