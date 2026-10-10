# 🤘 Cache Me If You Can

A hands-on tutorial on LLM inference, given at [ML in PL 2026](https://conference.mlinpl.org/2026/program#tutorial-9) (Sunday, 11 October, Warsaw) by Maciek Draguła and Nathan Maurice from Tenstorrent.

A language model at work is a black box: a prompt goes in, tokens stream out, and everything in between is someone else's problem. In this tutorial it becomes yours. Through experiments in Python on Tenstorrent AI accelerators, you look at the prefill and decode phases, build a KV cache yourself, try batching and smaller number formats, and finish with a speedrun: make a model as fast as you can.

## At the workshop

Each participant gets their own machine in the cloud: an instance on Tenstorrent Console with one n150 card, opened as VS Code in the browser. Everything is already installed on it.

1. Open your instance with the URL on your paper slip and log in with the login and password printed there.
2. Open `notebooks/00_hello.ipynb`, choose the kernel **MLinPL 2026**, and run it. Arrived late? Start with Act 0 anyway: it starts a warm-up that the later acts need.
3. How the notebooks work: [notebooks/README.md](notebooks/README.md).

## Program

| Time | | |
|------|--|--|
| 9:00 | Act 0 | `00_hello.ipynb`: your card says hello; the warm-up starts in the background |
| 9:20 | Talk 1 | Two phases of a language model |
| 9:40 | Act I | `01_two_acts.ipynb`: time to first token, tokens per second, and the speed limit memory bandwidth sets |
| 10:10 | | Break |
| 10:20 | Talk 2 | The chip you are driving |
| 10:30 | Act II | `02_the_cache.ipynb`: build a KV cache into a real GPT-2, in TT-NN |
| 11:10 | | Break |
| 11:20 | Act III | `03_the_levers.ipynb`: batch size and number formats |
| 11:45 | Act IV | `04_the_sting.ipynb`: the Speedrun, make GPT-2 as fast as you can |
| 12:35 | Talk 3 | What changes at scale, and the results |

Acts I and III serve DeepSeek-R1-Distill-Qwen-1.5B with tt_transformers, the way you would in production. Acts II and IV use GPT-2 small, written in about a hundred lines of TT-NN, so that you can read and change every line.

## What is here

```
notebooks/             the five notebooks, with gaps for you to fill
notebooks/solutions/   the answers for Acts 0 to III (Act IV is a race; its solution comes after the workshop)
slides/                the three talks (Markdown, PowerPoint and HTML)
common/                the shared code: the two models, the Speedrun race, the warm-up
tools/                 setup_instance.sh and helpers for preparing a machine
GLOSSARY.md            every term, defined
```

## Running it on your own

The notebooks also work without the event, on any machine with an n150 card and TT-NN installed (tested with TT-NN 0.77):

```bash
git clone https://github.com/mdragulaTT/cache-me-if-you-can.git && cd cache-me-if-you-can
bash tools/setup_instance.sh
```

The script checks the machine, installs missing Python packages, downloads about 4 GB of model weights, registers the "MLinPL 2026" notebook kernel and runs a short test. Then open the notebooks in `notebooks/` in order; the talks in `slides/` go between them (Talk 1 after Act 0, Talk 2 after Act I, Talk 3 at the end).

## Built with

[tt-metal and TT-NN](https://github.com/tenstorrent/tt-metal) (including tt_transformers), [GPT-2](https://huggingface.co/openai-community/gpt2) and [DeepSeek-R1-Distill-Qwen-1.5B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B) from Hugging Face. TT-NN documentation: [docs.tenstorrent.com](https://docs.tenstorrent.com/tt-metal/latest/ttnn/ttnn/api.html).

## License

Apache-2.0, like tt-metal: see [LICENSE](LICENSE). © 2026 Tenstorrent AI ULC.
