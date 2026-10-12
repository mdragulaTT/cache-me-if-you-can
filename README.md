# 🤘 Cache Me If You Can

A hands-on tutorial on LLM inference, given at [ML in PL 2026](https://conference.mlinpl.org/2026/program#tutorial-9) (11 October 2026, Warsaw) by Maciek Draguła and Nathan Maurice from Tenstorrent.

A language model at work is a black box: a prompt goes in, tokens stream out, and everything in between is someone else's problem. In this tutorial it becomes yours. Through experiments in Python on a Tenstorrent AI accelerator, you look at the prefill and decode phases, build a KV cache yourself, try batching and smaller number formats, and finish with a speedrun: make a model as fast as you can. Knowing PyTorch is enough; it takes about four hours.

## What you need

- A machine with one **Tenstorrent n150** card, which is what the tutorial was built and tested on.
- **[tt-metal](https://github.com/tenstorrent/tt-metal) built from source**, with `TT_METAL_HOME` pointing to the checkout (tested with TT-NN 0.77). Acts I and III use tt_transformers, which lives in the tt-metal source tree.
- About 5 GB of free disk and an internet connection, to download the model weights once.
- VS Code with the Jupyter extension, or JupyterLab.

## Set up

```bash
git clone https://github.com/mdragulaTT/cache-me-if-you-can.git
cd cache-me-if-you-can
bash tools/setup_instance.sh
```

The script uses tt-metal's Python (`$TT_METAL_HOME/python_env`; pass `PYTHON=/path/to/python` to use another one that can `import ttnn`). It checks the machine and installs missing Python packages. It also downloads GPT-2 and DeepSeek-R1-Distill-Qwen-1.5B (about 4 GB) to `~/mlinpl-weights`, registers a notebook kernel called **MLinPL 2026**, and runs a short test on the card. It is safe to run again.

## Go through it

Open the notebooks with the **MLinPL 2026** kernel, in this order (times as at the workshop):

| | | Time |
|--|--|--|
| Act 0 | `notebooks/00_hello.ipynb`: your card says hello; a warm-up starts in the background | 20 min |
| Talk 1 | `slides/talk1_two_acts.pdf`: two phases of a language model | 20 min |
| Act I | `notebooks/01_two_acts.ipynb`: time to first token, tokens per second, and the speed limit memory bandwidth sets | 30 min |
| Talk 2 | `slides/talk2_the_chip.pdf`: the chip you are driving | 10 min |
| Act II | `notebooks/02_the_cache.ipynb`: build a KV cache into a real GPT-2, in TT-NN | 40 min |
| Act III | `notebooks/03_the_levers.ipynb`: batch size and number formats | 25 min |
| Act IV | `notebooks/04_the_sting.ipynb`: the Speedrun, make GPT-2 as fast as you can | 50 min or more |
| Talk 3 | `slides/talk3_at_scale.pdf`: what changes at scale | 15 min |

Acts I and III serve DeepSeek-R1-Distill-Qwen-1.5B with tt_transformers, the way you would in production. Acts II and IV use GPT-2 small, written in about a hundred lines of TT-NN, so that you can read and change every line. How the notebooks work (the card, the warm-up, the gaps to fill in) is explained in [notebooks/README.md](notebooks/README.md).

## What is here

```
notebooks/             the five notebooks, with gaps for you to fill
notebooks/solutions/   the answers to all five acts, including a reference solution for the Speedrun
slides/                the three talks (PDF)
common/                the shared code: the two models, the Speedrun race, the warm-up
tools/                 setup_instance.sh and helpers for preparing a machine
GLOSSARY.md            every term, defined
```

## Built with

[tt-metal and TT-NN](https://github.com/tenstorrent/tt-metal) (including tt_transformers), [GPT-2](https://huggingface.co/openai-community/gpt2) and [DeepSeek-R1-Distill-Qwen-1.5B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B) from Hugging Face. TT-NN documentation: [docs.tenstorrent.com](https://docs.tenstorrent.com/tt-metal/latest/ttnn/ttnn/api.html).

## License

Apache-2.0, like tt-metal: see [LICENSE](LICENSE). © 2026 Tenstorrent AI ULC.
