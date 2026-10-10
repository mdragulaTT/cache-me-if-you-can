# Notebooks

Five notebooks, one per act. When VS Code asks for a kernel, choose **Select Kernel → Jupyter Kernel → MLinPL 2026** (the Python that has TT-NN; `tools/setup_instance.sh` registers it). Each notebook opens the card itself, so any of them can be started fresh. If a notebook hangs, restart its kernel and run it from the top.

**One rule:** the card can serve only one process at a time, and it stays with that process until the process ends. Each notebook runs as its own process (its kernel), so **the last cell of every act stops that notebook's kernel** and frees the card for the next act. Your outputs stay on the page. If an act says another notebook still holds the card, go to that notebook and click Restart.

| | Notebook | You fill in |
|--|--|--|
| Act 0 | `00_hello.ipynb` | nothing: roll call and the background warm-up |
| Act I | `01_two_acts.ipynb` | the measurement, the speed-limit formula |
| Act II | `02_the_cache.ipynb` | attention, the KV cache, the cache-size formula |
| Act III | `03_the_levers.ipynb` | the batch-size loop, users within the budget |
| Act IV | `04_the_sting.ipynb` | a faster GPT-2: the open-ended race |

A gap looks like this and fails loudly until you fill it in:

```python
# >>> YOUR CODE HERE: follow the formula above step by step; you need ttnn.matmul, ttnn.permute (to transpose k), ttnn.add and ttnn.softmax
raise NotImplementedError("...")
# <<<
```

To see how a TT-NN function is called, run `help(ttnn.permute)` (any function) in a cell, or open the [TT-NN API reference](https://docs.tenstorrent.com/tt-metal/latest/ttnn/ttnn/api.html).

`solutions/` has every gap filled for Acts 0 to III. If you are stuck for more than a few minutes, look: the measurements are the point, not the typing. Unfamiliar word? See [../GLOSSARY.md](../GLOSSARY.md).
