#!/usr/bin/env python3
"""Draw the Act II diagrams (PNG) into notebooks/media/. Re-run after editing; the notebook
builder embeds these files into the notebooks as attachments."""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

OUT = Path(__file__).resolve().parents[1] / "notebooks" / "media"
OUT.mkdir(parents=True, exist_ok=True)
NEW, OLD, SKIP, CACHE = "#5B3FD6", "#C9C2F2", "#EEEEF3", "#B8731A"
plt.rcParams.update({"font.size": 11, "font.family": "DejaVu Sans"})


def grid(ax, n, fill, title, row_labels=True):
    """Causal attention matrix for n tokens; fill(i, j) -> colour or None (masked)."""
    for i in range(n):
        for j in range(n):
            c = fill(i, j)
            ax.add_patch(Rectangle((j, n - 1 - i), 1, 1, facecolor=c or "white",
                                   edgecolor="#9a98ad" if c else "#e3e2ea", lw=0.8, hatch=None if c else "////"))
    ax.set_xlim(0, n); ax.set_ylim(0, n); ax.set_aspect("equal")
    ax.set_xticks([j + 0.5 for j in range(n)], [f"k{j+1}" for j in range(n)])
    ax.set_yticks([n - 0.5 - i for i in range(n)], [f"q{i+1}" for i in range(n)] if row_labels else [""] * n)
    ax.tick_params(length=0); ax.set_title(title, fontsize=12, pad=8)
    for s in ax.spines.values(): s.set_visible(False)


# 1. the causal attention matrix, with the newest row highlighted
n = 6
fig, ax = plt.subplots(figsize=(5.2, 5.0))
grid(ax, n, lambda i, j: None if j > i else (NEW if i == n - 1 else OLD),
     "Attention scores  q_i · k_j  for 6 tokens")
ax.annotate("new token: the only row\nthis step needs", xy=(n, 0.5), xytext=(n + 0.4, 0.5),
            va="center", fontsize=10, color=NEW, annotation_clip=False)
ax.text(n - 1.2, n - 0.9, "masked:\ntokens cannot\nsee the future", ha="center", fontsize=9, color="#77768a")
fig.savefig(OUT / "attention_matrix.png", dpi=150, bbox_inches="tight"); plt.close(fig)

# 2. without vs with a cache: what is computed for the newest token
fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.8))
grid(axes[0], n, lambda i, j: None if j > i else NEW,
     "Without a cache: recompute everything\n(Q, K, V for all 6 tokens, all 21 scores)")
grid(axes[1], n, lambda i, j: None if j > i else (NEW if i == n - 1 else SKIP),
     "With a cache: one row\n(q6 against cached k1..k5, plus the new k6)", row_labels=True)
for j in range(n - 1):
    axes[1].add_patch(Rectangle((j, -0.75), 1, 0.45, facecolor=CACHE, edgecolor="white", clip_on=False))
axes[1].add_patch(Rectangle((n - 1, -0.75), 1, 0.45, facecolor=NEW, edgecolor="white", clip_on=False))
axes[1].text(-0.2, -0.53, "K, V:", ha="right", va="center", fontsize=10)
axes[1].text(n / 2, -1.25, "orange = read from the cache   ·   purple = computed this step", ha="center", fontsize=9.5)
fig.savefig(OUT / "cache_vs_no_cache.png", dpi=150, bbox_inches="tight"); plt.close(fig)

# 3. why Q is not cached: a key/value column is reused by every later row; a query row is used once
fig, axes = plt.subplots(1, 2, figsize=(10.4, 5.0))
grid(axes[0], n, lambda i, j: None if j > i else (CACHE if j == 2 else SKIP),
     "Key k3 (and value v3) are needed by\nrow 3 and by every later row  ->  keep them")
grid(axes[1], n, lambda i, j: None if j > i else (NEW if i == 2 else SKIP),
     "Query q3 is needed only in row 3,\nwhen token 3 is processed  ->  nothing to keep")
fig.savefig(OUT / "why_not_q.png", dpi=150, bbox_inches="tight"); plt.close(fig)
print("diagrams ->", sorted(p.name for p in OUT.glob("*.png")))
