#!/usr/bin/env python3
"""Draw the Act II diagrams (PNG) into 2026/notebooks/media/. Re-run after editing; the notebook
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


# ================================================================================================
# Diagrams for the talks (2026/slides/media/)
# ================================================================================================
from matplotlib.patches import FancyBboxPatch

SLIDES = Path(__file__).resolve().parents[1] / "slides" / "media"
SLIDES.mkdir(parents=True, exist_ok=True)
INK, MUTED = "#15151E", "#6b6a80"


def box(ax, x, y, w, h, text="", fc=OLD, ec="#9a98ad", fs=10, color=INK, round_=0.08, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={round_}", fc=fc, ec=ec, lw=1))
    if text:
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=color,
                fontweight="bold" if bold else "normal")


def arrow(ax, x0, y0, x1, y1, color=MUTED):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle="-|>", color=color, lw=1.4))


def blank(ax, w, h):
    ax.set_xlim(0, w); ax.set_ylim(0, h); ax.set_aspect("equal"); ax.axis("off")


# 4. prefill vs decode
fig, ax = plt.subplots(figsize=(11, 4.6)); blank(ax, 22, 9.2)
ax.text(0, 8.6, "Prefill: the whole prompt in one pass", fontsize=13, fontweight="bold", color=INK)
for i, t in enumerate(["The", "chip", "reads", "all", "these", "tokens", "at", "once"]):
    box(ax, 0.3 + i * 1.25, 6.4, 1.1, 0.9, t, fc=OLD, fs=9)
arrow(ax, 10.4, 6.85, 11.6, 6.85)
box(ax, 11.7, 6.1, 3.2, 1.5, "model\n(one pass)", fc=NEW, ec=NEW, color="white", bold=True)
arrow(ax, 15.0, 6.85, 16.2, 6.85)
box(ax, 16.3, 6.4, 1.6, 0.9, "first", fc=CACHE, ec=CACHE, color="white", fs=9)
ax.text(18.2, 6.85, "time to first token", va="center", fontsize=10, color=MUTED)
ax.text(0, 4.7, "Decode: one token per pass, again and again", fontsize=13, fontweight="bold", color=INK)
for k, t in enumerate(["first", "token", "of", "the", "answer"]):
    x = 0.3 + k * 4.3
    box(ax, x, 2.5, 1.5, 0.9, t, fc=CACHE if k == 0 else OLD, ec=CACHE if k == 0 else "#9a98ad",
        color="white" if k == 0 else INK, fs=9)
    if k < 4:
        arrow(ax, x + 1.55, 2.95, x + 1.95, 2.95)
        box(ax, x + 2.0, 2.45, 1.6, 1.0, "model", fc=NEW, ec=NEW, color="white", fs=9, bold=True)
        arrow(ax, x + 3.65, 2.95, x + 4.25, 2.95)
ax.text(0.3, 1.2, "Every pass reads all the weights again, to produce a single token: decode is limited by memory, prefill by arithmetic.",
        fontsize=10, color=MUTED)
fig.savefig(SLIDES / "prefill_decode.png", dpi=150, bbox_inches="tight"); plt.close(fig)

# 5. a GPU and a Tensix grid, schematically (no numbers: structure only)
fig, axes = plt.subplots(1, 2, figsize=(12, 5.2))
ax = axes[0]; blank(ax, 10, 10)
ax.text(5, 9.6, "A GPU (schematic)", ha="center", fontsize=13, fontweight="bold", color=INK)
for r in range(3):
    for c in range(6):
        box(ax, 0.6 + c * 1.5, 6.6 - r * 1.15, 1.2, 0.9, "SM", fc=OLD, fs=8)
box(ax, 0.6, 2.4, 8.7, 0.9, "one large shared cache (L2)", fc=SKIP, fs=10)
box(ax, 0.6, 0.9, 8.7, 0.9, "memory (HBM)", fc=CACHE, ec=CACHE, color="white", fs=10)
for c in range(6):
    arrow(ax, 1.2 + c * 1.5, 4.25, 1.2 + c * 1.5, 3.35)
arrow(ax, 5, 2.35, 5, 1.85)
ax.text(5, 0.2, "many cores share one cache and one memory;\nthe hardware moves data through the cache for you",
        ha="center", va="top", fontsize=9.5, color=MUTED)
ax = axes[1]; blank(ax, 10, 10)
ax.text(5, 9.6, "A Tenstorrent chip (schematic)", ha="center", fontsize=13, fontweight="bold", color=INK)
for r in range(4):
    for c in range(5):
        x, y = 1.0 + c * 1.6, 6.9 - r * 1.45
        box(ax, x, y, 1.25, 1.1, "", fc=OLD, fs=8)
        box(ax, x + 0.12, y + 0.12, 1.01, 0.38, "SRAM", fc="white", fs=7, round_=0.04)
        ax.text(x + 0.62, y + 0.78, "Tensix", ha="center", va="center", fontsize=7.5, color=INK)
        if c < 4:
            ax.plot([x + 1.25, x + 1.6], [y + 0.55, y + 0.55], color=NEW, lw=1.6)
        if r < 3:
            ax.plot([x + 0.62, x + 0.62], [y, y - 0.35], color=NEW, lw=1.6)
box(ax, 0.1, 2.6, 0.6, 5.4, "", fc=CACHE, ec=CACHE)
ax.text(0.4, 5.3, "DRAM", rotation=90, ha="center", va="center", fontsize=9, color="white", fontweight="bold")
box(ax, 9.25, 2.6, 0.6, 5.4, "", fc=SKIP)
ax.text(9.55, 5.3, "Ethernet", rotation=90, ha="center", va="center", fontsize=9, color=INK)
ax.text(5, 2.2, "each core has its own fast memory (SRAM); data moves\nbetween cores explicitly, over a network on the chip (purple);\ncards connect to each other over standard Ethernet",
        ha="center", va="top", fontsize=9.5, color=MUTED)
fig.savefig(SLIDES / "gpu_vs_tensix.png", dpi=150, bbox_inches="tight"); plt.close(fig)

# 6. speculative decoding
fig, ax = plt.subplots(figsize=(11, 4.4)); blank(ax, 22, 8.8)
ax.text(0, 8.3, "Prompt:", fontsize=11, color=MUTED)
for i, t in enumerate(["1,", "2,", "3,", "4,"]):
    box(ax, 2.0 + i * 1.3, 7.8, 1.1, 0.8, t, fc=SKIP, fs=10)
ax.text(0, 6.6, "1. A small draft model guesses the next few tokens, quickly:", fontsize=11, color=INK)
for i, t in enumerate(["5,", "6,", "7,", "9"]):
    box(ax, 2.0 + i * 1.3, 5.5, 1.1, 0.8, t, fc=OLD, fs=10)
ax.text(0, 4.3, "2. The large model checks all the guesses in one pass, the way prefill reads a prompt:", fontsize=11, color=INK)
marks = [("5,", True), ("6,", True), ("7,", True), ("9", False)]
for i, (t, ok) in enumerate(marks):
    box(ax, 2.0 + i * 1.3, 3.2, 1.1, 0.8, t, fc=NEW if ok else "white", ec=NEW if ok else "#c0392b", color="white" if ok else "#c0392b", fs=10)
    ax.text(2.55 + i * 1.3, 2.75, "✓" if ok else "✗", ha="center", va="top", fontsize=13, color=NEW if ok else "#c0392b")
box(ax, 2.0 + 4 * 1.3, 3.2, 1.1, 0.8, "8,", fc=CACHE, ec=CACHE, color="white", fs=10)
ax.text(2.0 + 5 * 1.3 + 0.2, 3.6, "← the large model's own choice", va="center", fontsize=10, color=CACHE)
ax.text(0, 1.4, "3. Keep the correct guesses and the large model's correction: four tokens from one pass of the large model.",
        fontsize=11, color=INK)
fig.savefig(SLIDES / "speculative_decoding.png", dpi=150, bbox_inches="tight"); plt.close(fig)
print("slide diagrams ->", sorted(p.name for p in SLIDES.glob("*.png")))
