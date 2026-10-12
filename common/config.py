"""Every setting the notebooks use, in one place.

Change values here, never inside a notebook. Measured values were checked on a Console n150 on
October 5 (ttnn 0.77.0).
"""
import os
from pathlib import Path

# --- Where things live ---------------------------------------------------------------------------
# Model weights are downloaded once by tools/stage_weights.py (setup_instance.sh runs it), so the
# notebooks never fetch anything from the internet.
WEIGHTS_DIR = Path(os.environ.get("MLINPL_WEIGHTS_DIR", Path.home() / "mlinpl-weights"))

# Acts I and III: a production-style model served through tt_transformers (part of tt-metal).
LLM_MODEL_ID = os.environ.get("MLINPL_LLM_MODEL", "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
# Acts II and IV: the model whose attention we write by hand.
GPT2_MODEL_ID = "openai-community/gpt2"


def local_weights(model_id: str) -> Path:
    """Folder holding the pre-downloaded weights and tokenizer of `model_id`."""
    return WEIGHTS_DIR / model_id.split("/")[-1]


# --- Act I: the two phases -----------------------------------------------------------------------
# tt_transformers rounds a prompt up to the next of 128, 1024, 2048 or 4096 tokens before prefill, so
# the time to first token climbs in steps. Three target lengths per step make the steps visible; with
# the chat template the real prompts are 71/99/127, 323/631/967, 1191/1527/1975, 2339/3123/3935 tokens.
ACT1_PROMPT_LENGTHS = (30, 70, 90, 300, 600, 950, 1150, 1500, 1950, 2300, 3100, 3900)  # tokens
PREFILL_STEPS = (128, 1024, 2048, 4096)
ACT1_NEW_TOKENS = 32
MAX_SEQ_LEN = 4096

# --- Act II: the KV cache ------------------------------------------------------------------------
ACT2_PROMPT = "The Tenstorrent workshop at ML in PL starts with a simple question:"
ACT2_NEW_TOKENS = 40
ACT2_CONTEXT_LENGTHS = (64, 256, 512, 896)     # GPT-2 can see at most 1024 positions

# --- Act III: serving many users -----------------------------------------------------------------
# A *user* is anything holding a conversation with the model at the same time: a person in a chat
# window, an agent calling the model in a loop, a job summarising documents.
#
# Why a per-token budget: a fast reader reads ~5 words/s, roughly 6-7 tokens/s (~150 ms per token).
# Slower than that and people watch the cursor. Products aim well above it, because reasoning models
# generate hundreds of "thinking" tokens before the visible answer and an agent chains many calls,
# so every millisecond per token is multiplied. 20 tokens/s per user (50 ms) is a common target.
TOKEN_BUDGET_MS = 50.0          # n150 (Oct 5): 1, 8 and 32 users take 22-25 ms per token
# Why a first-token budget: past ~2 s a chat window feels stuck.
TTFT_BUDGET_MS = 2000.0         # n150 (Oct 5): 40-400 ms for prompts of 128-3,900 tokens
ACT3_BATCH_SIZES = (1, 8, 32)   # batch sizes the warm-up prepares
ACT3_NEW_TOKENS = 48

# --- Act IV: the Speedrun ---------------------------------------------------------------------------
# The race reads each text in common/speedrun.py token by token: the first SPEEDRUN_PROMPT_TOKENS in
# one go (not timed), then SPEEDRUN_RACE_TOKENS one at a time (timed). After every token the model
# predicts the next one.
SPEEDRUN_PROMPT_TOKENS = 128
SPEEDRUN_RACE_TOKENS = 128
# Quality rule: a run counts if it predicts the real next token at most this many percentage points
# less often than the original model. Measured on a p100a: correct fast versions land within about
# 2 points of the original, 4-bit weights everywhere lose about 8, real bugs lose 15 or more.
SPEEDRUN_QUALITY_TOLERANCE = 3.0
# A leaderboard, for running Act IV as a live race: speedrun.submit() posts a valid result to a Google
# Form with three short-answer questions, and the form's sheet is the board. Put the form's
# .../formResponse URL and its entry ids here, and add a speedrun.submit(...) cell to Act IV.
SPEEDRUN_FORM_URL = None
SPEEDRUN_FORM_FIELDS = {"nickname": "entry.0", "ms_per_token": "entry.0", "what_you_changed": "entry.0"}
# Device memory reserved for recorded traces (Act IV opens the card with this).
TRACE_REGION_BYTES = 64 * 1024 * 1024

# --- Hardware facts used by the napkin math --------------------------------------------------------
# Peak DRAM bandwidth and memory per card, from Tenstorrent's public specifications.
CARD_FACTS = {
    "n150": dict(arch="Wormhole", dram_gb=12, bandwidth_gbs=288, tensix_cores=72),
    "p100": dict(arch="Blackhole", dram_gb=28, bandwidth_gbs=448, tensix_cores=120),
    "p150": dict(arch="Blackhole", dram_gb=32, bandwidth_gbs=512, tensix_cores=140),
}

# --- Warm-up -----------------------------------------------------------------------------------------
# The warm-up runs in two parts, each in the background with its own log: Act 0 starts part 1 (Act I's
# model, done during Talk 1); the end of Act I starts part 2 (Acts II-IV, done during the break and
# Talk 2). Every act waits until no warm-up is using the card.
WARMUP_LOGS = {part: Path(__file__).resolve().parents[1] / "notebooks" / name
               for part, name in ((1, "warmup.log"), (2, "warmup2.log"))}
WARMUP_LOG = WARMUP_LOGS[1]
