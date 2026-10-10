#!/usr/bin/env python3
"""Download every model the workshop uses into WEIGHTS_DIR, once, before the session.

Run on each workshop machine during setup (or once on shared storage):
    python tools/stage_weights.py              # both models, about 4 GB
    python tools/stage_weights.py --gpt2-only  # only GPT-2 (Acts II and IV), 0.5 GB
Afterwards the notebooks run with HF_HUB_OFFLINE=1 and never touch the internet.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from huggingface_hub import snapshot_download
from common.config import GPT2_MODEL_ID, LLM_MODEL_ID, local_weights

MODELS = [(GPT2_MODEL_ID, ["config.json", "model.safetensors", "tokenizer*", "vocab.json", "merges.txt", "generation_config.json"])]
if "--gpt2-only" not in sys.argv:          # Acts II and IV need only GPT-2
    MODELS.append((LLM_MODEL_ID, ["*.json", "*.safetensors", "tokenizer*", "*.model", "*.txt"]))
for model_id, patterns in MODELS:
    dest = local_weights(model_id)
    print(f"{model_id} -> {dest}", flush=True)
    snapshot_download(model_id, local_dir=dest, allow_patterns=patterns)
    size = sum(p.stat().st_size for p in dest.rglob("*") if p.is_file()) / 1e9
    print(f"  {size:.2f} GB", flush=True)
print("STAGING DONE")
