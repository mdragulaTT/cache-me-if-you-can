#!/usr/bin/env python3
"""Preparation check: how much of a decode step is the host sending commands?

Measures decode time per token for the Act I model with trace on and with trace off, and loads the
model twice in one process (Acts III and IV do this). Run on a workshop machine:
    python 2026/tools/check_dispatch.py
and copy the two numbers into Act I (search for CALIBRATE).
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.llm import LLM

PROMPT = "Explain in a few sentences why decode is slower per token than prefill."
results = {}
for trace in (True, False):
    llm = LLM(max_batch_size=1, enable_trace=trace)
    llm.generate(PROMPT, max_new_tokens=8)                                     # untimed: compile, capture trace
    r = llm.generate(PROMPT, max_new_tokens=64, stop_at_eos=False)
    f = llm.facts
    limit_ms = f["weight_bytes_per_token"] / (llm.card["bandwidth_gbs"] * 1e9) * 1e3
    results[trace] = r.decode_ms_per_token
    print(f"trace {'on ' if trace else 'off'}: {r.decode_ms_per_token:5.1f} ms/token   "
          f"(card {llm.card['name']}, bandwidth limit {limit_ms:.1f} ms/token, load {llm.load_s:.0f} s)", flush=True)
    llm.close()
on, off = results[True], results[False]
print(f"host command sending accounts for about {off - on:.1f} ms of {off:.1f} ms per token without trace; "
      f"trace removes it")
print("CHECK DONE")
