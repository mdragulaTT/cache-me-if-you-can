"""Warm-up, started in the background by Act 0.

Compiles the kernels the acts need by running each configuration once, in two parts:
  part 1 (started by Act 0):      the Act I model at batch size 1
  part 2 (started by Act I's end): GPT-2 for Acts II and IV, then Act III's batch sizes and preset
`python warmup.py all` runs both. Ends with WARMUP DONE, or WARMUP FAILED and the error.
"""
import os
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import config  # noqa: E402


def say(msg):
    print(f"[warm-up {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def warm_llm(**kwargs):
    from common.llm import LLM
    t0 = time.perf_counter()
    llm = LLM(**kwargs)
    llm.generate("Say hello to the workshop in one sentence.", max_new_tokens=16)
    llm.close()
    say(f"language model {kwargs} ready ({time.perf_counter() - t0:.0f} s)")


def warm_gpt2():
    import ttnn
    from common import gpt2_reference as ref
    from common.gpt2 import GPT2
    t0 = time.perf_counter()
    device = ttnn.open_device(device_id=0)
    try:
        model = GPT2(device, config.local_weights(config.GPT2_MODEL_ID))
        for use_cache in (False, True):
            ref.generate(model, config.ACT2_PROMPT, config.ACT2_NEW_TOKENS, use_cache)
        filler = model.tokenizer.encode(" The chip reads the weights again for every token it writes.") * 200
        for n in config.ACT2_CONTEXT_LENGTHS:
            ref.forward_full(model, filler[:n] + [262])
            cache = ref.KVCache(model.n_layers)
            ref.forward_with_cache(model, cache, filler[:n])
            ref.forward_with_cache(model, cache, [262])
        from common import speedrun                  # Act IV: the starting point and the decode operations
        speedrun.reference(model, device)
        speedrun.warm_decode_ops(model, device)
    finally:
        ttnn.close_device(device)
    say(f"GPT-2 ready, Acts II and IV ({time.perf_counter() - t0:.0f} s)")


print(f"pid={os.getpid()}", flush=True)       # lets wait_for_warmup() notice if this process dies
part = sys.argv[1] if len(sys.argv) > 1 else "all"
try:
    say(f"start (part {part})")
    if part in ("1", "all"):
        warm_llm(max_batch_size=1)                   # Act I
    if part in ("2", "all"):
        warm_gpt2()                                  # Acts II and IV
        for b in config.ACT3_BATCH_SIZES:            # Act III
            if b != 1:
                warm_llm(max_batch_size=b)
        warm_llm(max_batch_size=1, preset="performance")
    say("WARMUP DONE")
except Exception:
    traceback.print_exc()
    say("WARMUP FAILED")
    sys.exit(1)
