#!/usr/bin/env bash
# One-time setup of a workshop machine: a Console instance (or any box) with one Tenstorrent card.
#
#   bash 2026/tools/setup_instance.sh             # everything (about 4 GB of weights)
#   bash 2026/tools/setup_instance.sh --gpt2-only # only what Acts II and IV need (0.5 GB)
#
# Safe to run again. Uses tt-metal's own Python environment ($TT_METAL_HOME/python_env), whichever
# virtual environment the terminal happens to have active (override with PYTHON=/path/to/python).
set -uo pipefail
cd "$(dirname "$0")/../.." || exit 1
PY=${PYTHON:-}
if [ -z "$PY" ]; then
  for candidate in "${TT_METAL_HOME:-/nonexistent}/python_env/bin/python3" "$HOME/tt-metal/python_env/bin/python3" python3; do
    if "$candidate" -c "import ttnn" >/dev/null 2>&1; then PY=$candidate; break; fi
  done
fi
PY=${PY:-python3}
echo "Using Python: $("$PY" -c 'import sys; print(sys.executable)' 2>/dev/null || echo "$PY")"
step(){ printf '\n== %s\n' "$1"; }

step "1/4 checking the machine (full report in pod_probe.log)"
bash 2026/tools/pod_probe.sh > pod_probe.log 2>&1
grep -E "^tt_transformers found|card:|^Needed|^https://" pod_probe.log | awk '!seen[$0]++' | sed 's/^/   /'
if ! "$PY" -c "import ttnn" >/dev/null 2>&1; then
  echo "   ttnn cannot be imported with $PY. Activate the tt-metal Python environment, or run with PYTHON=/path/to/python."
  exit 1
fi

step "2/4 Python packages"
missing=$("$PY" - <<'PY'
import importlib.util
need = ["transformers", "safetensors", "huggingface_hub", "matplotlib", "nbformat", "nbclient", "ipykernel", "pytest", "loguru"]
print(" ".join(m for m in need if importlib.util.find_spec(m) is None))
PY
)
if [ -n "$missing" ]; then
  echo "   installing: $missing"
  "$PY" -m pip install --quiet $missing || "$PY" -m pip install --quiet --user $missing || { echo "   pip install failed"; exit 1; }
else
  echo "   all present"
fi
# A named notebook kernel for this Python, so "MLinPL 2026" can be picked in VS Code's kernel list.
"$PY" -m ipykernel install --user --name mlinpl-2026 --display-name "MLinPL 2026" >/dev/null 2>&1 \
  && echo "   notebook kernel \"MLinPL 2026\" -> $("$PY" -c 'import sys; print(sys.executable)')"

step "3/4 model weights -> ${MLINPL_WEIGHTS_DIR:-$HOME/mlinpl-weights}"
"$PY" 2026/tools/stage_weights.py "$@" || { echo "   download failed: no internet on this machine? Copy the weights folder (~/mlinpl-weights) over from a machine that has it."; exit 1; }

step "4/4 smoke test: the card computes, GPT-2 answers"
"$PY" - <<'PY' 2>&1 | grep -E "^(OK|FAIL)"
import sys
sys.path.insert(0, "2026")
try:
    import torch, ttnn
    from common import config
    from common.gpt2 import GPT2
    from common import gpt2_reference as ref
    d = ttnn.open_device(device_id=0, trace_region_size=config.TRACE_REGION_BYTES)
    try:
        t = ttnn.from_torch(torch.full((32, 32), 20.0), layout=ttnn.TILE_LAYOUT, device=d)
        assert ttnn.to_torch(ttnn.multiply(t, 6.0))[0, 0].item() == 120.0
        m = GPT2(d, config.local_weights(config.GPT2_MODEL_ID))
        text, _ = ref.generate(m, "The workshop starts", 8, use_cache=True)
        g = d.compute_with_storage_grid_size()
        print(f"OK {d.arch()} with {g.x}x{g.y} compute cores; GPT-2 says: The workshop starts{text!r}")
    finally:
        ttnn.close_device(d)
except Exception as e:
    print(f"FAIL {type(e).__name__}: {str(e)[:300]}")
PY
echo
echo "Done. Open 2026/notebooks/00_hello.ipynb with the \"MLinPL 2026\" kernel."
[ -f 2026/tools/run_all.sh ] && echo "To run every notebook once (and compile all kernels): bash 2026/tools/run_all.sh"
