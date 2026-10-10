#!/usr/bin/env bash
# Workshop-machine check for MLinPL 2026. Read-only; changes nothing.
# Run inside a Console instance:   bash 2026/tools/pod_probe.sh 2>&1 | tee pod_probe.log
# The log is safe to share: it prints names of environment variables, never their values.
set -u
hr(){ printf '\n== %s ==\n' "$1"; }
check_url(){  # prints OK/NO for one URL without masking curl failures
  if code=$(timeout 15 curl -s -o /dev/null -w '%{http_code}' "$1"); then echo "$1: HTTP $code"; else echo "$1: NO (curl failed)"; fi
}
hr "identity"; date -u; hostname; nproc; free -g | head -2
hr "disk"; df -h / "$HOME" /tmp 2>/dev/null
echo "Needed: ~4 GB for weights (2026/tools/stage_weights.py) + ~1 GB kernel cache"
hr "tenstorrent card"; ls -la /dev/tenstorrent* 2>&1; (command -v tt-smi >/dev/null && timeout 30 tt-smi -ls) 2>&1 | head -20
hr "tt-metal"
echo "TT_METAL_HOME=${TT_METAL_HOME:-<unset>}"
for d in "${TT_METAL_HOME:-}" /opt/tt-metal "$HOME/tt-metal" /tt-metal; do
  [ -n "$d" ] && [ -d "$d/models/tt_transformers" ] && { echo "tt_transformers found in $d"; git -C "$d" describe --tags 2>/dev/null; git -C "$d" log -1 --format='%h %ad' --date=short 2>/dev/null; }
done
hr "python + packages"; command -v python3; python3 --version
python3 - <<'PY' 2>&1 | grep -v DEBUG | head -20
import importlib
for m in ("ttnn", "torch", "transformers", "safetensors", "huggingface_hub", "matplotlib", "ipykernel", "nbclient"):
    try:
        mod = importlib.import_module(m); print(f"{m:16} OK  {getattr(mod, '__version__', '')}")
    except Exception as e:
        print(f"{m:16} MISSING ({type(e).__name__})")
try:
    import models.tt_transformers; print("tt_transformers  importable")
except Exception as e:
    print("tt_transformers  NOT importable:", type(e).__name__, "(set TT_METAL_HOME)")
try:
    import ttnn, torch, time
    t0 = time.perf_counter(); d = ttnn.open_device(device_id=0)
    x = ttnn.from_torch(torch.full((32, 32), 20.0), layout=ttnn.TILE_LAYOUT, device=d)
    ok = ttnn.to_torch(ttnn.multiply(x, 6.0))[0, 0].item() == 120.0
    g = d.compute_with_storage_grid_size()
    print(f"card: {d.arch()}, {g.x}x{g.y} compute cores, one op {'OK' if ok else 'WRONG'} in {time.perf_counter() - t0:.1f}s")
    ttnn.close_device(d)
except Exception as e:
    print("card test FAILED:", repr(e)[:200])
PY
hr "jupyter"; command -v jupyter && jupyter --version 2>&1 | head -6; ls ~/.vscode-server/extensions 2>/dev/null | grep -i -E "jupyter|python" | head -5
hr "network"; for u in https://huggingface.co https://github.com https://pypi.org; do check_url "$u"; done
hr "caches"; echo "HF_HOME=${HF_HOME:-<unset>} TT_METAL_CACHE=${TT_METAL_CACHE:-<unset>} MLINPL_WEIGHTS_DIR=${MLINPL_WEIGHTS_DIR:-<unset>}"
du -sh ~/.cache/* 2>/dev/null | head
hr "pod limits"; cat /sys/fs/cgroup/memory.max 2>/dev/null || cat /sys/fs/cgroup/memory/memory.limit_in_bytes 2>/dev/null
echo "environment variable NAMES mentioning pod/idle/timeout (values hidden):"; env | grep -i -E "kube|pod|timeout|idle|ttl" | cut -d= -f1 | head -20
hr "done"; echo "Next: leave this instance idle for 60+ minutes and check that it survives (lunch-break test)."
