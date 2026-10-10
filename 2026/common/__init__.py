"""Shared helpers for the MLinPL 2026 notebooks.

tt_transformers (used by Acts I and III) lives in the tt-metal source tree under `models/`.
If it is not importable yet, look for the tt-metal checkout via TT_METAL_HOME.

The card serves one process at a time, and a process that has used it keeps it until the process
ends (closing the device is not enough). So: Act 0's roll call runs in its own short-lived
process, every act ends by stopping its own kernel (release_card), and every act starts by
waiting until nothing else holds the card (wait_for_warmup), saying who holds it if it waits.
"""
import os
import sys
import time
from pathlib import Path

# Keep notebook output readable: library progress bars and log chatter off, errors on.
# Set MLINPL_LOG_LEVEL=INFO (or DEBUG) to see everything when debugging.
import warnings
os.environ.setdefault("TQDM_DISABLE", "1")
os.environ.setdefault("TT_LOGGER_LEVEL", "ERROR")
os.environ.setdefault("LOGURU_LEVEL", "WARNING")       # for helper processes started from a notebook
warnings.filterwarnings("ignore", message="IProgress not found")
try:
    from loguru import logger as _logger
    _logger.remove()
    _logger.add(sys.stderr, level=os.environ.get("MLINPL_LOG_LEVEL", "ERROR"))
except ImportError:
    pass

try:
    import models.tt_transformers  # noqa: F401
except ImportError:
    for candidate in (os.environ.get("TT_METAL_HOME"), os.environ.get("MLINPL_TT_METAL")):
        if candidate and (Path(candidate) / "models" / "tt_transformers").is_dir():
            sys.path.insert(0, candidate)
            break


def _warmup_alive(text):
    """True if the warm-up process named on the log's first line is still running."""
    first = text.splitlines()[0] if text else ""
    if not first.startswith("pid="):
        return True                     # not started writing yet
    try:
        os.kill(int(first[4:]), 0)
        return True
    except (OSError, ValueError):
        return False


def _part_running(log):
    if not log.exists():
        return False
    text = log.read_text(errors="replace")
    return "WARMUP DONE" not in text and "WARMUP FAILED" not in text and _warmup_alive(text)


def card_holders():
    """Other processes that hold the Tenstorrent card, as (process id, command line)."""
    me, found = os.getpid(), []
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit() or int(proc.name) == me:
            continue
        try:
            fds = list((proc / "fd").iterdir())
        except OSError:
            continue
        for fd in fds:
            try:
                if "tenstorrent" in os.readlink(fd):
                    cmd = (proc / "cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace").strip()
                    found.append((int(proc.name), cmd))
                    break
            except OSError:
                pass
    return found


def release_other_notebooks():
    """Stop every other notebook kernel that still holds the card. Their outputs stay on the page;
    their variables are lost."""
    import signal
    stopped = [pid for pid, cmd in card_holders() if "ipykernel" in cmd]
    for pid in stopped:
        os.kill(pid, signal.SIGTERM)
    print(f"stopped {len(stopped)} other notebook kernel(s): {stopped}" if stopped else "no other notebook holds the card")


def release_card():
    """Stop this notebook's kernel, which frees the card for the next act and for the warm-up.
    The outputs stay on the page. (A process that has used the card keeps it until it ends.)"""
    import threading
    print("This notebook's kernel stops now, so the next act can use the card. Your outputs stay on the page.",
          flush=True)
    if not os.environ.get("MLINPL_KEEP_KERNEL"):
        threading.Timer(1.0, os._exit, args=(0,)).start()


def start_warmup(part):
    """Start warm-up part 1 or 2 in the background. It keeps running even if the notebook's kernel stops."""
    import subprocess
    from .config import WARMUP_LOGS
    log = WARMUP_LOGS[part]
    if _part_running(log):
        print(f"warm-up part {part} is already running; progress in {log}")
        return None
    script = Path(__file__).resolve().parent / "warmup.py"
    env = {**os.environ, "PYTHONUNBUFFERED": "1", "TT_LOGGER_LEVEL": "warn"}  # its log shows "waiting for the card"
    proc = subprocess.Popen([sys.executable, str(script), str(part)], stdout=log.open("w"), stderr=subprocess.STDOUT,
                            stdin=subprocess.DEVNULL, cwd=str(script.parents[2]), env=env, start_new_session=True)
    print(f"warm-up part {part} started in the background (process {proc.pid}); progress in {log}")
    return proc


def wait_for_warmup(timeout_s=2400):
    """Wait until nothing else holds the card: a warm-up still running, another notebook, another program.
    Says who holds it while it waits."""
    from .config import WARMUP_LOGS
    t0, told, waited_for = time.time(), set(), set()
    while True:
        running = [part for part, log in WARMUP_LOGS.items() if _part_running(log)]
        holders = card_holders()
        kernels = [pid for pid, cmd in holders if "ipykernel" in cmd]
        others = [(pid, cmd) for pid, cmd in holders if "ipykernel" not in cmd and "warmup.py" not in cmd]
        if not running and not holders:
            for part in sorted(waited_for):
                text = WARMUP_LOGS[part].read_text(errors="replace")
                print(f"Warm-up part {part} finished." if "WARMUP DONE" in text else
                      f"Warm-up part {part} stopped before finishing (see notebooks/{WARMUP_LOGS[part].name}); "
                      "the first runs will be slower while kernels compile.")
            return
        for part in running:
            if part not in waited_for:
                print(f"Warm-up part {part} is still using the card; waiting for it to finish...", flush=True)
                waited_for.add(part)
        if kernels and "kernels" not in told:
            print(f"Another notebook still holds the card (process {', '.join(map(str, kernels))}). A program that has "
                  "used the card keeps it until it stops.\nRun the last cell of that notebook, or click Restart there, "
                  "or run here:  from common import release_other_notebooks; release_other_notebooks()", flush=True)
            told.add("kernels")
        if others and "others" not in told:
            print(f"Another program is using the card (process {others[0][0]}: {others[0][1][:80]}); waiting...", flush=True)
            told.add("others")
        if time.time() - t0 > timeout_s:
            raise TimeoutError(f"the card is still busy after {timeout_s} s - call a presenter")
        time.sleep(3)
