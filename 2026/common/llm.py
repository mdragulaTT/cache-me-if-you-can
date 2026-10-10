"""A thin wrapper around tt_transformers for Acts I, III and IV.

The notebooks only ever use `LLM` and `GenResult` from here, so if the workshop machines run a
different tt-metal version, this is the only file that needs adjusting. The call sequence mirrors
tt-metal's own demo (models/tt_transformers/demo/simple_text_demo.py):
create_tt_model -> Generator -> prefill_forward_text -> decode_forward.
"""
import json
import os
import shutil
import struct
import tempfile
import time
from dataclasses import dataclass, field
from math import prod

from .config import CARD_FACTS, LLM_MODEL_ID, MAX_SEQ_LEN, local_weights

WEIGHTS = local_weights(LLM_MODEL_ID)
# tt_transformers reads weights and tokenizer from HF_MODEL. The weights were downloaded before the
# workshop (tools/stage_weights.py), so the session never touches the internet.
os.environ.setdefault("HF_MODEL", str(WEIGHTS))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
# tt_transformers converts the weights into the card's formats and normally writes the converted
# tensors to disk so the next load can read them back. Reading them back hangs on our test card for
# large tensors (ttnn 0.79: load_tensor_flatbuffer spins; it hit the embedding table and the 32-user
# KV cache). Converting from scratch takes about 15 s, so while the model is built, ttnn.as_tensor is
# called without a cache file: nothing is written to disk and nothing is read back. Its bookkeeping
# files go to a temporary folder, deleted by close().
os.environ.setdefault("TT_TRANSFORMERS_FORCE_MODEL_LOAD", "1")
CONVERT_DIR = WEIGHTS.parent / ".converted"


class _NoTensorDiskCache:
    """Within this block, ttnn.as_tensor ignores cache_file_name (see the comment above)."""

    def __enter__(self):
        self._original = ttnn.as_tensor
        original = self._original

        def as_tensor(*args, cache_file_name=None, **kwargs):
            return original(*args, **kwargs)

        ttnn.as_tensor = as_tensor

    def __exit__(self, *exc):
        ttnn.as_tensor = self._original

import torch  # noqa: E402
import ttnn  # noqa: E402
from models.tt_transformers.tt.common import (  # noqa: E402
    PagedAttentionConfig,
    create_tt_model,
    preprocess_inputs_prefill,
)
from models.tt_transformers.tt.generator import Generator  # noqa: E402
from models.tt_transformers.tt.model_config import DecodersPrecision, TensorGroup, determine_device_name  # noqa: E402

# Bytes per number on the device. bfp8 / bfp4: 8 or 4 bits per number, plus one 8-bit exponent
# shared by each block of 16 numbers.
BYTES_PER_VALUE = {ttnn.bfloat16: 2.0, ttnn.bfloat8_b: 1.0625, ttnn.bfloat4_b: 0.5625}


def model_facts():
    """Size and shape of the served model, read from the staged files (no weights are loaded)."""
    cfg = json.loads((WEIGHTS / "config.json").read_text())
    params = 0
    for f in WEIGHTS.glob("*.safetensors"):
        with open(f, "rb") as fh:
            header = json.loads(fh.read(struct.unpack("<Q", fh.read(8))[0]))
        params += sum(prod(v["shape"]) for k, v in header.items() if k != "__metadata__")
    n_heads = cfg["num_attention_heads"]
    return dict(
        name=LLM_MODEL_ID.split("/")[-1],
        params=params,
        n_layers=cfg["num_hidden_layers"],
        n_heads=n_heads,
        n_kv_heads=cfg.get("num_key_value_heads", n_heads),
        d_head=cfg.get("head_dim", cfg["hidden_size"] // n_heads),
    )


@dataclass
class GenResult:
    texts: list
    prompt_lens: list
    ttft_ms: float                 # time to first token: the prefill, for the whole batch
    decode_ms_per_token: float     # mean time of one decode step (one new token for every user)
    n_generated: int               # tokens produced per user
    batch_size: int
    token_times: list = field(default_factory=list, repr=False)

    @property
    def tokens_per_s_per_user(self):
        return 1000.0 / self.decode_ms_per_token

    @property
    def tokens_per_s_total(self):
        return self.tokens_per_s_per_user * self.batch_size

    def __repr__(self):
        return (f"GenResult(users={self.batch_size}, prompt={self.prompt_lens[0]} tokens, "
                f"first token after {self.ttft_ms:.0f} ms, then {self.decode_ms_per_token:.1f} ms/token "
                f"= {self.tokens_per_s_per_user:.1f} tokens/s per user, {self.tokens_per_s_total:.1f} tokens/s total)")


class LLM:
    """One model on one card for a fixed number of users. Create, generate, close."""

    def __init__(self, max_batch_size=1, max_seq_len=MAX_SEQ_LEN, preset="accuracy", instruct=True,
                 enable_trace=True):
        """preset: tt_transformers' precision preset. "accuracy" keeps attention weights and the KV cache
        in bf16 and the feed-forward weights in bfp8; "performance" uses bfp8 for attention and the
        cache and bfp4 for the largest feed-forward matrices."""
        assert preset in ("accuracy", "performance"), preset
        self.max_batch_size, self.max_seq_len = max_batch_size, max_seq_len
        self.instruct, self.enable_trace, self.preset = instruct, enable_trace, preset
        self.dtype_name = preset
        t0 = time.perf_counter()
        CONVERT_DIR.mkdir(parents=True, exist_ok=True)
        self._convert_dir = tempfile.mkdtemp(dir=CONVERT_DIR)
        os.environ["TT_CACHE_PATH"] = self._convert_dir          # read by tt_transformers' ModelArgs
        self.mesh_device = ttnn.open_mesh_device(ttnn.MeshShape(1, 1))
        self.card_name = determine_device_name(self.mesh_device).lower()
        if self.card_name not in CARD_FACTS:
            self.close()
            raise RuntimeError(f"unknown card {self.card_name!r}: add it to CARD_FACTS in common/config.py")
        self.card = CARD_FACTS[self.card_name] | {"name": self.card_name}
        self.paged = PagedAttentionConfig(block_size=32, max_num_blocks=max_batch_size * max_seq_len // 32)
        with _NoTensorDiskCache():
            self.model_args, self.model, kv_cache, _ = create_tt_model(
                self.mesh_device, instruct=instruct, max_batch_size=max_batch_size,
                optimizations=lambda args: getattr(DecodersPrecision, preset)(args.n_layers, args.model_name),
                max_seq_len=max_seq_len, paged_attention_config=self.paged, dtype=ttnn.bfloat8_b,
            )
        self.kv_cache = [kv_cache]                       # one entry per model replica
        self.tokenizer = self.model_args.tokenizer
        self.generator = Generator([self.model], [self.model_args], self.mesh_device, tokenizer=self.tokenizer)
        perm = torch.randperm(self.paged.max_num_blocks)  # a shuffled page table, as in the demo
        self.page_table = torch.argsort(perm).reshape(max_batch_size, self.paged.max_num_blocks // max_batch_size)
        eos = getattr(self.tokenizer, "eos_token_id", None)
        self.stop_tokens = set(getattr(self.tokenizer, "stop_tokens", None) or ([eos] if eos is not None else []))
        # The first generation runs tt_transformers' one-off preparation (a sweep over prompt lengths
        # and capturing the trace). Do it here, so the first timing a participant sees is a real one.
        self.generate("Hi", max_new_tokens=2)
        self.load_s = time.perf_counter() - t0

    @property
    def facts(self):
        """Model shape plus the bytes one decode step must read from DRAM, using the precision each
        group of weights actually has on the device (tt_transformers keeps some in bf16)."""
        f = model_facts()
        cfg = json.loads((WEIGHTS / "config.json").read_text())
        d, ff, vocab = cfg["hidden_size"], cfg["intermediate_size"], cfg["vocab_size"]
        q, kv = f["n_heads"] * f["d_head"], f["n_kv_heads"] * f["d_head"]
        opt = self.model_args.decoders_optimizations
        size = lambda layer, group: BYTES_PER_VALUE[opt.get_tensor_dtype(layer, group)]
        per_layer = [
            d * (q + 2 * kv) * size(i, TensorGroup.WQKV) + q * d * size(i, TensorGroup.WO)
            + 2 * d * ff * size(i, TensorGroup.FF1_FF3) + ff * d * size(i, TensorGroup.FF2)
            for i in range(f["n_layers"])
        ]
        lm_head = vocab * d * BYTES_PER_VALUE[getattr(self.model_args, "lm_head_dtype", ttnn.bfloat8_b)]
        # The input embedding table is not counted: each step reads one row of it, not all of it.
        f["weight_bytes_per_token"] = sum(per_layer) + lm_head
        f["kv_bytes_per_token_of_context"] = 2 * f["n_layers"] * kv * size(0, TensorGroup.KV_CACHE)
        f["preset"] = self.preset
        return f

    def generate(self, prompts, max_new_tokens=64, stop_at_eos=True, on_token=None, pad_prompt=None):
        """Greedy generation. `prompts` is a string or a list; the first `max_batch_size` are used. Empty
        batch places are filled by repeating the prompts, or with `pad_prompt` if given (a short prompt
        keeps the padding cheap).

        Times are wall-clock from the host: `ttft_ms` covers the prefill of every prompt in the
        batch; `decode_ms_per_token` is the mean time of one decode step. With stop_at_eos=False
        every user gets exactly max_new_tokens tokens, which keeps measurements comparable.
        """
        prompts = [prompts] if isinstance(prompts, str) else list(prompts)
        B = self.max_batch_size
        assert prompts, "no prompts"
        prompts = prompts[:B]                      # more prompts than places: the first B are used
        if pad_prompt is not None:
            prompts = prompts + [pad_prompt] * (B - len(prompts))
        else:
            prompts = (prompts * (B // len(prompts) + 1))[:B]
        tokens_list, _, decoding_pos, prefill_lens = preprocess_inputs_prefill(
            prompts, self.tokenizer, [self.model_args], self.instruct, max_new_tokens, max_prefill_len=self.max_seq_len
        )
        tokens = torch.stack(tokens_list).view(B, -1)

        t_start = time.perf_counter()
        logits = self.generator.prefill_forward_text(
            tokens, page_table=self.page_table, kv_cache=self.kv_cache, prompt_lens=decoding_pos,
            enable_trace=self.enable_trace,
        )
        out = torch.argmax(logits, dim=-1).view(B, 1)
        t_first = time.perf_counter()

        pos = torch.tensor(decoding_pos)
        generated = [[int(t)] for t in out[:, 0]]
        done = [stop_at_eos and int(t) in self.stop_tokens for t in out[:, 0]]
        token_times = [t_first]
        if on_token:
            on_token(0, out[:, 0].tolist())

        for step in range(max_new_tokens - 1):
            if all(done):
                break
            logits, _ = self.generator.decode_forward(
                out, pos, enable_trace=self.enable_trace, page_table=self.page_table,
                kv_cache=self.kv_cache, reset_batch=(step == 0),
            )
            out = torch.argmax(logits, dim=-1).view(B, 1)
            pos = pos + 1
            token_times.append(time.perf_counter())
            for b in range(B):
                if not done[b]:
                    tok = int(out[b, 0])
                    generated[b].append(tok)
                    done[b] = stop_at_eos and tok in self.stop_tokens
            if on_token:
                on_token(step + 1, out[:, 0].tolist())

        steps = len(token_times) - 1
        decode_ms = (token_times[-1] - token_times[0]) / steps * 1e3 if steps else float("nan")
        texts = [self.tokenizer.decode(g, skip_special_tokens=True) for g in generated]
        return GenResult(texts, list(prefill_lens), (t_first - t_start) * 1e3, decode_ms,
                         len(generated[0]), B, token_times)

    def close(self):
        """Free everything on the card first, then close the card. tt_transformers' Generator releases its
        recorded traces when it is deleted, and that must happen while the card is still open: in the
        other order newer tt-metal versions report operations on a closed ("remote-only") device."""
        import gc
        for name in ("generator", "model", "kv_cache"):
            self.__dict__.pop(name, None)
        gc.collect()
        ttnn.close_mesh_device(self.mesh_device)
        shutil.rmtree(self._convert_dir, ignore_errors=True)
