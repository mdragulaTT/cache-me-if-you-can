"""The reference attention and KV cache for Act II, used by the warm-up and the tests.

Mirrors the solution notebook exactly, so that the warm-up compiles the same kernels the notebook
will run. If you change Act II's solution, change this file too.
"""
import math
import time

import ttnn

from .gpt2 import causal_mask


def attention(model, q, k, v, mask):
    scores = ttnn.matmul(q, ttnn.permute(k, (0, 1, 3, 2))) * (1.0 / math.sqrt(model.d_head))
    if mask is not None:
        scores = ttnn.add(scores, mask)
    return ttnn.matmul(ttnn.softmax(scores, dim=-1), v)


class KVCache:
    def __init__(self, n_layers):
        self.k, self.v, self.length = [None] * n_layers, [None] * n_layers, 0

    def append(self, layer, k_new, v_new):
        if self.k[layer] is None:
            self.k[layer], self.v[layer] = k_new, v_new
        else:
            self.k[layer] = ttnn.concat([self.k[layer], k_new], dim=2)
            self.v[layer] = ttnn.concat([self.v[layer], v_new], dim=2)
        return self.k[layer], self.v[layer]


def forward_full(model, token_ids):
    T = len(token_ids)
    x = model.embed(token_ids, start_pos=0)
    mask = causal_mask(model.device, n_new=T, n_total=T)
    for layer in range(model.n_layers):
        q, k, v = model.qkv(x, layer)
        x = model.finish_block(x, attention(model, q, k, v, mask), layer)
    return model.next_token_logits(x)


def forward_with_cache(model, cache, new_ids):
    x = model.embed(new_ids, start_pos=cache.length)
    for layer in range(model.n_layers):
        q, k, v = model.qkv(x, layer)
        k_all, v_all = cache.append(layer, k, v)
        mask = causal_mask(model.device, n_new=len(new_ids), n_total=k_all.shape[2])
        x = model.finish_block(x, attention(model, q, k_all, v_all, mask), layer)
    cache.length += len(new_ids)
    return model.next_token_logits(x)


def generate(model, prompt, n_new, use_cache):
    ids = model.tokenizer.encode(prompt)
    n_prompt, step_ms = len(ids), []
    cache = KVCache(model.n_layers) if use_cache else None
    logits = forward_with_cache(model, cache, ids) if use_cache else forward_full(model, ids)
    for _ in range(n_new):
        ids.append(int(logits.argmax()))
        t0 = time.perf_counter()
        logits = forward_with_cache(model, cache, ids[-1:]) if use_cache else forward_full(model, ids)
        step_ms.append((time.perf_counter() - t0) * 1e3)
    return model.tokenizer.decode(ids[n_prompt:]), step_ms
