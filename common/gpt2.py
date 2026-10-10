"""GPT-2 small (124M parameters) in TT-NN — every part except attention.

Act II writes attention and the KV cache itself, in the notebook, so the mechanism stays in plain
sight. This file holds the rest of the model: loading the real weights, token and position
embeddings, layer norms, the projection matrices and the feed-forward block.
"""
import json
from pathlib import Path

import torch
import ttnn
from safetensors.torch import load_file
from transformers import AutoTokenizer


class GPT2:
    def __init__(self, device, weights_dir: Path):
        cfg = json.loads((weights_dir / "config.json").read_text())
        self.device = device
        # Add up products in 32-bit ("fp32 accumulation"). With TT-NN's default 16-bit accumulation this GPT-2
        # picked the same next token as the original PyTorch model only 73% of the time; with it, 85%, and it
        # predicts real text as well as the original (31.9% vs 31.6% on the Act IV texts).
        self.precise = ttnn.init_device_compute_kernel_config(device.arch(), math_fidelity=ttnn.MathFidelity.HiFi4,
                                                              math_approx_mode=False, fp32_dest_acc_en=True,
                                                              packer_l1_acc=False)
        self.n_layers, self.n_heads, self.d_model = cfg["n_layer"], cfg["n_head"], cfg["n_embd"]
        self.d_head = self.d_model // self.n_heads
        self.max_positions = cfg["n_positions"]
        self.tokenizer = AutoTokenizer.from_pretrained(weights_dir)

        sd = {k.removeprefix("transformer."): v.float() for k, v in load_file(weights_dir / "model.safetensors").items()}
        self.wte, self.wpe = sd["wte.weight"], sd["wpe.weight"]       # embeddings: looked up on the host

        def mat(t):   # weight matrix, bf16, tiled, in device DRAM
            return ttnn.from_torch(t.to(torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device)
        def vec(t):   # bias / norm parameter as a [1, n] row
            return mat(t.reshape(1, -1))

        self.layers = []
        for i in range(self.n_layers):
            p = f"h.{i}."
            self.layers.append(dict(
                ln1_w=vec(sd[p + "ln_1.weight"]), ln1_b=vec(sd[p + "ln_1.bias"]),
                w_qkv=mat(sd[p + "attn.c_attn.weight"]), b_qkv=vec(sd[p + "attn.c_attn.bias"]),  # [768, 3*768]
                w_o=mat(sd[p + "attn.c_proj.weight"]), b_o=vec(sd[p + "attn.c_proj.bias"]),
                ln2_w=vec(sd[p + "ln_2.weight"]), ln2_b=vec(sd[p + "ln_2.bias"]),
                w_fc=mat(sd[p + "mlp.c_fc.weight"]), b_fc=vec(sd[p + "mlp.c_fc.bias"]),
                w_proj=mat(sd[p + "mlp.c_proj.weight"]), b_proj=vec(sd[p + "mlp.c_proj.bias"]),
            ))
        self.lnf_w, self.lnf_b = vec(sd["ln_f.weight"]), vec(sd["ln_f.bias"])
        self.lm_head = mat(self.wte.T.contiguous())                   # output projection shares the embedding

    # --- pieces the notebook strings together ----------------------------------------------------
    def embed(self, token_ids, start_pos):
        """Token + position embeddings for tokens that sit at positions start_pos, start_pos+1, ..."""
        assert start_pos + len(token_ids) <= self.max_positions, "GPT-2 sees at most 1024 positions"
        x = self.wte[torch.tensor(token_ids)] + self.wpe[start_pos:start_pos + len(token_ids)]
        return ttnn.from_torch(x.unsqueeze(0).to(torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=self.device)

    def _split_heads(self, t):
        """[1, T, d_model] -> [1, heads, T, d_head]"""
        T = t.shape[1]
        return ttnn.permute(ttnn.reshape(t, (1, T, self.n_heads, self.d_head)), (0, 2, 1, 3))

    def qkv(self, x, layer):
        """Layer norm, then project every token to its query, key and value: each [1, heads, T, d_head]."""
        L = self.layers[layer]
        h = ttnn.layer_norm(x, weight=L["ln1_w"], bias=L["ln1_b"], epsilon=1e-5, compute_kernel_config=self.precise)
        qkv = ttnn.linear(h, L["w_qkv"], bias=L["b_qkv"], compute_kernel_config=self.precise)
        d = self.d_model
        return (self._split_heads(qkv[:, :, 0:d]),
                self._split_heads(qkv[:, :, d:2 * d]),
                self._split_heads(qkv[:, :, 2 * d:3 * d]))

    def finish_block(self, x, attn_out, layer):
        """Merge heads, output projection, residual; then layer norm, feed-forward, residual."""
        L = self.layers[layer]
        T = attn_out.shape[2]
        a = ttnn.reshape(ttnn.permute(attn_out, (0, 2, 1, 3)), (1, T, self.d_model))
        x = ttnn.add(x, ttnn.linear(a, L["w_o"], bias=L["b_o"], compute_kernel_config=self.precise))
        h = ttnn.layer_norm(x, weight=L["ln2_w"], bias=L["ln2_b"], epsilon=1e-5, compute_kernel_config=self.precise)
        h = ttnn.gelu(ttnn.linear(h, L["w_fc"], bias=L["b_fc"], compute_kernel_config=self.precise))
        return ttnn.add(x, ttnn.linear(h, L["w_proj"], bias=L["b_proj"], compute_kernel_config=self.precise))

    def next_token_logits(self, x):
        """Final layer norm and output projection for the last position only. Returns a host tensor."""
        T = x.shape[1]
        h = ttnn.layer_norm(x[:, T - 1:T, :], weight=self.lnf_w, bias=self.lnf_b, epsilon=1e-5, compute_kernel_config=self.precise)
        return ttnn.to_torch(ttnn.matmul(h, self.lm_head, compute_kernel_config=self.precise)).float()[0, -1]


def causal_mask(device, n_new, n_total):
    """Additive mask for n_new queries (the last n_new positions) over n_total keys.

    Query i sits at position n_total - n_new + i and may look at keys 0 .. that position.
    Returns None when nothing needs masking (a single new token may see everything before it).
    """
    if n_new == 1:
        return None
    pos_q = torch.arange(n_total - n_new, n_total).view(-1, 1)
    pos_k = torch.arange(n_total).view(1, -1)
    m = torch.where(pos_k > pos_q, float("-inf"), 0.0).view(1, 1, n_new, n_total)
    return ttnn.from_torch(m.to(torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device)
