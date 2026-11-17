"""
Collect per-channel activation statistics of a Hugging Face Llama 3.x / Qwen2.5 model on a
calibration text (fp32 numpy forward pass, no torch). The result feeds export_hf_q8.py:
  --imatrix stats.npz   importance-weighted Q4 rounding (mean x^2 of each input channel)

    python3 calibrate.py <hf dir> <calibration.txt> <stats.npz> [max_tokens]

Keys: "<input>.<layer>.sq" (mean x^2) and "<input>.<layer>.max" (max |x|) for the inputs
att_in (wq/wk/wv), wo_in (wo), ffn_in (w1/w3), w2_in (w2).
"""
import json
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from export_hf_q8 import load_safetensors  # noqa: E402


def bpe_tokenizer(model_dir, arch):
    tj = json.load(open(os.path.join(model_dir, "tokenizer.json")))
    vocab = tj["model"]["vocab"]
    ranks = {tuple(m.split(" ")) if isinstance(m, str) else tuple(m): i for i, m in enumerate(tj["model"]["merges"])}
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    b2u = {b: chr(c) for b, c in zip(bs, cs)}
    digits = "1" if arch == "qwen2" else "1,3"
    pat = re.compile(r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|(?:[^\r\n\w]|_)?[^\W\d_]+|\d{" + digits + r"}| ?(?:[^\s\w]|_)+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+")

    def bpe(word):
        parts = list(word)
        while len(parts) > 1:
            best = min((ranks.get((parts[i], parts[i + 1]), 1e18), i) for i in range(len(parts) - 1))
            if best[0] == 1e18:
                break
            i = best[1]
            parts[i:i + 2] = [parts[i] + parts[i + 1]]
        return [vocab[p] for p in parts]

    def encode(text):
        out = []
        for piece in pat.findall(text):
            out += bpe("".join(b2u[b] for b in piece.encode("utf-8")))
        return out

    return encode


def main():
    model_dir, text_path, out_path = sys.argv[1], sys.argv[2], sys.argv[3]
    max_tokens = int(sys.argv[4]) if len(sys.argv) > 4 else 2048
    cfg = json.load(open(os.path.join(model_dir, "config.json")))
    T = {}
    for f in sorted(x for x in os.listdir(model_dir) if x.endswith(".safetensors")):
        T.update(load_safetensors(os.path.join(model_dir, f)))
    ids = bpe_tokenizer(model_dir, cfg.get("model_type"))(open(text_path, encoding="utf-8").read())[:max_tokens]
    D, H, KV, L = cfg["hidden_size"], cfg["num_attention_heads"], cfg["num_key_value_heads"], cfg["num_hidden_layers"]
    hd, eps, N = D // H, cfg["rms_norm_eps"], len(ids)
    inv = 1.0 / cfg["rope_theta"] ** (np.arange(0, hd, 2) / hd)
    rs = cfg.get("rope_scaling")
    if rs and rs.get("rope_type") == "llama3":
        f, lo, hi, old = rs["factor"], rs["low_freq_factor"], rs["high_freq_factor"], rs["original_max_position_embeddings"]
        wl = 2 * np.pi / inv
        sm = (old / wl - lo) / (hi - lo)
        inv = np.where(wl < old / hi, inv, np.where(wl > old / lo, inv / f, (1 - sm) * inv / f + sm * inv))
    ang = np.arange(N)[:, None] * inv[None, :]
    cos, sin = np.cos(np.concatenate([ang, ang], 1)), np.sin(np.concatenate([ang, ang], 1))

    def rot(x):
        x1, x2 = x[..., : hd // 2], x[..., hd // 2:]
        return x * cos[:, None] + np.concatenate([-x2, x1], -1) * sin[:, None]

    def rms(x, w):
        return x / np.sqrt((x * x).mean(-1, keepdims=True) + eps) * w

    g = lambda n: T[n].astype(np.float32)
    gb = lambda n: T[n].astype(np.float32) if n in T else 0.0
    stats = {}

    def st(name, l, x):
        stats[f"{name}.{l}.sq"] = (x * x).mean(0).astype(np.float32)
        stats[f"{name}.{l}.max"] = np.abs(x).max(0).astype(np.float32)

    x = g("model.embed_tokens.weight")[ids]
    mask = np.triu(np.full((N, N), -np.inf, np.float32), 1)
    for l in range(L):
        p = f"model.layers.{l}."
        h = rms(x, g(p + "input_layernorm.weight"))
        st("att_in", l, h)
        q = rot((h @ g(p + "self_attn.q_proj.weight").T + gb(p + "self_attn.q_proj.bias")).reshape(N, H, hd))
        k = rot((h @ g(p + "self_attn.k_proj.weight").T + gb(p + "self_attn.k_proj.bias")).reshape(N, KV, hd))
        v = (h @ g(p + "self_attn.v_proj.weight").T + gb(p + "self_attn.v_proj.bias")).reshape(N, KV, hd)
        k, v = np.repeat(k, H // KV, 1), np.repeat(v, H // KV, 1)
        s = np.einsum("qhd,khd->hqk", q, k) / np.sqrt(hd) + mask
        s = np.exp(s - s.max(-1, keepdims=True))
        s /= s.sum(-1, keepdims=True)
        o = np.einsum("hqk,khd->qhd", s, v).reshape(N, D)
        st("wo_in", l, o)
        x = x + o @ g(p + "self_attn.o_proj.weight").T
        h = rms(x, g(p + "post_attention_layernorm.weight"))
        st("ffn_in", l, h)
        a, b = h @ g(p + "mlp.gate_proj.weight").T, h @ g(p + "mlp.up_proj.weight").T
        m = a / (1 + np.exp(-a)) * b
        st("w2_in", l, m)
        x = x + m @ g(p + "mlp.down_proj.weight").T
        print(f"layer {l + 1}/{L}", end="\r", flush=True)
    np.savez(out_path, **stats)
    print(f"\n{N} calibration tokens -> {out_path}")


if __name__ == "__main__":
    main()
