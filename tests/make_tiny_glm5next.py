# A tiny random GLM-5.3-Flash style checkpoint (glm5_next) with the real tokenizer, for testing the exporter
# and the engine against the transformers implementation without the 330 GB model:
#   make_tiny_glm5next.py <real hf dir (config, tokenizer)> <out dir>
# Tensor names, dtypes (FP8 e4m3 with 128x128 block scales where the real checkpoint has them, BF16, F32)
# and layer pattern (KDA x3, MLA + indexer; dense FFN, then MoE) follow the real model. index_topk is
# small so that the DSA indexer has to choose pools already in a short prompt.
import json
import os
import shutil
import struct
import sys

import numpy as np
import torch

src, out = sys.argv[1], sys.argv[2]
os.makedirs(out, exist_ok=True)
for f in ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja"):
    shutil.copy(os.path.join(src, f), out)
real = json.load(open(os.path.join(src, "config.json")))
tc = dict(real["text_config"])
L = 6
types = ["linear_attention" if l % 4 != 3 else "deepseek_sparse_attention" for l in range(L)]
tc.update(hidden_size=256, intermediate_size=512, num_hidden_layers=L, num_attention_heads=4, num_key_value_heads=4,
          q_lora_rank=128, kv_lora_rank=128, qk_nope_head_dim=64, qk_head_dim=64, v_head_dim=64,
          n_routed_experts=16, num_experts_per_tok=4, moe_intermediate_size=128, first_k_dense_replace=2,
          index_n_heads=4, index_head_dim=64, index_topk=16, index_kpool=4,
          layer_types=types, mlp_layer_types=["dense"] * 2 + ["sparse"] * (L - 2), indexer_types=["full"] * L,
          max_position_embeddings=4096)
la = dict(tc["linear_attn_config"], num_heads=4, head_dim=64,
          kda_layers=[l for l in range(L) if types[l] == "linear_attention"],
          full_attn_layers=[l for l in range(L) if types[l] != "linear_attention"])
tc["linear_attn_config"] = la
cfg = dict(real, text_config=tc)
json.dump(cfg, open(os.path.join(out, "config.json"), "w"), indent=1)

rng = np.random.default_rng(1234)
D, V, F, mh, E = 256, tc["vocab_size"], 512, 128, 16
H, dh, nh, ql, kvl, qk, vh, ih, ihh = 4, 64, 4, 128, 128, 64, 64, 64, 4
T = {}   # name -> (dtype, np array) ; FP8 tensors get their scale_inv too


def rnd(*shape, std=0.05):
    return (rng.standard_normal(shape) * std).astype(np.float32)


def put(name, a, dt):
    T[name] = (dt, a)


def fp8(name, a):
    """quantize to e4m3 with 128x128 block scales, as the real checkpoint stores it"""
    r, c = a.shape
    br, bc = (r + 127) // 128, (c + 127) // 128
    sc = np.zeros((br, bc), np.float32)
    q = np.zeros_like(a)
    for i in range(br):
        for j in range(bc):
            blk = a[i * 128:(i + 1) * 128, j * 128:(j + 1) * 128]
            s = np.abs(blk).max() / 448.0 + 1e-12
            sc[i, j] = s
            q[i * 128:(i + 1) * 128, j * 128:(j + 1) * 128] = blk / s
    t8 = torch.from_numpy(q).to(torch.float8_e4m3fn)
    T[name] = ("F8_E4M3", t8.view(torch.uint8).numpy())
    T[name + "_scale_inv"] = ("F32", sc)


P = "model.language_model."
put(P + "embed_tokens.weight", rnd(V, D, std=1.0), "BF16")
put("lm_head.weight", rnd(V, D, std=0.05), "BF16")
put(P + "norm.weight", 1 + rnd(D, std=0.1), "BF16")
for l in range(L):
    p = f"{P}layers.{l}."
    put(p + "input_layernorm.weight", 1 + rnd(D, std=0.1), "BF16")
    put(p + "post_attention_layernorm.weight", 1 + rnd(D, std=0.1), "BF16")
    for part in ("attn", "ffn"):
        put(p + f"hc_{part}_fn", rnd(24, 4 * D, std=0.05), "BF16")
        put(p + f"hc_{part}_base", rnd(24, std=0.5), "F32")
        put(p + f"hc_{part}_scale", 1 + rnd(3, std=0.2), "F32")
    a = p + "self_attn."
    if types[l] == "linear_attention":
        for n in "qkv":
            put(a + f"{n}_proj.weight", rnd(H * dh, D), "BF16")
            put(a + f"{n}_conv1d.weight", rnd(H * dh, 1, 4, std=0.4), "BF16")
        put(a + "f_a_proj.weight", rnd(dh, D), "BF16")
        put(a + "f_b_proj.weight", rnd(H * dh, dh, std=0.3), "BF16")
        put(a + "g_a_proj.weight", rnd(dh, D), "BF16")
        put(a + "g_b_proj.weight", rnd(H * dh, dh, std=0.3), "BF16")
        put(a + "b_proj.weight", rnd(H, D, std=0.2), "BF16")
        put(a + "o_proj.weight", rnd(D, H * dh), "BF16")
        put(a + "o_norm.weight", 1 + rnd(dh, std=0.1), "BF16")
        put(a + "dt_bias", rnd(H * dh, std=1.0), "F32")
        put(a + "A_log", rnd(H, std=0.5), "F32")
    else:
        fp8(a + "q_a_proj.weight", rnd(ql, D))
        put(a + "q_a_layernorm.weight", 1 + rnd(ql, std=0.1), "BF16")
        fp8(a + "q_b_proj.weight", rnd(nh * qk, ql, std=0.1))
        fp8(a + "kv_a_proj_with_mqa.weight", rnd(kvl, D))
        put(a + "kv_a_layernorm.weight", 1 + rnd(kvl, std=0.1), "BF16")
        put(a + "kv_b_proj.weight", rnd(nh * (qk + vh), kvl, std=0.1), "BF16")
        fp8(a + "o_proj.weight", rnd(D, nh * vh))
        put(a + "indexer.wq_b.weight", rnd(ihh * ih, ql, std=0.1), "BF16")
        put(a + "indexer.wk.weight", rnd(ih, D), "BF16")
        put(a + "indexer.k_norm.weight", 1 + rnd(ih, std=0.1), "BF16")
        put(a + "indexer.k_norm.bias", rnd(ih, std=0.1), "BF16")
        put(a + "indexer.weights_proj.weight", rnd(ihh, D, std=0.2), "BF16")
        put(a + "indexer.index_kpool_compress_ape", rnd(4, ih, std=0.5), "BF16")
        put(a + "indexer.index_kpool_compress_gate", rnd(ih, D), "BF16")
    m = p + "mlp."
    if l < 2:
        fp8(m + "gate_proj.weight", rnd(F, D))
        fp8(m + "up_proj.weight", rnd(F, D))
        fp8(m + "down_proj.weight", rnd(D, F))
    else:
        put(m + "gate.weight", rnd(E, D, std=0.1), "BF16")
        put(m + "gate.e_score_correction_bias", rnd(E, std=0.05), "F32")
        for n, sh in (("gate_proj", (mh, D)), ("up_proj", (mh, D)), ("down_proj", (D, mh))):
            fp8(m + f"shared_experts.{n}.weight", rnd(*sh))
            for e in range(E):
                fp8(m + f"experts.{e}.{n}.weight", rnd(*sh))

# write two shards (the exporter has to cope with any split)
names = sorted(T)
half = len(names) // 2
index = {}
for k, part in enumerate((names[:half], names[half:])):
    fn = f"model-{k + 1:05d}-of-00002.safetensors"
    hdr, off, blobs = {}, 0, []
    for n in part:
        dt, a = T[n]
        if dt == "BF16":
            b = (torch.from_numpy(np.ascontiguousarray(a)).to(torch.bfloat16).view(torch.int16).numpy()).tobytes()
        else:
            b = np.ascontiguousarray(a).tobytes()
        hdr[n] = dict(dtype=dt, shape=list(a.shape), data_offsets=[off, off + len(b)])
        off += len(b)
        blobs.append(b)
        index[n] = fn
    hj = json.dumps(hdr).encode()
    hj += b" " * ((8 - len(hj) % 8) % 8)
    with open(os.path.join(out, fn), "wb") as f:
        f.write(struct.pack("<Q", len(hj)))
        f.write(hj)
        for b in blobs:
            f.write(b)
json.dump({"weight_map": index}, open(os.path.join(out, "model.safetensors.index.json"), "w"))
print(f"tiny glm5_next: {len(T)} tensors in 2 shards -> {out}")
