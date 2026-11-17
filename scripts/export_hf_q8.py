"""
Export a Hugging Face Llama 3.x checkpoint (model.safetensors + config.json) to the
llama3.c int8 (version 2) format, using only numpy (no torch / safetensors needed).

    python3 export_hf_q8.py /ai/models/Llama-3.2-1B-Instruct llama32_1b_instruct_q80.bin

Header extension (inside the 256-byte v2 header, after group_size, all little-endian):
    float rope_theta, float rope_factor, float low_freq_factor, float high_freq_factor,
    int original_max_position_embeddings          (factor == 0 means "no llama3 rope scaling")
    int arch (0 = llama3, 1 = qwen2), float norm_eps, int qkv_bias,
    int special_start (first special token id), int bos_id, int eos_id
    int q4_mask: bit k set = matrix kind k (0 wq, 1 wk, 2 wv, 3 wo, 4 w1, 5 w2, 6 w3) is stored as Q4
    int head_dim, n_experts, n_active, moe_hidden, qk_norm, norm_topk   (Qwen3 / Qwen3-MoE; 0 = n/a)
fp32 section: attention norms, ffn norms, final norm, [qkv biases], [q_norm (L, head_dim), k_norm (L, head_dim)]
MoE layers: router wg (L, n_experts, dim) in Q8 after wo; w1/w3 (L, n_experts * moe_hidden, dim) and
w2 (L, n_experts * dim, moe_hidden), every expert a contiguous block of rows.
If qkv_bias, the fp32 section continues after the final norm with bq (L, dim), bk (L, kv_dim), bv (L, kv_dim).
Q8 tensors: int8 values, then fp32 scales per `gs` values. Q4 tensors: groups of 32 values packed in
16 bytes (byte k = value k in the low nibble, value k+16 in the high nibble, stored as q+8), then fp32
scales per 32 values. Embedding / classifier are always Q8.

    python3 export_hf_q8.py <hf dir> <out.bin> [gs] [--q4 [kinds]]   kinds e.g. wq,wk,wo,w1,w3 (default all)
Supported: LlamaForCausalLM (Llama 3.x), Qwen2ForCausalLM (Qwen2.5), Qwen3ForCausalLM, Qwen3MoeForCausalLM.
"""
import json
import os
import struct
import sys

import numpy as np


def load_safetensors(path):
    """returns {name: np.ndarray(float32)} lazily converted from bf16/f16/f32"""
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n))
    base = 8 + n
    raw = np.memmap(path, dtype=np.uint8, mode="r")
    tensors = {}
    for name, info in header.items():
        if name == "__metadata__":
            continue
        a, b = info["data_offsets"]
        buf = raw[base + a : base + b]
        dt = info["dtype"]
        if dt == "BF16":
            arr = (buf.view(np.uint16).astype(np.uint32) << 16).view(np.float32)
        elif dt == "F16":
            arr = buf.view(np.float16).astype(np.float32)
        elif dt == "F32":
            arr = np.array(buf.view(np.float32))
        else:
            raise ValueError(f"unsupported dtype {dt} for {name}")
        tensors[name] = arr.reshape(info["shape"])
    return tensors


def hf_to_meta_qk(w, n_heads):
    """undo the HF rotate-half permutation of wq/wk (or their biases) so RoPE can rotate adjacent pairs"""
    if w.ndim == 1:
        return w.reshape(n_heads, 2, -1).transpose(0, 2, 1).reshape(-1)
    d1, d2 = w.shape
    return w.reshape(n_heads, 2, d1 // n_heads // 2, d2).transpose(0, 2, 1, 3).reshape(d1, d2)


def quantize_q80(w, gs):
    g = w.astype(np.float32).reshape(-1, gs)
    scale = np.abs(g).max(axis=1) / 127.0
    scale = np.where(scale > 1e-30, scale, 0.0).astype(np.float32)   # all-(near-)zero groups: avoid 1/denormal
    inv = np.where(scale > 0, 1.0 / np.where(scale > 0, scale, 1.0), 0.0).astype(np.float32)
    q = np.round(g * inv[:, None]).astype(np.int8)
    err = np.abs(q.astype(np.float32) * scale[:, None] - g).max()
    return q, scale.astype(np.float32), err


KINDS = ["wq", "wk", "wv", "wo", "w1", "w2", "w3"]


def quantize_q4(w, gs=32, imp=None):
    """symmetric 4-bit per group of 32. The scale is searched to minimize the squared error,
    weighted by the importance of each input channel (mean x^2 on calibration text) if given,
    and then refined by weighted least squares for the chosen integers."""
    rows, cols = w.reshape(-1, w.shape[-1]).shape
    g = w.astype(np.float32).reshape(-1, gs)
    if imp is None:
        wt = np.ones_like(g)
    else:
        wt = np.broadcast_to(np.asarray(imp, np.float32).reshape(1, cols // gs, gs), (rows, cols // gs, gs)).reshape(-1, gs)
        wt = wt + 1e-8 * wt.mean()
    amax_idx = np.abs(g).argmax(axis=1)
    vmax = g[np.arange(len(g)), amax_idx]                 # signed value with the largest magnitude
    vmax = np.where(np.abs(vmax) > 1e-30, vmax, 0.0).astype(np.float32)
    best_err = np.full(len(g), np.inf, np.float32)
    best_d = np.zeros(len(g), np.float32)
    for f in np.linspace(-8.0, -5.5, 21):                 # d = vmax / f, f = -8 maps the extreme to -8 exactly
        d = vmax / f
        inv = np.where(d != 0, 1.0 / np.where(d != 0, d, 1.0), 0.0)
        q = np.clip(np.round(g * inv[:, None]), -8, 7)
        # least-squares refinement of d for these integers
        num = (wt * g * q).sum(axis=1)
        den = (wt * q * q).sum(axis=1)
        d2 = np.where(den > 0, num / np.where(den > 0, den, 1.0), d)
        d2 = np.where(np.abs(d2) > 1e-30, d2, 0.0)
        for dd in (d, d2):
            inv2 = np.where(dd != 0, 1.0 / np.where(dd != 0, dd, 1.0), 0.0)
            q2 = np.clip(np.round(g * inv2[:, None]), -8, 7)
            err = (wt * (q2 * dd[:, None] - g) ** 2).sum(axis=1)
            better = err < best_err
            best_err[better] = err[better]
            best_d[better] = dd[better]
    inv = np.where(best_d != 0, 1.0 / np.where(best_d != 0, best_d, 1.0), 0.0)
    q = (np.clip(np.round(g * inv[:, None]), -8, 7) + 8).astype(np.uint8)   # 0..15
    packed = (q[:, :16] | (q[:, 16:] << 4)).astype(np.uint8)
    err = np.abs((q.astype(np.float32) - 8) * best_d[:, None] - g).max()
    return packed, best_d.astype(np.float32), err


INPUT_OF = {"wq": "att_in", "wk": "att_in", "wv": "att_in", "wo": "wo_in", "w1": "ffn_in", "w3": "ffn_in", "w2": "w2_in"}
_JOB = {}


def load_imatrix(path):
    """per-layer input-channel importance (mean x^2 on calibration text), from calibrate.py"""
    z = np.load(path)
    return {k: z[k] for k in z.files}


def _quant_job(key):
    kind, l = key
    w = _JOB["get"][kind](l)
    if kind in _JOB["q4"] and kind != "wg":
        imp = _JOB["imp"].get(f"{INPUT_OF[kind]}.{l}.sq") if _JOB["imp"] else None
        q, sc, err = quantize_q4(w, imp=imp)
    else:
        q, sc, err = quantize_q80(w, _JOB["gs"])
    return q.tobytes() + sc.tobytes(), float(err)


def main():
    args = [a for a in sys.argv[1:]]
    smooth_alpha = None
    if "--smooth" in args:                         # SmoothQuant alpha, needs --imatrix stats (max |x|)
        i = args.index("--smooth")
        smooth_alpha = float(args[i + 1])
        del args[i:i + 2]
    imatrix_path = None
    if "--imatrix" in args:
        i = args.index("--imatrix")
        imatrix_path = args[i + 1]
        del args[i:i + 2]
    q4_kinds = []
    if "--q4" in args:
        i = args.index("--q4")
        if i + 1 < len(args) and not args[i + 1].startswith("-") and not args[i + 1].isdigit():
            q4_kinds = args[i + 1].split(",")
            del args[i:i + 2]
        else:
            q4_kinds = list(KINDS)
            del args[i]
    for k in q4_kinds:
        assert k in KINDS, f"unknown matrix kind {k}"
    q4_mask = sum(1 << KINDS.index(k) for k in q4_kinds)
    model_dir, out_path = args[0], args[1]
    gs = int(args[2]) if len(args) > 2 else 64
    cfg = json.load(open(os.path.join(model_dir, "config.json")))
    files = sorted(f for f in os.listdir(model_dir) if f.endswith(".safetensors"))
    T = {}
    for f in files:
        T.update(load_safetensors(os.path.join(model_dir, f)))

    dim = cfg["hidden_size"]
    hidden = cfg["intermediate_size"]
    L = cfg["num_hidden_layers"]
    nh = cfg["num_attention_heads"]
    nkv = cfg.get("num_key_value_heads", nh)
    vocab = cfg["vocab_size"]
    seq_len = cfg["max_position_embeddings"]
    head_dim = cfg.get("head_dim") or dim // nh
    mt = cfg.get("model_type", "llama")
    moe = mt == "qwen3_moe"
    n_experts = cfg.get("num_experts", 0) if moe else 0
    n_active = cfg.get("num_experts_per_tok", 0) if moe else 0
    moe_hidden = cfg.get("moe_intermediate_size", 0) if moe else 0
    if moe:
        hidden = moe_hidden # the group size must divide the rows of w2 (= one expert's hidden size)
    qk_norm = int("model.layers.0.self_attn.q_norm.weight" in T)
    emb = T["model.embed_tokens.weight"]
    shared = cfg.get("tie_word_embeddings", False) or "lm_head.weight" not in T
    if not shared and np.array_equal(emb, T["lm_head.weight"]):
        shared = True

    # the group size must divide both dim and hidden_dim (the rows of w2 have hidden_dim elements)
    while dim % gs or hidden % gs:
        gs //= 2
        print(f"BACKOFF: group size -> {gs}")

    theta = float(cfg.get("rope_theta", 10000.0))
    rs = cfg.get("rope_scaling") or {}
    if rs.get("rope_type", rs.get("type")) == "llama3":
        rope = (theta, float(rs["factor"]), float(rs["low_freq_factor"]),
                float(rs["high_freq_factor"]), int(rs["original_max_position_embeddings"]))
    else:
        rope = (theta, 0.0, 0.0, 0.0, 0)

    out = open(out_path, "wb")
    out.write(struct.pack("<Ii", 0x616B3432, 2))
    out.write(struct.pack("<7i", dim, hidden, L, nh, nkv, vocab, seq_len))
    out.write(struct.pack("<B", int(shared)))
    out.write(struct.pack("<i", gs))
    out.write(struct.pack("<ffffi", *rope))
    arch = {"llama": 0, "qwen2": 1, "qwen3": 2, "qwen3_moe": 2}[mt]
    qkv_bias = int(f"model.layers.0.self_attn.q_proj.bias" in T)
    special_start = {0: 128000, 1: 151643, 2: 151643}[arch]
    eos = cfg.get("eos_token_id", 0)
    eos = eos[0] if isinstance(eos, list) else eos
    bos = cfg.get("bos_token_id") or special_start
    out.write(struct.pack("<ifiiii", arch, float(cfg.get("rms_norm_eps", 1e-5)), qkv_bias, special_start, bos, eos))
    out.write(struct.pack("<i", q4_mask))
    out.write(struct.pack("<6i", head_dim, n_experts, n_active, moe_hidden, qk_norm, int(bool(cfg.get("norm_topk_prob", False)))))
    out.write(b"\0" * (256 - out.tell()))

    p = "model.layers.{}."
    # ---- optional SmoothQuant: x' = x / s, W' = W * s (columns), folded into the producers of x
    stats = load_imatrix(imatrix_path) if imatrix_path else None
    hd = dim // nh
    SM = {}
    if smooth_alpha is not None:
        assert stats is not None, "--smooth needs --imatrix stats.npz"
        a = smooth_alpha

        def sfac(xmax, wmax):
            return np.clip(xmax ** a / np.maximum(wmax, 1e-5) ** (1 - a), 1e-2, 1e4).astype(np.float32)

        for l in range(L):
            q = p.format(l)
            colmax = lambda *names: np.max([np.abs(T[q + n].astype(np.float32)).max(0) for n in names], 0)
            SM[("att_in", l)] = sfac(stats[f"att_in.{l}.max"], colmax("self_attn.q_proj.weight", "self_attn.k_proj.weight", "self_attn.v_proj.weight"))
            SM[("ffn_in", l)] = sfac(stats[f"ffn_in.{l}.max"], colmax("mlp.gate_proj.weight", "mlp.up_proj.weight"))
            SM[("w2_in", l)] = sfac(stats[f"w2_in.{l}.max"], colmax("mlp.down_proj.weight"))
            # attention output channels come from (shared, GQA) v channels: one factor per v channel
            xm = stats[f"wo_in.{l}.max"].reshape(nkv, nh // nkv, hd).max(1)
            wm = colmax("self_attn.o_proj.weight").reshape(nkv, nh // nkv, hd).max(1)
            sv = sfac(xm, wm).reshape(-1)                                  # (kv_dim,)
            SM[("v", l)] = sv
            SM[("wo_in", l)] = np.repeat(sv.reshape(nkv, 1, hd), nh // nkv, 1).reshape(-1)
        print(f"SmoothQuant alpha={a}")

    def S(key, l):
        return SM.get((key, l))

    def eff(kind, l):
        """effective (smoothed) weight in HF layout"""
        q = p.format(l)
        name = {"wq": "self_attn.q_proj", "wk": "self_attn.k_proj", "wv": "self_attn.v_proj", "wo": "self_attn.o_proj",
                "w1": "mlp.gate_proj", "w2": "mlp.down_proj", "w3": "mlp.up_proj"}[kind]
        w = T[q + name + ".weight"].astype(np.float32)
        if not SM:
            return w
        cs = S(INPUT_OF[kind], l)
        w = w * cs[None, :]
        if kind == "wv":
            w = w / S("v", l)[:, None]
        if kind == "w3":
            w = w / S("w2_in", l)[:, None]
        return w

    def experts(l, name):
        """all experts of a MoE layer stacked along the rows (expert e = rows block e)"""
        q = p.format(l)
        return np.concatenate([T[f"{q}mlp.experts.{e}.{name}.weight"].astype(np.float32) for e in range(n_experts)], 0)

    def norm(name, key, l):
        n = T[p.format(l) + name].astype(np.float32)
        return n / S(key, l) if SM else n

    for l in range(L):
        out.write(norm("input_layernorm.weight", "att_in", l).tobytes())
    for l in range(L):
        out.write(norm("post_attention_layernorm.weight", "ffn_in", l).tobytes())
    out.write(T["model.norm.weight"].astype(np.float32).tobytes())
    if qkv_bias:
        for l in range(L):
            out.write(hf_to_meta_qk(T[p.format(l) + "self_attn.q_proj.bias"], nh).astype(np.float32).tobytes())
        for l in range(L):
            out.write(hf_to_meta_qk(T[p.format(l) + "self_attn.k_proj.bias"], nkv).astype(np.float32).tobytes())
        for l in range(L):
            bv = T[p.format(l) + "self_attn.v_proj.bias"].astype(np.float32)
            out.write((bv / S("v", l) if SM else bv).tobytes())
    if qk_norm: # per-head norms, permuted inside the head like the q/k rows
        for l in range(L):
            out.write(hf_to_meta_qk(T[p.format(l) + "self_attn.q_norm.weight"].astype(np.float32), 1).tobytes())
        for l in range(L):
            out.write(hf_to_meta_qk(T[p.format(l) + "self_attn.k_norm.weight"].astype(np.float32), 1).tobytes())

    maxerr = 0.0

    def write_q(w, q4=False):
        nonlocal maxerr
        q, sc, err = quantize_q4(w) if q4 else quantize_q80(w, gs)
        maxerr = max(maxerr, float(err))
        out.write(q.tobytes())
        out.write(sc.tobytes())

    write_q(emb)
    getters = [
        lambda l: hf_to_meta_qk(eff("wq", l), nh),
        lambda l: hf_to_meta_qk(eff("wk", l), nkv),
        lambda l: eff("wv", l),
        lambda l: eff("wo", l),
        (lambda l: experts(l, "gate_proj")) if moe else (lambda l: eff("w1", l)),
        (lambda l: experts(l, "down_proj")) if moe else (lambda l: eff("w2", l)),
        (lambda l: experts(l, "up_proj")) if moe else (lambda l: eff("w3", l)),
    ]
    imp = None
    if stats is not None:   # importance of the (smoothed) inputs: E[(x/s)^2]
        imp = {}
        for l in range(L):
            for key in ("att_in", "wo_in", "ffn_in", "w2_in"):
                v = stats[f"{key}.{l}.sq"]
                imp[f"{key}.{l}.sq"] = v / S(key, l) ** 2 if SM else v
    _JOB.update(get=dict(zip(KINDS, getters)), q4=q4_kinds, gs=gs, imp=imp)
    keys = [(kind, l) for kind in KINDS[:4] for l in range(L)]
    if moe:
        _JOB["get"]["wg"] = lambda l: T[p.format(l) + "mlp.gate.weight"]
        keys += [("wg", l) for l in range(L)]
    keys += [(kind, l) for kind in KINDS[4:] for l in range(L)]
    import multiprocessing as mp
    with mp.get_context("fork").Pool(min(len(keys), os.cpu_count() or 1, 64)) as pool:
        for blob, err in pool.imap(_quant_job, keys, chunksize=1):
            out.write(blob)
            maxerr = max(maxerr, err)
    if not shared:
        write_q(T["lm_head.weight"])
    out.close()
    print(f"dim={dim} hidden={hidden} layers={L} heads={nh} kv_heads={nkv} vocab={vocab} "
          f"seq_len={seq_len} shared={shared} gs={gs} rope={rope} arch={mt} qkv_bias={qkv_bias} q4={q4_kinds} "
          f"head_dim={head_dim} experts={n_experts}x{moe_hidden} top{n_active} qk_norm={qk_norm}")
    print(f"max quantization error {maxerr:.5f}; wrote {os.path.getsize(out_path) / 1e9:.2f} GB")


if __name__ == "__main__":
    main()
