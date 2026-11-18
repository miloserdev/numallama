"""
Export a Hugging Face Llama 3.x checkpoint (model.safetensors + config.json) to the
llama3.c int8 (version 2) format, using only numpy (no torch / safetensors needed).

    python3 export_hf_q8.py /path/to/Llama-3.2-1B-Instruct llama32_1b_instruct_q80.bin

Header extension (inside the 256-byte v2 header, after group_size, all little-endian):
    float rope_theta, float rope_factor, float low_freq_factor, float high_freq_factor,
    int original_max_position_embeddings          (factor == 0 means "no llama3 rope scaling")
    int arch (0 = llama3, 1 = qwen2), float norm_eps, int qkv_bias,
    int special_start (first special token id), int bos_id, int eos_id
    int q4_mask: bit k set = matrix kind k (0 wq, 1 wk, 2 wv, 3 wo, 4 w1, 5 w2, 6 w3) is stored as Q4
    int head_dim, n_experts, n_active, moe_hidden, qk_norm, norm_topk   (Qwen3 / Qwen3-MoE; 0 = n/a)
    int q4_format: 1 = asymmetric groups of 32 with fp16 scale + min (0 / absent: the old symmetric Q4)
fp32 section: attention norms, ffn norms, final norm, [qkv biases], [q_norm (L, head_dim), k_norm (L, head_dim)]
MoE layers: router wg (L, n_experts, dim) in Q8 after wo; w1/w3 (L, n_experts * moe_hidden, dim) and
w2 (L, n_experts * dim, moe_hidden), every expert a contiguous block of rows.
If qkv_bias, the fp32 section continues after the final norm with bq (L, dim), bk (L, kv_dim), bv (L, kv_dim).
Q8 tensors: int8 values, then fp32 scales per `gs` values. Q4 tensors (q4_format 1): groups of 32
values u in 0..15 packed in 16 bytes (byte k = u[k] in the low nibble, u[k+16] in the high nibble), then
per group [s, m] as two fp16 (value = s * u + m). Embedding / classifier are always Q8.

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
    """asymmetric 4-bit per group of 32: value = s * u + m, u in 0..15, s and m in fp16.
    For every group, candidate grids are tried (scale (15 + d) / (max - min), d in -1..1 step 0.1)
    and for each the integers are fixed and (s, m) solved by weighted least squares; the best fit
    wins. Weights: importance of the input channel (mean x^2 on calibration text) if given,
    otherwise |w| + rms(group) (larger weights matter more, as in llama.cpp's k-quants).
    Returns packed nibbles (byte k = u[k] | u[k+16] << 4) and per group [s, m] as fp16 pairs."""
    rows, cols = w.reshape(-1, w.shape[-1]).shape
    g = w.astype(np.float32).reshape(-1, gs)
    if imp is None:
        wt = np.abs(g) + np.sqrt((g * g).mean(axis=1, keepdims=True))
    else:   # (cols,) or, for stacked MoE experts, (n_experts, cols): one row per block of rows
        imp = np.asarray(imp, np.float32).reshape(-1, cols)
        imp = imp / max(float(imp.mean()), 1e-30)
        per = rows // imp.shape[0]
        wt = np.repeat(imp, per, axis=0).reshape(rows, cols // gs, gs).reshape(-1, gs)
        wt = wt * (np.abs(g) + np.sqrt((g * g).mean(axis=1, keepdims=True)))
    wt = wt + 1e-12
    gmin, gmax = g.min(axis=1), g.max(axis=1)
    rng = np.maximum(gmax - gmin, 1e-30)
    sw = wt.sum(axis=1)
    sx = (wt * g).sum(axis=1)
    best_err = np.full(len(g), np.inf, np.float32)
    best_s = rng / 15.0
    best_m = gmin.copy()
    for d in np.arange(-1.0, 1.0001, 0.1):
        iscale = (15.0 + d) / rng
        u = np.clip(np.round((g - gmin[:, None]) * iscale[:, None]), 0, 15)
        sl = (wt * u).sum(axis=1)
        sll = (wt * u * u).sum(axis=1)
        slx = (wt * u * g).sum(axis=1)
        D = sw * sll - sl * sl
        ok = D > 1e-30
        Ds = np.where(ok, D, 1.0)
        s_ = np.where(ok, (sw * slx - sl * sx) / Ds, rng / 15.0)
        m_ = np.where(ok, (sll * sx - sl * slx) / Ds, gmin)
        # the integers for the fitted (s, m), then the error
        inv = np.where(np.abs(s_) > 1e-30, 1.0 / np.where(np.abs(s_) > 1e-30, s_, 1.0), 0.0)
        u2 = np.clip(np.round((g - m_[:, None]) * inv[:, None]), 0, 15)
        err = (wt * (u2 * s_[:, None] + m_[:, None] - g) ** 2).sum(axis=1)
        better = err < best_err
        best_err[better] = err[better]
        best_s[better] = s_[better]
        best_m[better] = m_[better]
    s16 = best_s.astype(np.float16)
    m16 = best_m.astype(np.float16)
    sf, mf = s16.astype(np.float32), m16.astype(np.float32)   # integers for the stored (fp16) values
    inv = np.where(np.abs(sf) > 1e-30, 1.0 / np.where(np.abs(sf) > 1e-30, sf, 1.0), 0.0)
    u = np.clip(np.round((g - mf[:, None]) * inv[:, None]), 0, 15).astype(np.uint8)
    packed = (u[:, :16] | (u[:, 16:] << 4)).astype(np.uint8)
    err = np.abs(u.astype(np.float32) * sf[:, None] + mf[:, None] - g).max()
    sm = np.stack([s16, m16], axis=1)   # (groups, 2) fp16
    return packed, sm, err


INPUT_OF = {"wq": "att_in", "wk": "att_in", "wv": "att_in", "wo": "wo_in", "w1": "ffn_in", "w3": "ffn_in", "w2": "w2_in"}
_JOB = {}


def load_imatrix(path):
    """per-layer input-channel importance (mean x^2 on calibration text): a .imx file written by
    `runq <q8 model> -m imatrix -i data/calibration.txt -o stats.imx`, or an .npz from calibrate.py.
    MoE: ffn_in / w2_in are (n_experts, channels), one row per expert; experts that saw no
    calibration token get the mean over the other experts of the layer."""
    if not path.endswith(".imx"):
        z = np.load(path)
        return {k: z[k] for k in z.files}
    raw = open(path, "rb").read()
    magic, L, dim, q_dim, H, E, mh = struct.unpack_from("<7i", raw, 0)
    assert magic == 0x31584D4E, "not an imatrix file"
    off = 7 * 4 + 8
    EE = max(E, 1)
    def take(n):
        nonlocal off
        a = np.frombuffer(raw, np.float32, n, off).copy()
        off += 4 * n
        return a
    att = take(L * dim).reshape(L, dim)
    wo = take(L * q_dim).reshape(L, q_dim)
    ffn = take(L * EE * dim).reshape(L, EE, dim)
    w2 = take(L * EE * H).reshape(L, EE, H)
    out = {}
    for l in range(L):
        out[f"att_in.{l}.sq"] = att[l]
        out[f"wo_in.{l}.sq"] = wo[l]
        if E > 0:
            cnt = np.frombuffer(raw, np.int64, L * E, off).reshape(L, E)[l]
            for a in (ffn, w2):
                seen = cnt > 0
                if seen.any() and not seen.all():
                    a[l][~seen] = a[l][seen].mean(axis=0)
            out[f"ffn_in.{l}.sq"] = ffn[l]
            out[f"w2_in.{l}.sq"] = w2[l]
        else:
            out[f"ffn_in.{l}.sq"] = ffn[l, 0]
            out[f"w2_in.{l}.sq"] = w2[l, 0]
    return out


def _quant_job(key):
    kind, l = key
    w = _JOB["get"][kind](l)
    if kind in _JOB["q4"] and kind != "wg":
        imp = _JOB["imp"].get(f"{INPUT_OF[kind]}.{l}.sq") if _JOB["imp"] else None
        q, sc, err = quantize_q4(w, imp=imp)
    else:
        q, sc, err = quantize_q80(w, _JOB["gs"])
    return q.tobytes() + sc.tobytes(), float(err)


KINDS_HYB = KINDS + ["li", "lo"]   # + linear-attention input (qkv|z|b|a) and output projections


def _hyb_job(key):
    kind, l = key
    w = _JOB["get"][kind](l)
    if kind in _JOB["q4"]:
        imp = _JOB["imp"].get(f"{_JOB['inp'][kind]}.{l}.sq") if _JOB["imp"] else None
        if imp is not None:   # statistics rows can be wider (wo / out_proj share one width)
            imp = np.asarray(imp)[: w.shape[-1]]
        q, sc, err = quantize_q4(w, imp=imp)
    else:
        q, sc, err = quantize_q80(w, _JOB["gs"])
    return q.tobytes() + sc.tobytes(), float(err)


def export_qwen35(cfg_all, T, out_path, q4_kinds, stats, gs):
    """Qwen3.5 / Qwen3.8 dense text model (model_type qwen3_5), arch 3. Hybrid layers: Gated DeltaNet
    linear attention, every `full_attention_interval`-th layer gated full attention with partial RoPE.
    Layout after the 256-byte header (which adds int hybrid[8] = full_interval, linear k heads,
    linear v heads, k head dim, v head dim, conv kernel, attention output gate, rotary dims):
    fp32: att norms (L, dim), ffn norms (L, dim), final norm, q_norm / k_norm (Lf, head_dim) of the full
    layers, then for the linear layers conv (Ll, conv_dim, K), dt_bias (Ll, nv), A = -exp(A_log)
    (Ll, nv), gated-norm weight (Ll, dv). All zero-centered RMSNorm weights are stored as 1 + w.
    Quantized: embedding; full layers wq (q rows), wgate, wk, wv, wo; linear layers w_in = [qkv | z | b | a]
    rows, w_out; every layer w1, w2, w3; classifier. Vision tower and MTP head are not exported."""
    cfg = cfg_all.get("text_config", cfg_all)
    P = "model.language_model." if "model.language_model.embed_tokens.weight" in T else "model."
    dim, hidden, L = cfg["hidden_size"], cfg["intermediate_size"], cfg["num_hidden_layers"]
    nh, nkv, hd = cfg["num_attention_heads"], cfg["num_key_value_heads"], cfg["head_dim"]
    vocab, eps = cfg["vocab_size"], float(cfg["rms_norm_eps"])
    rp = cfg.get("rope_parameters", {})
    theta = float(rp.get("rope_theta", cfg.get("rope_theta", 10000.0)))
    rot = int(hd * rp.get("partial_rotary_factor", cfg.get("partial_rotary_factor", 1.0)))
    nk, nv = cfg["linear_num_key_heads"], cfg["linear_num_value_heads"]
    dk, dv, K = cfg["linear_key_head_dim"], cfg["linear_value_head_dim"], cfg["linear_conv_kernel_dim"]
    types = cfg["layer_types"]
    interval = cfg.get("full_attention_interval", 4)
    assert all((t == "full_attention") == ((l + 1) % interval == 0) for l, t in enumerate(types)), "irregular layer types"
    full = [l for l in range(L) if types[l] == "full_attention"]
    lin = [l for l in range(L) if types[l] != "full_attention"]
    gate = int(cfg.get("attn_output_gate", True))
    seq_len = min(int(cfg.get("max_position_embeddings", 262144)), 262144)
    g = lambda n: T[P + n].astype(np.float32)
    lay = lambda l, n: T[f"{P}layers.{l}.{n}"].astype(np.float32)
    eos = 248046 if vocab > 200000 else cfg.get("eos_token_id", 0)
    bos = cfg.get("bos_token_id") or 248044
    special_start = 248044
    q4_mask = sum(1 << KINDS_HYB.index(k) for k in q4_kinds)

    out = open(out_path, "wb")
    out.write(struct.pack("<Ii", 0x616B3432, 2))
    out.write(struct.pack("<7i", dim, hidden, L, nh, nkv, vocab, seq_len))
    out.write(struct.pack("<B", 0))
    out.write(struct.pack("<i", gs))
    out.write(struct.pack("<ffffi", theta, 0.0, 0.0, 0.0, 0))
    out.write(struct.pack("<ifiiii", 3, eps, 0, special_start, bos, eos))
    out.write(struct.pack("<i", q4_mask))
    out.write(struct.pack("<6i", hd, 0, 0, 0, 1, 0))
    out.write(struct.pack("<i", 1))   # q4_format
    out.write(struct.pack("<8i", interval, nk, nv, dk, dv, K, gate, rot))
    out.write(b"\0" * (256 - out.tell()))

    for l in range(L):
        out.write((1.0 + lay(l, "input_layernorm.weight")).tobytes())
    for l in range(L):
        out.write((1.0 + lay(l, "post_attention_layernorm.weight")).tobytes())
    out.write((1.0 + g("norm.weight")).tobytes())
    for l in full:
        out.write((1.0 + lay(l, "self_attn.q_norm.weight")).tobytes())
    for l in full:
        out.write((1.0 + lay(l, "self_attn.k_norm.weight")).tobytes())
    conv_dim = 2 * nk * dk + nv * dv
    for l in lin:
        out.write(lay(l, "linear_attn.conv1d.weight").reshape(conv_dim, K).tobytes())
    for l in lin:
        out.write(lay(l, "linear_attn.dt_bias").tobytes())
    for l in lin:
        out.write((-np.exp(lay(l, "linear_attn.A_log"))).astype(np.float32).tobytes())
    for l in lin:
        out.write(lay(l, "linear_attn.norm.weight").tobytes())

    maxerr = 0.0
    q, sc, err = quantize_q80(g("embed_tokens.weight"), gs)
    out.write(q.tobytes()); out.write(sc.tobytes())

    def qrows(l):
        return lay(l, "self_attn.q_proj.weight").reshape(nh, 2, hd, dim)
    getters = {
        "wq": lambda i: qrows(full[i])[:, 0].reshape(nh * hd, dim),
        "wg": lambda i: qrows(full[i])[:, 1].reshape(nh * hd, dim),
        "wk": lambda i: lay(full[i], "self_attn.k_proj.weight"),
        "wv": lambda i: lay(full[i], "self_attn.v_proj.weight"),
        "wo": lambda i: lay(full[i], "self_attn.o_proj.weight"),
        "li": lambda i: np.concatenate([lay(lin[i], "linear_attn." + n + ".weight") for n in ("in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a")], 0),
        "lo": lambda i: lay(lin[i], "linear_attn.out_proj.weight"),
        "w1": lambda l: lay(l, "mlp.gate_proj.weight"),
        "w2": lambda l: lay(l, "mlp.down_proj.weight"),
        "w3": lambda l: lay(l, "mlp.up_proj.weight"),
    }
    # importance: inputs of each kind; layer index = real layer
    inp = {"wq": "att_in", "wg": "att_in", "wk": "att_in", "wv": "att_in", "li": "att_in",
           "wo": "wo_in", "lo": "wo_in", "w1": "ffn_in", "w3": "ffn_in", "w2": "w2_in"}
    imp = {}
    if stats is not None:
        for l in range(L):
            for key in ("att_in", "wo_in", "ffn_in", "w2_in"):
                imp[f"{key}.{l}.sq"] = stats[f"{key}.{l}.sq"]
    real = {"wq": full, "wg": full, "wk": full, "wv": full, "wo": full, "li": lin, "lo": lin}
    q4set = set(q4_kinds) | ({"wg"} if "wq" in q4_kinds else set())
    def getter_real(kind):
        f_ = getters[kind]
        if kind in real:
            idx = {l: i for i, l in enumerate(real[kind])}
            return lambda l: f_(idx[l])
        return f_
    _JOB.update(get={k: getter_real(k) for k in getters}, q4=q4set, gs=gs, imp=imp if stats is not None else None, inp=inp)
    keys = [(k, l) for k in ("wq", "wg", "wk", "wv", "wo") for l in full]
    keys += [(k, l) for k in ("li", "lo") for l in lin]
    keys += [(k, l) for k in ("w1", "w2", "w3") for l in range(L)]
    import multiprocessing as mp
    procs = int(os.environ.get("EXPORT_PROCS", "64"))
    with mp.get_context("fork").Pool(min(len(keys), os.cpu_count() or 1, procs)) as pool:
        for blob, err in pool.imap(_hyb_job, keys, chunksize=1):
            out.write(blob)
            maxerr = max(maxerr, err)
    q, sc, err = quantize_q80(T["lm_head.weight"].astype(np.float32), gs)
    out.write(q.tobytes()); out.write(sc.tobytes())
    out.close()
    print(f"qwen3_5: dim={dim} hidden={hidden} layers={L} (full {len(full)}, linear {len(lin)}) heads={nh}/{nkv}x{hd} "
          f"rot={rot} linear={nk}k/{nv}v x {dk}/{dv} conv={K} vocab={vocab} q4={sorted(q4set)}")
    print(f"max quantization error {maxerr:.5f}; wrote {os.path.getsize(out_path) / 1e9:.2f} GB")


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
            q4_kinds = [k for k in KINDS if k != "wv"]   # wv (small, sensitive) stays Q8
            del args[i]
    model_dir, out_path = args[0], args[1]
    gs = int(args[2]) if len(args) > 2 else 64
    cfg = json.load(open(os.path.join(model_dir, "config.json")))
    files = sorted(f for f in os.listdir(model_dir) if f.endswith(".safetensors"))
    T = {}
    for f in files:
        T.update(load_safetensors(os.path.join(model_dir, f)))
    if cfg.get("model_type") == "qwen3_5":
        if "--q4" in sys.argv and q4_kinds == [k for k in KINDS if k != "wv"]:
            q4_kinds = [k for k in KINDS_HYB if k != "wv"]
        for k in q4_kinds:
            assert k in KINDS_HYB, f"unknown matrix kind {k}"
        export_qwen35(cfg, T, out_path, q4_kinds, load_imatrix(imatrix_path) if imatrix_path else None, gs)
        return
    for k in q4_kinds:
        assert k in KINDS, f"unknown matrix kind {k}"
    q4_mask = sum(1 << KINDS.index(k) for k in q4_kinds)

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
    out.write(struct.pack("<i", 1))   # q4_format
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
    procs = int(os.environ.get("EXPORT_PROCS", "64"))   # Q4 of stacked MoE experts: ~10 GB per process
    with mp.get_context("fork").Pool(min(len(keys), os.cpu_count() or 1, procs)) as pool:
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
