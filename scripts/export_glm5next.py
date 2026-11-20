"""
Export GLM-5.3-Flash (model_type glm5_next, Glm5NextForConditionalGeneration) to the numallama format,
arch 4. Text model only (the vision tower and the MTP layer are not exported).

The checkpoint is ~330 GB of FP8 / BF16 safetensors, more than the disk usually holds next to the
output, so the export streams: the output file is laid out up front (every tensor at a fixed
offset, file preallocated sparse) and each shard is converted on its own as soon as it is
downloaded, written in place with pwrite, and can then be deleted. Converted tensors are logged
(<out>.log), so an interrupted export resumes where it stopped.

    python3 export_glm5next.py plan  <hf dir> <out.bin> [--q4 kinds]   # header + layout (<out>.plan.json)
    python3 export_glm5next.py shard <out.bin> <shard.safetensors> [--ref dir]
    python3 export_glm5next.py watch <out.bin> <shard dir> [--ref dir] # convert + delete shards as they arrive
    python3 export_glm5next.py check <out.bin>                         # every tensor written?

--ref dir: tensors of layers 0..3, embeddings, final norm and lm_head are also copied there unchanged
(a 4-layer model with every block type, for comparing the engine against the reference implementation).

Header (256 bytes): the usual v2 fields (arch 4, n_kv_heads 1 and head_dim = kv_lora_rank: the KV cache
holds the MLA latent), then after hybrid[8] (zeros) the GLM block of 24 int32 (floats where noted):
  0 first_k_dense  1 q_lora_rank  2 kv_lora_rank  3 qk_head_dim  4 v_head_dim  5 linear heads
  6 linear head dim  7 conv kernel  8 hc_mult  9 sinkhorn iterations  10 hc_eps (f)  11 routed scale (f)
  12 swiglu limit (f)  13 forget gate lower bound (f)  14 indexer heads  15 indexer head dim
  16 indexer top-k  17 indexer k-pool  18/19 attention-layer bitmask (layers 0..31 / 32..63)
  20 Q4 mask over KINDS  21 indexer always selects the tail pool  22, 23 reserved
Layout after the header (L layers, LA attention layers, LL linear layers, LM MoE layers):
fp32: input norms (L, D), post-attention norms (L, D), final norm (D),
      mHC fn (L, 2, 24, 4D) [attn, ffn], base (L, 2, 24), scale (L, 2, 4) (3 used),
      linear: conv (LL, 3 * H * dh, K) [q | k | v channels], dt_bias (LL, H * dh), exp(A_log) (LL, H),
              output norm (LL, dh),
      attention: q_a norm (LA, q_lora), kv_a norm (LA, kv_lora), indexer k_norm weight (LA, ih),
              bias (LA, ih), k-pool position embedding (LA, kpool, ih),
      MoE router bias (LM, E)
quantized (Q8 group GS / Q4 asymmetric groups of 32, per tensor: values then scales), in this order:
  embedding (V, D)
  lin_in  (LL, 3 H dh + 2 dh + H, D)  rows [q | k | v | f_a | g_a | b]
  f_b     (LL, H dh, dh)      g_b (LL, H dh, dh)      lin_out (LL, D, H dh)
  mla_in  (LA, q_lora + kv_lora + 2 ih + ihh, D)  rows [q_a | kv_a | indexer wk | indexer weights_proj | k-pool gate]
  q_b     (LA, nh qk + ihh ih, q_lora)  rows [q_b | indexer wq_b]
  kv_bk   (LA, nh kv_lora, qk)   per head W_k^T (absorbed attention: q' = W_k^T q)
  kv_bv   (LA, nh v, kv_lora)    per head W_v
  o_proj  (LA, D, nh v)
  dense:  w13 (first_k_dense, 2 F, D) rows [gate | up], w2 (first_k_dense, D, F)
  MoE:    router (LM, E, D) Q8, shared w13 (LM, 2 mh, D), shared w2 (LM, D, mh),
          experts w1 (LM, E mh, D), w3 (LM, E mh, D), w2 (LM, E D, mh)
  lm_head (V, D)
"""
import fcntl
import json
import os
import re
import struct
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from export_hf_q8 import quantize_q4, quantize_q80  # noqa: E402

KINDS = ["lin_in", "f_b", "g_b", "lin_out", "mla_in", "q_b", "kv_bk", "kv_bv", "o_proj",
         "w13", "w2", "sh13", "sh2", "ex1", "ex3", "ex2", "router", "embed", "lm_head"]
DEFAULT_Q4 = ["ex1", "ex3", "ex2"]
GS = 64

# ---------------------------------------------------------------------------- FP8 / BF16 decoding


def _fp8_table():
    t = np.zeros(256, np.float32)
    for b in range(256):
        s, e, m = b >> 7, (b >> 3) & 15, b & 7
        if e == 15 and m == 7:
            v = np.nan
        elif e == 0:
            v = (m / 8.0) * 2.0 ** -6
        else:
            v = (1.0 + m / 8.0) * 2.0 ** (e - 7)
        t[b] = -v if s else v
    return t


FP8 = _fp8_table()


class Shard:
    def __init__(self, path):
        with open(path, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            self.header = json.loads(f.read(n))
        self.base = 8 + n
        self.raw = np.memmap(path, dtype=np.uint8, mode="r")
        self.header.pop("__metadata__", None)

    def names(self):
        return list(self.header)

    def bytes(self, name):
        info = self.header[name]
        a, b = info["data_offsets"]
        return self.raw[self.base + a: self.base + b]

    def get(self, name):
        info = self.header[name]
        buf, dt, shape = self.bytes(name), info["dtype"], info["shape"]
        if dt == "BF16":
            arr = (buf.view(np.uint16).astype(np.uint32) << 16).view(np.float32)
        elif dt == "F32":
            arr = np.array(buf.view(np.float32))
        elif dt == "F16":
            arr = buf.view(np.float16).astype(np.float32)
        elif dt == "F8_E4M3":
            arr = FP8[np.asarray(buf)]
            sname = name + "_scale_inv"
            arr = arr.reshape(shape)
            if sname in self.header:
                sc = self.get(sname)
                r, c = shape
                full = np.repeat(np.repeat(sc, 128, 0)[:r], 128, 1)[:, :c]
                arr = arr * full
            return arr.astype(np.float32)
        else:
            raise ValueError(f"unsupported dtype {dt} for {name}")
        return arr.reshape(shape)


# ---------------------------------------------------------------------------- layout


def model_dims(cfg_all):
    c = cfg_all.get("text_config", cfg_all)
    la = c["linear_attn_config"]
    L = c["num_hidden_layers"]
    types = c["layer_types"]
    attn = [l for l in range(L) if types[l] != "linear_attention"]
    lin = [l for l in range(L) if types[l] == "linear_attention"]
    mlp = c["mlp_layer_types"]
    dense = [l for l in range(L) if mlp[l] == "dense"]
    moe = [l for l in range(L) if mlp[l] == "sparse"]
    assert dense == list(range(len(dense))) and len(dense) == c["first_k_dense_replace"]
    assert c["qk_rope_head_dim"] == 0 and c["n_group"] == 1 and c["topk_group"] == 1
    assert c["scoring_func"] == "sigmoid" and c["n_shared_experts"] == 1 and c["num_key_value_heads"] == c["num_attention_heads"]
    assert all(t == "full" for t in c["indexer_types"])
    return dict(
        D=c["hidden_size"], F=c["intermediate_size"], L=L, V=c["vocab_size"], eps=float(c["rms_norm_eps"]),
        nh=c["num_attention_heads"], ql=c["q_lora_rank"], kvl=c["kv_lora_rank"], qk=c["qk_nope_head_dim"],
        vh=c["v_head_dim"], H=la["num_heads"], dh=la["head_dim"], K=la["short_conv_kernel_size"],
        lower=float(la["gate_lower_bound"]), hc=c["hc_mult"], sk=c["hc_sinkhorn_iters"], hc_eps=float(c["hc_eps"]),
        E=c["n_routed_experts"], A=c["num_experts_per_tok"], mh=c["moe_intermediate_size"],
        scale=float(c["routed_scaling_factor"]), limit=float(c["swiglu_limit"]),
        ihh=c["index_n_heads"], ih=c["index_head_dim"], itopk=c["index_topk"], kpool=c["index_kpool"],
        tail=int(c["index_kpool_always_select_tail"]), norm_topk=int(c["norm_topk_prob"]),
        attn=attn, lin=lin, dense=dense, moe=moe, max_pos=c["max_position_embeddings"])


class Layout:
    """byte offset of every tensor region in the output file"""

    def __init__(self, d, q4):
        self.d, self.q4 = d, set(q4)
        self.regions = {}   # key -> dict(off, ...)
        self.pos = 256
        D, L, LA, LL, LM = d["D"], d["L"], len(d["attn"]), len(d["lin"]), len(d["moe"])
        H, dh, K, nh = d["H"], d["dh"], d["K"], d["nh"]
        f = self.f32
        f("in_norm", L, D); f("post_norm", L, D); f("final_norm", 1, D)
        f("hc_fn", L * 2, 24 * d["hc"] * D); f("hc_base", L * 2, 24); f("hc_scale", L * 2, 4)
        f("conv", LL, 3 * H * dh * K); f("dt_bias", LL, H * dh); f("A", LL, H); f("o_norm", LL, dh)
        f("qa_norm", LA, d["ql"]); f("kva_norm", LA, d["kvl"]); f("ik_norm_w", LA, d["ih"])
        f("ik_norm_b", LA, d["ih"]); f("ape", LA, d["kpool"] * d["ih"]); f("router_bias", LM, d["E"])
        q = self.qt
        q("embed", 1, d["V"], D)
        q("lin_in", LL, 3 * H * dh + 2 * dh + H, D); q("f_b", LL, H * dh, dh); q("g_b", LL, H * dh, dh)
        q("lin_out", LL, D, H * dh)
        q("mla_in", LA, d["ql"] + d["kvl"] + 2 * d["ih"] + d["ihh"], D)
        q("q_b", LA, nh * d["qk"] + d["ihh"] * d["ih"], d["ql"])
        q("kv_bk", LA, nh * d["kvl"], d["qk"]); q("kv_bv", LA, nh * d["vh"], d["kvl"])
        q("o_proj", LA, D, nh * d["vh"])
        nd = len(d["dense"])
        q("w13", nd, 2 * d["F"], D); q("w2", nd, D, d["F"])
        q("router", LM, d["E"], D); q("sh13", LM, 2 * d["mh"], D); q("sh2", LM, D, d["mh"])
        q("ex1", LM, d["E"] * d["mh"], D); q("ex3", LM, d["E"] * d["mh"], D); q("ex2", LM, d["E"] * D, d["mh"])
        q("lm_head", 1, d["V"], D)
        self.size = self.pos

    def f32(self, key, n, per):
        self.regions[key] = dict(kind="f32", off=self.pos, n=n, per=per)
        self.pos += 4 * n * per

    def qt(self, key, n, rows, cols):
        bits = 4 if key in self.q4 else 8
        assert rows % 4 == 0 and cols % (32 if bits == 4 else GS) == 0, (key, rows, cols)
        qb = rows * cols // (2 if bits == 4 else 1)
        sb = rows * (cols // (32 if bits == 4 else GS)) * 4
        self.regions[key] = dict(kind="q", off=self.pos, n=n, rows=rows, cols=cols, bits=bits, qb=qb, sb=sb)
        self.pos += n * (qb + sb)

    def f32_at(self, key, i, start=0):
        r = self.regions[key]
        return r["off"] + 4 * (i * r["per"] + start)

    def rows_at(self, key, i, r0):
        """(values offset, scales offset) of row r0 of tensor i of a quantized array"""
        r = self.regions[key]
        base = r["off"] + i * (r["qb"] + r["sb"])
        bpr = r["cols"] // (2 if r["bits"] == 4 else 1)
        spr = r["cols"] // (32 if r["bits"] == 4 else GS) * 4
        return base + r0 * bpr, base + r["qb"] + r0 * spr, r


def write_header(f, d, q4):
    seq_len = min(int(d["max_pos"]), 131072)
    q4_mask = sum(1 << KINDS.index(k) for k in q4)
    amask = sum(1 << l for l in d["attn"])
    f.write(struct.pack("<Ii", 0x616B3432, 2))
    f.write(struct.pack("<7i", d["D"], d["F"], d["L"], d["nh"], 1, d["V"], seq_len))
    f.write(struct.pack("<B", 0))
    f.write(struct.pack("<i", GS))
    f.write(struct.pack("<ffffi", 0.0, 0.0, 0.0, 0.0, 0))
    f.write(struct.pack("<ifiiii", 4, d["eps"], 0, 154820, 154822, 154820))
    f.write(struct.pack("<i", 0))  # Qwen-style q4 mask: unused
    f.write(struct.pack("<6i", d["kvl"], d["E"], d["A"], d["mh"], 0, d["norm_topk"]))
    f.write(struct.pack("<i", 1))  # q4_format: asymmetric groups of 32
    f.write(struct.pack("<8i", 0, 0, 0, 0, 0, 0, 0, 0))
    f.write(struct.pack("<10i", len(d["dense"]), d["ql"], d["kvl"], d["qk"], d["vh"], d["H"], d["dh"], d["K"], d["hc"], d["sk"]))
    f.write(struct.pack("<4f", d["hc_eps"], d["scale"], d["limit"], d["lower"]))
    f.write(struct.pack("<4i2I4i", d["ihh"], d["ih"], d["itopk"], d["kpool"], amask & 0xFFFFFFFF, amask >> 32, q4_mask, d["tail"], 0, 0))
    assert f.tell() <= 256, f.tell()
    f.write(b"\0" * (256 - f.tell()))


# ---------------------------------------------------------------------------- tensor -> regions

PFX = "model.language_model."


def targets(d):
    """source tensor name -> handler (a function returning a list of writes)"""
    T = {}
    A = {l: i for i, l in enumerate(d["attn"])}
    LI = {l: i for i, l in enumerate(d["lin"])}
    M = {l: i for i, l in enumerate(d["moe"])}
    H, dh, nh = d["H"], d["dh"], d["nh"]
    T[PFX + "embed_tokens.weight"] = ("q", "embed", 0, 0, None)
    T["lm_head.weight"] = ("q", "lm_head", 0, 0, None)
    T[PFX + "norm.weight"] = ("f", "final_norm", 0, 0, None)
    for l in range(d["L"]):
        p = f"{PFX}layers.{l}."
        T[p + "input_layernorm.weight"] = ("f", "in_norm", l, 0, None)
        T[p + "post_attention_layernorm.weight"] = ("f", "post_norm", l, 0, None)
        for j, part in enumerate(("attn", "ffn")):
            T[p + f"hc_{part}_fn"] = ("f", "hc_fn", 2 * l + j, 0, None)
            T[p + f"hc_{part}_base"] = ("f", "hc_base", 2 * l + j, 0, None)
            T[p + f"hc_{part}_scale"] = ("f", "hc_scale", 2 * l + j, 0, None)
        a = p + "self_attn."
        if l in LI:
            i = LI[l]
            for j, n in enumerate("qkv"):
                T[a + f"{n}_proj.weight"] = ("q", "lin_in", i, j * H * dh, None)
                T[a + f"{n}_conv1d.weight"] = ("f", "conv", i, j * H * dh * d["K"], None)
            T[a + "f_a_proj.weight"] = ("q", "lin_in", i, 3 * H * dh, None)
            T[a + "g_a_proj.weight"] = ("q", "lin_in", i, 3 * H * dh + dh, None)
            T[a + "b_proj.weight"] = ("q", "lin_in", i, 3 * H * dh + 2 * dh, None)
            T[a + "f_b_proj.weight"] = ("q", "f_b", i, 0, None)
            T[a + "g_b_proj.weight"] = ("q", "g_b", i, 0, None)
            T[a + "o_proj.weight"] = ("q", "lin_out", i, 0, None)
            T[a + "dt_bias"] = ("f", "dt_bias", i, 0, None)
            T[a + "A_log"] = ("f", "A", i, 0, np.exp)
            T[a + "o_norm.weight"] = ("f", "o_norm", i, 0, None)
        else:
            i = A[l]
            ql, kvl, ih = d["ql"], d["kvl"], d["ih"]
            T[a + "q_a_proj.weight"] = ("q", "mla_in", i, 0, None)
            T[a + "kv_a_proj_with_mqa.weight"] = ("q", "mla_in", i, ql, None)
            T[a + "indexer.wk.weight"] = ("q", "mla_in", i, ql + kvl, None)
            T[a + "indexer.weights_proj.weight"] = ("q", "mla_in", i, ql + kvl + ih, None)
            T[a + "indexer.index_kpool_compress_gate"] = ("q", "mla_in", i, ql + kvl + ih + d["ihh"], None)
            T[a + "q_b_proj.weight"] = ("q", "q_b", i, 0, None)
            T[a + "indexer.wq_b.weight"] = ("q", "q_b", i, nh * d["qk"], None)
            T[a + "kv_b_proj.weight"] = ("kvb", None, i, 0, None)
            T[a + "o_proj.weight"] = ("q", "o_proj", i, 0, None)
            T[a + "q_a_layernorm.weight"] = ("f", "qa_norm", i, 0, None)
            T[a + "kv_a_layernorm.weight"] = ("f", "kva_norm", i, 0, None)
            T[a + "indexer.k_norm.weight"] = ("f", "ik_norm_w", i, 0, None)
            T[a + "indexer.k_norm.bias"] = ("f", "ik_norm_b", i, 0, None)
            T[a + "indexer.index_kpool_compress_ape"] = ("f", "ape", i, 0, None)
        m = p + "mlp."
        if l in d["dense"]:
            T[m + "gate_proj.weight"] = ("q", "w13", l, 0, None)
            T[m + "up_proj.weight"] = ("q", "w13", l, d["F"], None)
            T[m + "down_proj.weight"] = ("q", "w2", l, 0, None)
        else:
            i = M[l]
            T[m + "gate.weight"] = ("q", "router", i, 0, None)
            T[m + "gate.e_score_correction_bias"] = ("f", "router_bias", i, 0, None)
            T[m + "shared_experts.gate_proj.weight"] = ("q", "sh13", i, 0, None)
            T[m + "shared_experts.up_proj.weight"] = ("q", "sh13", i, d["mh"], None)
            T[m + "shared_experts.down_proj.weight"] = ("q", "sh2", i, 0, None)
            for e in range(d["E"]):
                T[m + f"experts.{e}.gate_proj.weight"] = ("q", "ex1", i, e * d["mh"], None)
                T[m + f"experts.{e}.up_proj.weight"] = ("q", "ex3", i, e * d["mh"], None)
                T[m + f"experts.{e}.down_proj.weight"] = ("q", "ex2", i, e * d["D"], None)
    return T


def ref_wanted(name):
    if name.startswith(PFX + "layers."):
        return int(name.split(".")[3]) <= 3
    return name in (PFX + "embed_tokens.weight", PFX + "norm.weight", "lm_head.weight")


# ---------------------------------------------------------------------------- conversion workers

_W = {}


def _pwrite(fd, data, off):
    mv = memoryview(data).cast("B")
    while len(mv):
        n = os.pwrite(fd, mv, off)
        mv, off = mv[n:], off + n


def _write_rows(lay, fd, key, i, r0, w):
    qoff, soff, r = lay.rows_at(key, i, r0)
    assert w.shape[1] == r["cols"] and r0 + w.shape[0] <= r["rows"], (key, w.shape, r0)
    if r["bits"] == 4:
        q, sc, err = quantize_q4(w)
    else:
        q, sc, err = quantize_q80(w, GS)
    _pwrite(fd, q.tobytes(), qoff)
    _pwrite(fd, sc.tobytes(), soff)
    return float(err)


def _job(name):
    sh, lay, fd, T = _W["shard"], _W["lay"], _W["fd"], _W["T"]
    d = lay.d
    kind, key, i, start, fn = T[name]
    w = sh.get(name)
    err = 0.0
    if kind == "f":
        a = w.astype(np.float32).reshape(-1)
        if fn is not None:
            a = fn(a).astype(np.float32)
        r = lay.regions[key]
        assert start + a.size <= r["per"], (name, a.size, r["per"])
        assert key in ("conv", "hc_scale") or a.size == r["per"], (name, a.size, r["per"])
        _pwrite(fd, a.tobytes(), lay.f32_at(key, i, start))
    elif kind == "q":
        w = w.reshape(w.shape[0], -1)
        err = _write_rows(lay, fd, key, i, start, w)
    elif kind == "kvb":   # (nh * (qk + v), kvl): per head [W_k (qk rows) | W_v (v rows)]
        nh, qk, vh, kvl = d["nh"], d["qk"], d["vh"], d["kvl"]
        w = w.reshape(nh, qk + vh, kvl)
        wk_t = np.ascontiguousarray(w[:, :qk, :].transpose(0, 2, 1)).reshape(nh * kvl, qk)
        wv = np.ascontiguousarray(w[:, qk:, :]).reshape(nh * vh, kvl)
        err = max(_write_rows(lay, fd, "kv_bk", i, 0, wk_t), _write_rows(lay, fd, "kv_bv", i, 0, wv))
    return name, err


def write_ref(sh, names, ref_dir):
    """copy tensors unchanged into a safetensors file of their own"""
    os.makedirs(ref_dir, exist_ok=True)
    hdr, off, parts = {}, 0, []
    for n in names:
        info = sh.header[n]
        b = sh.bytes(n)
        hdr[n] = dict(dtype=info["dtype"], shape=info["shape"], data_offsets=[off, off + len(b)])
        off += len(b)
        parts.append(b)
    hj = json.dumps(hdr).encode()
    hj += b" " * ((8 - len(hj) % 8) % 8)
    tag = re.sub(r"\W", "_", os.path.basename(sh.path))
    tmp = os.path.join(ref_dir, tag + ".tmp")
    with open(tmp, "wb") as f:
        f.write(struct.pack("<Q", len(hj)))
        f.write(hj)
        for b in parts:
            f.write(np.asarray(b).tobytes())
    os.rename(tmp, os.path.join(ref_dir, tag))


def load_plan(out_path):
    plan = json.load(open(out_path + ".plan.json"))
    d = plan["dims"]
    return d, Layout(d, plan["q4"])


def done_set(out_path):
    if not os.path.exists(out_path + ".log"):
        return set()
    return set(line.split("\t")[0] for line in open(out_path + ".log") if line.strip())


def convert_shard(out_path, shard_path, ref_dir=None, procs=None):
    import multiprocessing as mp
    d, lay = load_plan(out_path)
    T = targets(d)
    sh = Shard(shard_path)
    sh.path = shard_path
    done = done_set(out_path)
    names = [n for n in sh.names() if n in T and n not in done]
    skipped = [n for n in sh.names() if n not in T and not n.endswith("_scale_inv")
               and "visual" not in n and ".layers.%d." % d["L"] not in n]
    if skipped:
        raise SystemExit(f"{shard_path}: unknown tensors {skipped[:5]}")
    if ref_dir:
        rn = [n for n in sh.names() if ref_wanted(n) or (n.endswith("_scale_inv") and ref_wanted(n[: -len("_scale_inv")]))]
        if rn and not os.path.exists(os.path.join(ref_dir, re.sub(r"\W", "_", os.path.basename(shard_path)))):
            write_ref(sh, rn, ref_dir)
    fd = os.open(out_path, os.O_WRONLY)
    _W.update(shard=sh, lay=lay, fd=fd, T=T)
    procs = procs or int(os.environ.get("EXPORT_PROCS", "48"))
    maxerr = 0.0
    # big tensors first so the pool does not end on one long job
    names.sort(key=lambda n: -int(np.prod(sh.header[n]["shape"])))
    with open(out_path + ".log", "a") as log:
        if names:
            with mp.get_context("fork").Pool(max(1, min(procs, len(names)))) as pool:
                for name, err in pool.imap_unordered(_job, names, chunksize=1):
                    fcntl.flock(log, fcntl.LOCK_EX)
                    log.write(f"{name}\t{err:.5f}\n")
                    log.flush()
                    fcntl.flock(log, fcntl.LOCK_UN)
                    maxerr = max(maxerr, err)
    os.fsync(fd)
    os.close(fd)
    return len(names), maxerr


def cmd_plan(hf_dir, out_path, q4):
    cfg = json.load(open(os.path.join(hf_dir, "config.json")))
    d = model_dims(cfg)
    lay = Layout(d, q4)
    with open(out_path, "wb") as f:
        write_header(f, d, q4)
        f.truncate(lay.size)
    json.dump(dict(dims=d, q4=sorted(q4)), open(out_path + ".plan.json", "w"))
    # every expected tensor (for `check`)
    idx = json.load(open(os.path.join(hf_dir, "model.safetensors.index.json")))["weight_map"]
    T = targets(d)
    missing = [n for n in T if n not in idx]
    if missing:
        raise SystemExit(f"tensors missing from the checkpoint: {missing[:5]}")
    print(f"planned {out_path}: {lay.size / 1e9:.2f} GB, {len(T)} tensors, q4={sorted(q4)}")
    for k, r in lay.regions.items():
        sz = 4 * r["n"] * r["per"] if r["kind"] == "f32" else r["n"] * (r["qb"] + r["sb"])
        print(f"  {k:12s} {r['kind']}{'' if r['kind'] == 'f32' else r['bits']:<3} n={r['n']:<3} {sz / 1e9:8.3f} GB")


def cmd_check(out_path):
    d, lay = load_plan(out_path)
    T = targets(d)
    done = done_set(out_path)
    miss = [n for n in T if n not in done]
    print(f"{len(T) - len(miss)} / {len(T)} tensors written")
    if miss:
        print("missing e.g.", miss[:10])
        return 1
    return 0


def cmd_watch(out_path, shard_dir, ref_dir):
    import time
    while True:
        ready = sorted(f for f in os.listdir(shard_dir) if f.endswith(".safetensors"))
        if not ready:
            if all(os.path.exists(os.path.join(shard_dir, "model-%05d-of-00062.safetensors.done" % i)) for i in range(1, 63)):
                print("all shards converted", flush=True)
                return
            time.sleep(30)
            continue
        f = ready[0]
        p = os.path.join(shard_dir, f)
        t0 = time.time()
        n, err = convert_shard(out_path, p, ref_dir)
        open(p + ".done", "w").close()
        os.remove(p)
        print(f"[{time.strftime('%d %H:%M:%S')}] {f}: {n} tensors, max err {err:.4f}, {time.time() - t0:.0f} s", flush=True)


def main():
    a = sys.argv[1:]
    ref = None
    if "--ref" in a:
        i = a.index("--ref")
        ref = a[i + 1]
        del a[i: i + 2]
    q4 = DEFAULT_Q4
    if "--q4" in a:
        i = a.index("--q4")
        q4 = [k for k in a[i + 1].split(",") if k]
        del a[i: i + 2]
        assert all(k in KINDS for k in q4), q4
    if a[0] == "plan":
        cmd_plan(a[1], a[2], q4)
    elif a[0] == "shard":
        n, err = convert_shard(a[1], a[2], ref)
        print(f"{a[2]}: {n} tensors, max err {err:.4f}")
    elif a[0] == "watch":
        cmd_watch(a[1], a[2], ref)
    elif a[0] == "check":
        sys.exit(cmd_check(a[1]))
    else:
        raise SystemExit(__doc__)


if __name__ == "__main__":
    main()
