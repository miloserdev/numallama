#!/usr/bin/env python3
"""Export the text backbone of thinkingmachines/Inkling-2-Mini to an INK2 container.

Layout: magic "INK2JSON", u64 manifest length, JSON manifest padded to 1 MiB, then tensor
data at absolute offsets. The manifest holds the config and one entry per tensor
(name, dtype f32|q8|q4, shape, offset, bytes). Q8 is int8 data followed by f32 scales per
64 values; Q4 is packed nibbles (byte k = u[k] | u[k+16] << 4) followed by fp16 (scale, min)
per group of 32. Q4 experts are stored as [expert][rows][cols] blocks, each expert self-
contained. Names are the source names with the `model.llm.` prefix removed, so the loader and
forward can use the original layout. Vision and MTP tensors are skipped.

usage: export_inkling.py SOURCE.safetensors OUTPUT.ink2 [--layers 0,1] [--procs 64]
       [--experts q4|q8]  (experts default to q4; q8 keeps the source dequant exact within 1%)
"""
import argparse
import json
import os
import struct
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from export_hf_q8 import quantize_q4, quantize_q80

DATA_START = 1 << 20
DT = {"U8": np.uint8, "F8_E4M3": np.uint8, "F32": np.float32, "BF16": np.uint16}
E2M1 = np.array([0, .5, 1, 1.5, 2, 3, 4, 6, -0., -.5, -1, -1.5, -2, -3, -4, -6], np.float32)
_E4M3 = None


def e4m3():
    global _E4M3
    if _E4M3 is None:
        c = np.arange(256, dtype=np.uint8)
        ex = ((c & 0x7f) >> 3).astype(np.int32)
        ma = (c & 7).astype(np.float32)
        v = np.where(ex == 0, np.ldexp(ma, -9), np.ldexp(1 + ma / 8, ex - 7))
        v = np.where(c & 0x80, -v, v).astype(np.float32)
        v[(c & 0x7f) == 0x7f] = np.nan
        _E4M3 = v
    return _E4M3


class Safetensors:
    def __init__(self, path):
        self.path = path
        with open(path, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            self.header = json.loads(f.read(n))
        self.base = 8 + n

    def read(self, name, index=None):
        e = self.header[name]
        a, b = e["data_offsets"]
        with open(self.path, "rb") as f:
            f.seek(self.base + a)
            raw = np.frombuffer(f.read(b - a), DT[e["dtype"]])
        a = raw.reshape(e["shape"])
        if e["dtype"] == "BF16":
            a = (a.astype(np.uint32) << 16).view(np.float32)
        return a[index] if index is not None else a


def q8_blob(w):
    q, s, _ = quantize_q80(w.astype(np.float32), 64)
    return q.tobytes() + s.astype(np.float32).tobytes()


def q8_size(rows, cols):
    return rows * cols + rows * (cols // 64) * 4


def q4e_size(e, rows, cols):
    return e * (rows * cols // 2 + rows * (cols // 32) * 4)


def q8e_size(e, rows, cols):
    return e * q8_size(rows, cols)


def nvfp4(w, sc, amax, denom):
    out = np.empty((w.shape[0], w.shape[1] * 2), np.float32)
    out[:, 0::2] = E2M1[w & 15]
    out[:, 1::2] = E2M1[w >> 4]
    out.reshape(out.shape[0], -1, 16)[:] *= e4m3()[sc][:, :, None]
    return out * (amax / denom)


def mxfp8(w, sc):
    out = e4m3()[w].copy()
    bs = np.ldexp(np.ones(sc.shape, np.float32), sc.astype(np.int32) - 127)
    out.reshape(out.shape[0], -1, 32)[:] *= bs[:, :, None]
    return out


_W = None


def expert_worker(task):
    src, kind, e, off, stride, quant = task
    st, H, base = _W
    wname = src + "_weight" if not src.endswith("_weight") else src
    e0 = H[wname]
    a, b = e0["data_offsets"]
    with open(st, "rb") as f:
        f.seek(base + a)
        w = np.frombuffer(f.read(b - a), DT[e0["dtype"]]).reshape(e0["shape"])[e]
        sn = wname + "_scale"
        se = H[sn]
        a, b = se["data_offsets"]
        f.seek(base + a)
        sc = np.frombuffer(f.read(b - a), DT[se["dtype"]]).reshape(se["shape"])[e]
    if H[sn]["dtype"] == "F8_E4M3":  # NVFP4 weights are U8 with E4M3 block scales
        an = wname + "_amax"
        dn = wname + "_outer_scale_denominator"
        st2, H2, base2 = _W
        with open(st, "rb") as f:
            ae = H[an]; a, b = ae["data_offsets"]; f.seek(base + a)
            amax = np.frombuffer(f.read(b - a), np.float32).reshape(ae["shape"])[e]
            de = H[dn]; a, b = de["data_offsets"]; f.seek(base + a)
            denom = np.frombuffer(f.read(b - a), np.float32)[0]
        dense = nvfp4(w, sc, amax, denom)
    else:  # MXFP8
        dense = mxfp8(w, sc)
    if quant == "q8":
        return q8_blob(dense)
    q, sm, _ = quantize_q4(dense)
    return q.tobytes() + sm.tobytes()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("source")
    ap.add_argument("output")
    ap.add_argument("--layers", default="")
    ap.add_argument("--procs", type=int, default=min(64, os.cpu_count() or 1))
    ap.add_argument("--skip-embed", action="store_true")
    ap.add_argument("--experts", choices=("q4", "q8"), default="q4")
    args = ap.parse_args()

    cfg = json.load(open(os.path.join(os.path.dirname(args.source), "config.json")))
    tc = cfg.get("text_config", cfg)
    st = Safetensors(args.source)
    H = st.header
    L = tc["num_hidden_layers"]
    dim, inter = tc["hidden_size"], tc["intermediate_size"]
    nh, nkv, hd = tc["num_attention_heads"], tc["num_key_value_heads"], tc["head_dim"]
    E = tc["n_routed_experts"]
    loop = tc["loop_config"]
    local = set(tc.get("local_layer_ids", []))
    want_layers = set(int(x) for x in args.layers.split(",")) if args.layers else set(range(L))
    keep = []
    for name in H:
        if name == "__metadata__" or name.startswith(("model.visual.", "model.vision_tower.", "mtp.")):
            continue
        if not name.startswith("model.llm."):
            continue
        nm = name[len("model.llm."):]
        if nm.startswith("layers.") and int(nm.split(".")[1]) not in want_layers:
            continue
        if nm.startswith("embed.") and args.skip_embed:
            continue
        if nm.startswith("unembed.") and args.skip_embed:
            continue
        keep.append(nm)

    f32_suffix = (".static_w", "_sconv.weight", ".rel_logits_proj.proj", ".q_l2_clip.scale",
                  ".rel_l2_clip.scale", ".attn_queries", ".mlp_queries", ".output_query",
                  "cross_pass_queries", "cross_pass_kv_queries", "norm.weight", ".gate.bias")
    desc = []
    for nm in keep:
        if nm.endswith(("_scale", "_amax", "_outer_scale_denominator")):
            continue
        src = "model.llm." + nm
        if nm.endswith(("w13_weight", "w2_weight")):
            rows, cols = (2 * inter, dim) if nm.endswith("w13_weight") else (dim, inter)
            kind = "w13" if nm.endswith("w13_weight") else "w2"
            size = q8e_size(E, rows, cols) if args.experts == "q8" else q4e_size(E, rows, cols)
            desc.append({"name": nm.replace("_weight", ""), "dtype": args.experts, "shape": [E, rows, cols],
                         "src": src, "kind": kind, "expert": True, "size": size})
            continue
        arr = st.read(src)
        if nm.endswith(f32_suffix) or ".attn_queries." in nm or ".mlp_queries." in nm:
            desc.append({"name": nm, "dtype": "f32", "shape": list(arr.shape), "src": src,
                         "size": int(arr.size) * 4})
            continue
        if arr.ndim != 2:
            raise SystemExit(f"{nm}: expected a 2-D matrix, got {arr.shape}")
        rows, cols = arr.shape
        if cols % 64:
            raise SystemExit(f"{nm}: cols {cols} not divisible by 64")
        desc.append({"name": nm, "dtype": "q8", "shape": [rows, cols], "src": src,
                     "size": q8_size(rows, cols)})

    off = DATA_START
    for d in desc:
        d["offset"] = off
        off = (off + d["size"] + 63) & ~63
    total = off
    out = open(args.output, "wb")
    out.truncate(total)
    print(f"{len(desc)} tensors, {total/1e9:.2f} GB, data at {DATA_START}", flush=True)

    # non-expert tensors (experts are written per expert below)
    for d in desc:
        if d.get("expert"):
            continue
        w = st.read(d["src"])
        blob = w.astype(np.float32).tobytes() if d["dtype"] == "f32" else q8_blob(w)
        assert len(blob) == d["size"], (d["name"], len(blob), d["size"])
        out.seek(d["offset"])
        out.write(blob)
    print("non-expert tensors written", flush=True)

    tasks = []
    for d in desc:
        if not d.get("expert"):
            continue
        stride = d["size"] // E
        for e in range(E):
            tasks.append((d["src"], d["kind"], e, d["offset"], stride, d["dtype"]))
    if tasks:
        global _W
        _W = (args.source, H, st.base)
        import multiprocessing as mp
        with mp.get_context("fork").Pool(args.procs) as pool:
            for n, blob in enumerate(pool.imap(expert_worker, tasks, chunksize=1)):
                _, _, e, off0, stride, _ = tasks[n]
                out.seek(off0 + e * stride)
                out.write(blob)
                if (n + 1) % 512 == 0:
                    print(f"experts {n+1}/{len(tasks)}", flush=True)
    out.close()

    manifest = {
        "config": {
            "arch": "inkling2_loop", "phys_layers": L,
            "slots": L + (loop["num_passes"] - 1) * (loop["core_end"] - loop["core_start"]),
            "dim": dim, "inter": inter, "heads": nh, "kv_heads": nkv, "head_dim": hd,
            "vocab": tc["vocab_size"], "unpadded_vocab": tc["unpadded_vocab_size"],
            "experts": E, "active": tc["num_experts_per_tok"],
            "loop_start": loop["core_start"], "loop_end": loop["core_end"],
            "passes": loop["num_passes"], "gate_rank": loop["gate_rank"],
            "d_rel": tc["d_rel"], "rel_extent": tc["rel_extent"],
            "window": tc["sliding_window_size"], "conv_k": tc["sconv_kernel_size"],
            "res_block": tc["n_layers_per_attn_res_block"], "local": sorted(local),
            "eps": tc["rms_norm_eps"], "qk_clip": tc["qk_l2_clip_threshold"],
            "rel_clip": tc["rel_l2_clip_threshold"], "log_floor": tc["log_scaling_n_floor"],
            "log_alpha": tc["log_scaling_alpha"], "final_softcap": tc["final_logit_softcapping"],
            "gate_softcap": tc["mlp_gate_softcap"], "up_softcap": tc["mlp_up_softcap"],
            "route_scale": tc["route_scale"], "eos": cfg["eos_token_id"], "special_start": 199998,
        },
        "tensors": desc,
    }
    js = json.dumps(manifest, separators=(",", ":")).encode()
    if len(js) > DATA_START - 16:
        raise SystemExit("manifest does not fit the 1 MiB header")
    with open(args.output, "r+b") as f:
        f.seek(0)
        f.write(b"INK2JSON")
        f.write(struct.pack("<Q", len(js)))
        f.write(js)
    print("manifest", len(js), "bytes; done", args.output, flush=True)


if __name__ == "__main__":
    main()
