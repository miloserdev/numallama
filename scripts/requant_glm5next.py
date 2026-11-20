"""
Re-quantize Q8 tensor kinds of a GLM-5.3-Flash numallama file (export_glm5next.py) to Q4, e.g. for a
smaller / faster variant once the original checkpoint is gone. The Q8 values are dequantized and
quantized again (asymmetric groups of 32); the Q8 rounding error is ~1/16 of Q4's, so the result is
close to a direct export.

    python3 requant_glm5next.py <in.bin> <out.bin> <kinds>    kinds e.g. lin_in,lin_out,q_b,o_proj,sh13,sh2

Kinds: see KINDS in export_glm5next.py (the router, the embedding and lm_head stay Q8).
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from export_glm5next import GS, KINDS, Layout, write_header, _pwrite  # noqa: E402
from export_hf_q8 import quantize_q4  # noqa: E402

_J = {}


def _job(key):
    kind, i = key
    src, dst = _J["src"], _J["dst"]
    r_in, r_out = src.regions[kind], dst.regions[kind]
    rows, cols = r_in["rows"], r_in["cols"]
    base_in = r_in["off"] + i * (r_in["qb"] + r_in["sb"])
    base_out = r_out["off"] + i * (r_out["qb"] + r_out["sb"])
    fi, fo = _J["fi"], _J["fo"]
    if r_in["bits"] == r_out["bits"]:  # copy
        n, off = r_in["qb"] + r_in["sb"], 0
        while off < n:
            k = min(n - off, 1 << 26)
            _pwrite(fo, os.pread(fi, k, base_in + off), base_out + off)
            off += k
        return 0.0
    assert r_in["bits"] == 8 and r_out["bits"] == 4
    err, step = 0.0, max(4, (1 << 24) // cols) // 4 * 4  # rows per piece
    for r0 in range(0, rows, step):
        nr = min(step, rows - r0)
        q = np.frombuffer(os.pread(fi, nr * cols, base_in + r0 * cols), np.int8).astype(np.float32).reshape(nr, cols // GS, GS)
        s = np.frombuffer(os.pread(fi, nr * (cols // GS) * 4, base_in + r_in["qb"] + r0 * (cols // GS) * 4), np.float32)
        w = (q * s.reshape(nr, cols // GS, 1)).reshape(nr, cols)
        pk, sm, e = quantize_q4(w)
        _pwrite(fo, pk.tobytes(), base_out + r0 * cols // 2)
        _pwrite(fo, sm.tobytes(), base_out + r_out["qb"] + r0 * (cols // 32) * 4)
        err = max(err, e)
    return err


def main():
    src_path, dst_path, kinds = sys.argv[1], sys.argv[2], [k for k in sys.argv[3].split(",") if k]
    assert all(k in KINDS and k not in ("router", "embed", "lm_head") for k in kinds), kinds
    plan = json.load(open(src_path + ".plan.json"))
    d = plan["dims"]
    q4 = sorted(set(plan["q4"]) | set(kinds))
    src, dst = Layout(d, plan["q4"]), Layout(d, q4)
    with open(dst_path, "wb") as f:
        write_header(f, d, q4)
        f.truncate(dst.size)
    json.dump(dict(dims=d, q4=q4), open(dst_path + ".plan.json", "w"))
    fi, fo = os.open(src_path, os.O_RDONLY), os.open(dst_path, os.O_WRONLY)
    # fp32 section: one block
    first_q = min(r["off"] for r in src.regions.values() if r["kind"] == "q")
    off = 256
    while off < first_q:
        k = min(first_q - off, 1 << 26)
        _pwrite(fo, os.pread(fi, k, off), off)
        off += k
    _J.update(src=src, dst=dst, fi=fi, fo=fo)
    keys = [(k, i) for k, r in src.regions.items() if r["kind"] == "q" for i in range(r["n"])]
    keys.sort(key=lambda kv: -(src.regions[kv[0]]["rows"] * src.regions[kv[0]]["cols"]))
    import multiprocessing as mp
    maxerr = 0.0
    with mp.get_context("fork").Pool(int(os.environ.get("EXPORT_PROCS", "64"))) as pool:
        for err in pool.imap_unordered(_job, keys, chunksize=1):
            maxerr = max(maxerr, err)
    os.fsync(fo)
    print(f"{dst_path}: {dst.size / 1e9:.2f} GB, q4={q4}, max requantization error {maxerr:.4f}")


if __name__ == "__main__":
    main()
