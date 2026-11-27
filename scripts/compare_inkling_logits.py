#!/usr/bin/env python3
"""Compare Inkling-2-Mini logits against the llama.cpp oracle.

usage: compare_inkling_logits.py oracle.bin candidate.bin [candidate.bin ...]

Each file is a flat float32 array over the unpadded vocabulary (200058 values), as written by
tests/check_inkling (`logits_ours.bin`) or the oracle runner. Prints top-1, KL(oracle||cand),
top-10 overlap, Pearson correlation and the largest logit difference; additionally the KL between
candidate pairs, which separates the quantization loss from the forward/format difference.
"""
import sys
import numpy as np


def softmax(x):
    x = x.astype(np.float64)
    x -= x.max()
    e = np.exp(x)
    return e / e.sum()


def kl(p, q):
    return float(np.sum(p * np.log((p + 1e-30) / (q + 1e-30))))


def main():
    if len(sys.argv) < 3:
        raise SystemExit(__doc__)
    files = sys.argv[1:]
    logits = [np.fromfile(f, dtype=np.float32) for f in files]
    n = min(map(len, logits))
    logits = [x[:n] for x in logits]
    base = logits[0]
    pb = softmax(base)
    tb = base.argsort()[-10:][::-1]
    print(f"vocab {n}  oracle top1 {tb[0]} ({base[tb[0]]:.6f})")
    for name, x in zip(files[1:], logits[1:]):
        px = softmax(x)
        tx = x.argsort()[-10:][::-1]
        print(f"{name}: top1 {tx[0]} ({x[tx[0]]:.6f})  KL {kl(pb, px):.6f}  "
              f"top10 {len(set(tb) & set(tx))}/10  corr {np.corrcoef(base, x)[0, 1]:.5f}  "
              f"max|d| {np.max(np.abs(base - x)):.4f}")
    if len(logits) > 2:
        for i in range(1, len(logits)):
            for j in range(i + 1, len(logits)):
                a, b = softmax(logits[i]), softmax(logits[j])
                print(f"{files[i]} vs {files[j]}: KL {kl(a, b):.6f}  "
                      f"corr {np.corrcoef(logits[i], logits[j])[0, 1]:.5f}  "
                      f"max|d| {np.max(np.abs(logits[i] - logits[j])):.4f}")


if __name__ == "__main__":
    main()
