#!/usr/bin/env python3
"""Compare two logits dumps written by tests/dump_logits (f32 KV vs int8 KV).

usage: kv_quality.py ref.bin test.bin [positions]
Prints mean KL(ref||test), max KL, argmax agreement and the worst position.
"""
import sys

import numpy as np


def load(path, vocab):
    a = np.fromfile(path, dtype=np.float32)
    if a.size % vocab:
        raise SystemExit(f"{path}: size {a.size} is not a multiple of vocab {vocab}")
    return a.reshape(-1, vocab)


def main():
    if len(sys.argv) < 3:
        raise SystemExit(__doc__)
    vocab = int(sys.argv[3]) if len(sys.argv) > 3 else 151936
    a = load(sys.argv[1], vocab)
    b = load(sys.argv[2], vocab)
    n = min(len(a), len(b))
    if len(a) != len(b):
        print(f"warning: {len(a)} vs {len(b)} positions, comparing {n}")

    kl = np.empty(n, dtype=np.float64)
    agree = 0
    for i in range(n):
        x = a[i].astype(np.float64)
        y = b[i].astype(np.float64)
        x -= x.max()
        y -= y.max()
        p = np.exp(x)
        p /= p.sum()
        q = np.exp(y)
        q /= q.sum()
        kl[i] = float(np.sum(p * (np.log(p + 1e-30) - np.log(q + 1e-30))))
        agree += int(np.argmax(p) == np.argmax(q))
    worst = int(np.argmax(kl))
    print(f"positions {n}")
    print(f"KL mean {kl.mean():.6f} | KL max {kl.max():.6f} (pos {worst}) | argmax {agree}/{n} ({100.0*agree/n:.1f}%)")


if __name__ == "__main__":
    main()
