"""
Export a Hugging Face byte-level BPE tokenizer (tokenizer.json of Llama 3.x or Qwen2.5) to the
llama3.c tokenizer.bin format:  int max_token_length, then for every id: float score, int len, bytes.
The score is the BPE rank (== token id for these tiktoken-style vocabularies); ids missing from the
vocabulary (padding up to vocab_size) get unique placeholder strings.

    python3 export_tokenizer.py /path/to/Qwen2.5-3B-Instruct qwen25_tokenizer.bin
"""
import json
import os
import struct
import sys


def byte_decoder():
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {chr(c): b for b, c in zip(bs, cs)}


def main():
    model_dir, out_path = sys.argv[1], sys.argv[2]
    tj = json.load(open(os.path.join(model_dir, "tokenizer.json")))
    cfg = json.load(open(os.path.join(model_dir, "config.json")))
    vocab_size = cfg.get("vocab_size") or cfg.get("text_config", {})["vocab_size"]
    bd = byte_decoder()
    toks = {}
    for s, i in tj["model"]["vocab"].items():
        toks[i] = bytes(bd[c] for c in s)
    for a in tj.get("added_tokens", []):
        toks[a["id"]] = a["content"].encode("utf-8")
    vocab_size = max(vocab_size, max(toks) + 1)
    seen = set()
    with open(out_path, "wb") as f:
        entries = []
        for i in range(vocab_size):
            b = toks.get(i, f"<|unused_{i}|>".encode())
            if b in seen or b == b"" or b"\0" in b:
                b = f"<|dup_{i}|>".encode()
            seen.add(b)
            entries.append(b)
        f.write(struct.pack("<i", max(len(b) for b in entries)))
        for i, b in enumerate(entries):
            f.write(struct.pack("<fi", float(i), len(b)))
            f.write(b)
    print(f"wrote {out_path}: {vocab_size} tokens")


if __name__ == "__main__":
    main()
