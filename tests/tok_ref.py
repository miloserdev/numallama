# reference Llama 3 tokenizer (tiktoken-style byte BPE with merge ranks) from tokenizer.json, stdlib only
import json, re, sys
import os
tj = json.load(open(os.environ['TOKJSON']))
ND = os.environ.get('NDIG','1,3')
vocab = tj['model']['vocab']; merges = tj['model']['merges']
ranks = {tuple(m.split(' ')) if isinstance(m, str) else tuple(m): i for i, m in enumerate(merges)}
bs = list(range(ord('!'), ord('~') + 1)) + list(range(ord('¡'), ord('¬') + 1)) + list(range(ord('®'), ord('ÿ') + 1))
cs = bs[:]; n = 0
for b in range(256):
    if b not in bs: bs.append(b); cs.append(256 + n); n += 1
b2u = {b: chr(c) for b, c in zip(bs, cs)}
L = r"[^\W\d_]"
pat = re.compile(r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|(?:[^\r\n\w]|_)?[^\W\d_]+|\d{" + ND + "}| ?(?:[^\s\w]|_)+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+")
def bpe(w):
    parts = list(w)
    while len(parts) > 1:
        best = min(((ranks.get((parts[i], parts[i + 1]), 1e18), i) for i in range(len(parts) - 1)))
        if best[0] == 1e18: break
        i = best[1]; parts[i:i + 2] = [parts[i] + parts[i + 1]]
    return [vocab[p] for p in parts]
for line in sys.stdin:
    line = line.rstrip('\n'); out = []
    for piece in pat.findall(line):
        out += bpe(''.join(b2u[b] for b in piece.encode('utf-8')))
    print(' '.join(map(str, out)))
