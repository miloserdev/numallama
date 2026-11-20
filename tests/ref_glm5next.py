# Reference logits of a truncated GLM-5.3-Flash (the first n layers) from the transformers implementation
# (transformers >= 5.23, torch; fp32, eager attention), for comparison with `GLM_LAYERS=n tests/dump_logits`.
#   ref_glm5next.py <hf dir with config.json> <ref dir (export_glm5next.py --ref)> <ids file> <n layers> <out.npy>
#                   [our logits (N, vocab) f32 ...]
# The ref dir holds the original tensors of layers 0..3, embeddings, final norm and lm_head. Writes the
# logits of every position after each layer 1..n (mean of the mHC streams -> final norm -> lm_head),
# shape (n, N, vocab), and compares the last layer with our logits if given.
import glob
import json
import os
import struct
import sys

import numpy as np
import torch

hf_dir, ref_dir, ids_file, nl, out = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), sys.argv[5]
ours = sys.argv[6] if len(sys.argv) > 6 else None
torch.set_num_threads(int(os.environ.get("REF_THREADS", "64")))

FP8 = None


def fp8_table():
    t = np.zeros(256, np.float32)
    for b in range(256):
        s, e, m = b >> 7, (b >> 3) & 15, b & 7
        v = np.nan if (e == 15 and m == 7) else ((m / 8.0) * 2.0 ** -6 if e == 0 else (1 + m / 8.0) * 2.0 ** (e - 7))
        t[b] = -v if s else v
    return t


def load_dir(d):
    T = {}
    for f in sorted(glob.glob(os.path.join(d, "*safetensors"))):
        raw = np.memmap(f, dtype=np.uint8, mode="r")
        n = struct.unpack("<Q", bytes(raw[:8]))[0]
        hdr = json.loads(bytes(raw[8: 8 + n]))
        for name, info in hdr.items():
            if name == "__metadata__":
                continue
            a, b = info["data_offsets"]
            T[name] = (info["dtype"], info["shape"], raw[8 + n + a: 8 + n + b])
    return T


RAW = load_dir(ref_dir)
TAB = fp8_table()


def get(name):
    return torch.from_numpy(get_np(name))


def get_np(name):
    dt, shape, buf = RAW[name]
    if dt == "BF16":
        a = (np.asarray(buf).view(np.uint16).astype(np.uint32) << 16).view(np.float32)
    elif dt == "F32":
        a = np.asarray(buf).view(np.float32).copy()
    elif dt == "F8_E4M3":
        a = TAB[np.asarray(buf)].reshape(shape)
        sc = get_np(name + "_scale_inv")
        r, c = shape
        a = a * np.repeat(np.repeat(sc, 128, 0)[:r], 128, 1)[:, :c]
    else:
        raise ValueError(dt)
    return np.ascontiguousarray(a.reshape(shape).astype(np.float32))


from transformers.models.glm5_next.configuration_glm5_next import Glm5NextTextConfig  # noqa: E402
from transformers.models.glm5_next.modeling_glm5_next import Glm5NextTextModel  # noqa: E402

tc = json.load(open(os.path.join(hf_dir, "config.json")))["text_config"]
tc = dict(tc)
types = ["linear_attention" if t == "linear_attention" else "indexed_attention" for t in tc["layer_types"]]
tc["num_hidden_layers"] = nl
tc["layer_types"] = types[:nl]
tc["mlp_layer_types"] = tc["mlp_layer_types"][:nl]
tc["indexer_types"] = tc["indexer_types"][:nl]
la = dict(tc["linear_attn_config"])
la["kda_layers"] = [l for l in la["kda_layers"] if l < nl]
la["full_attn_layers"] = [l for l in la["full_attn_layers"] if l < nl]
tc["linear_attn_config"] = la
tc.pop("quantization_config", None)
tc["dtype"] = "float32"
cfg = Glm5NextTextConfig(**tc)
cfg._attn_implementation = "eager"
model = Glm5NextTextModel(cfg).float().eval()

P = "model.language_model."
sd = {"embed_tokens.weight": get(P + "embed_tokens.weight"), "norm.weight": get(P + "norm.weight")}
for l in range(nl):
    p, q = f"{P}layers.{l}.", f"layers.{l}."
    for n in ("input_layernorm.weight", "post_attention_layernorm.weight"):
        sd[q + n] = get(p + n)
    for part in ("attn", "ffn"):
        for k in ("fn", "base", "scale"):
            sd[q + f"{part}_hc.{k}"] = get(p + f"hc_{part}_{k}")
    a = p + "self_attn."
    if types[l] == "linear_attention":
        for n in ("q_proj", "k_proj", "v_proj", "b_proj", "g_a_proj", "g_b_proj", "o_proj"):
            sd[q + f"self_attn.{n}.weight"] = get(a + n + ".weight")
        sd[q + "self_attn.o_norm.weight"] = get(a + "o_norm.weight")
        for n in ("f_a_proj.weight", "f_b_proj.weight", "dt_bias", "A_log"):
            sd[q + "self_attn.forget_gate." + n] = get(a + n)
        sd[q + "self_attn.conv1d.weight"] = torch.cat([get(a + f"{c}_conv1d.weight") for c in "qkv"], 0)
    else:
        for n in ("q_a_proj", "q_b_proj", "kv_a_proj_with_mqa", "kv_b_proj", "o_proj"):
            sd[q + f"self_attn.{n}.weight"] = get(a + n + ".weight")
        for n in ("q_a_layernorm", "kv_a_layernorm"):
            sd[q + f"self_attn.{n}.weight"] = get(a + n + ".weight")
        for n in ("wq_b.weight", "wk.weight", "k_norm.weight", "k_norm.bias", "weights_proj.weight",
                  "index_kpool_compress_ape", "index_kpool_compress_gate"):
            sd[q + "self_attn.indexer." + n] = get(a + "indexer." + n)
    m = p + "mlp."
    if tc["mlp_layer_types"][l] == "dense":
        for n in ("gate_proj", "up_proj", "down_proj"):
            sd[q + f"mlp.{n}.weight"] = get(m + n + ".weight")
    else:
        E = tc["n_routed_experts"]
        sd[q + "mlp.gate.weight"] = get(m + "gate.weight")
        sd[q + "mlp.gate.e_score_correction_bias"] = get(m + "gate.e_score_correction_bias")
        for n in ("gate_proj", "up_proj", "down_proj"):
            sd[q + f"mlp.shared_experts.{n}.weight"] = get(m + f"shared_experts.{n}.weight")
        sd[q + "mlp.experts.gate_up_proj"] = torch.stack(
            [torch.cat([get(m + f"experts.{e}.gate_proj.weight"), get(m + f"experts.{e}.up_proj.weight")], 0) for e in range(E)])
        sd[q + "mlp.experts.down_proj"] = torch.stack([get(m + f"experts.{e}.down_proj.weight") for e in range(E)])
missing, unexpected = model.load_state_dict(sd, strict=False)
missing = [k for k in missing if not k.endswith("e_score_correction_bias")]
assert not missing and not unexpected, (missing[:8], unexpected[:8])
lm_head = get("lm_head.weight")

ids = [int(x) for x in open(ids_file).read().split()]
N = len(ids)
caps = []
hooks = [layer.register_forward_hook(lambda mod, inp, outp: caps.append(outp[0] if isinstance(outp, tuple) else outp))
         for layer in model.layers]
margins = {}
if os.environ.get("REF_ROUTER"):   # debug: gap between the k-th and (k+1)-th expert choice score per position
    for li, layer in enumerate(model.layers):
        if hasattr(layer.mlp, "gate"):
            def hook(mod, inp, outp, li=li):
                ch = torch.sigmoid(outp[0]) + mod.e_score_correction_bias
                srt = torch.sort(ch, -1, descending=True).values
                margins[li] = (srt[:, mod.top_k - 1] - srt[:, mod.top_k]).detach().numpy()
            layer.mlp.gate.register_forward_hook(hook)
if os.environ.get("REF_SEL"):   # debug: tokens selected by the first indexer, per position
    for layer in model.layers:
        if getattr(layer.self_attn, "indexer", None) is not None:
            def ihook(mod, inp, outp):
                for i, row in enumerate(outp[0].tolist()):
                    print(f"sel {i}:", " ".join(str(v) for v in sorted(x for x in row if x >= 0)))
            layer.self_attn.indexer.register_forward_hook(ihook)
            break
with torch.no_grad():
    model(input_ids=torch.tensor([ids]), use_cache=False)
    res = []
    for h in caps:
        x = model.norm(model.hc_head(h))[0]
        res.append((x @ lm_head.T).numpy())
for li, m in margins.items():
    print(f"layer {li} router margin per position:", " ".join(f"{i}:{v:.4f}" for i, v in enumerate(m)))
res = np.stack(res)
np.save(out, res)
print(f"reference: {nl} layers, {N} tokens -> {out}")
if ours:
    o = np.fromfile(ours, np.float32).reshape(N, -1)[:, : res.shape[-1]]
    r = res[-1]
    for i in range(N):
        a, b = r[i], o[i]
        c = np.corrcoef(a, b)[0, 1]
        rel = np.abs(a - b).max() / np.abs(a).max()
        ta, tb = np.argsort(-a)[:5], np.argsort(-b)[:5]
        print(f"pos {i:4d}: corr {c:.5f} max rel diff {rel:.4f} top1 ref {ta[0]} ours {tb[0]} {'OK' if ta[0] == tb[0] else 'DIFF'}"
              f"  top5 overlap {len(set(ta) & set(tb))}")
