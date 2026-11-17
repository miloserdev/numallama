# independent fp32 numpy reference for HF Llama / Qwen2 (rotate-half RoPE, optional llama3 scaling, qkv bias)
import sys, json, os, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'scripts'))
from export_hf_q8 import load_safetensors
md = sys.argv[1]; cfg = json.load(open(md + '/config.json')); T = {}
for f in sorted(x for x in os.listdir(md) if x.endswith('.safetensors')): T.update(load_safetensors(os.path.join(md, f)))
ids = [int(x) for x in open(sys.argv[2]).read().split()]
D, H, KV, L = cfg['hidden_size'], cfg['num_attention_heads'], cfg['num_key_value_heads'], cfg['num_hidden_layers']
hd = D // H; eps = cfg['rms_norm_eps']; N = len(ids)
inv = 1.0 / cfg['rope_theta'] ** (np.arange(0, hd, 2) / hd)
rs = cfg.get('rope_scaling')
if rs and rs.get('rope_type') == 'llama3':
    f, lo, hi, old = rs['factor'], rs['low_freq_factor'], rs['high_freq_factor'], rs['original_max_position_embeddings']
    wl = 2 * np.pi / inv; sm = (old / wl - lo) / (hi - lo)
    inv = np.where(wl < old / hi, inv, np.where(wl > old / lo, inv / f, (1 - sm) * inv / f + sm * inv))
ang = np.arange(N)[:, None] * inv[None, :]; cos = np.cos(np.concatenate([ang, ang], 1)); sin = np.sin(np.concatenate([ang, ang], 1))
def rot(x):
    x1, x2 = x[..., :hd // 2], x[..., hd // 2:]
    return x * cos[:, None] + np.concatenate([-x2, x1], -1) * sin[:, None]
def rms(x, w): return x / np.sqrt((x * x).mean(-1, keepdims=True) + eps) * w
g = lambda n: T[n].astype(np.float32)
gb = lambda n: T[n].astype(np.float32) if n in T else 0.0
x = g('model.embed_tokens.weight')[ids]
mask = np.triu(np.full((N, N), -np.inf, np.float32), 1)
for l in range(L):
    p = f'model.layers.{l}.'
    h = rms(x, g(p + 'input_layernorm.weight'))
    q = rot((h @ g(p + 'self_attn.q_proj.weight').T + gb(p + 'self_attn.q_proj.bias')).reshape(N, H, hd))
    k = rot((h @ g(p + 'self_attn.k_proj.weight').T + gb(p + 'self_attn.k_proj.bias')).reshape(N, KV, hd))
    v = (h @ g(p + 'self_attn.v_proj.weight').T + gb(p + 'self_attn.v_proj.bias')).reshape(N, KV, hd)
    k = np.repeat(k, H // KV, 1); v = np.repeat(v, H // KV, 1)
    s = np.einsum('qhd,khd->hqk', q, k) / np.sqrt(hd) + mask
    s = np.exp(s - s.max(-1, keepdims=True)); s /= s.sum(-1, keepdims=True)
    x = x + np.einsum('hqk,khd->qhd', s, v).reshape(N, D) @ g(p + 'self_attn.o_proj.weight').T
    h = rms(x, g(p + 'post_attention_layernorm.weight'))
    a = h @ g(p + 'mlp.gate_proj.weight').T; b = h @ g(p + 'mlp.up_proj.weight').T
    x = x + (a / (1 + np.exp(-a)) * b) @ g(p + 'mlp.down_proj.weight').T
W = g('lm_head.weight') if 'lm_head.weight' in T else g('model.embed_tokens.weight')
logits = rms(x, g('model.norm.weight')) @ W.T
c = np.fromfile(sys.argv[3], np.float32).reshape(N, -1)[:, :logits.shape[1]]
def lsm(z): z = z - z.max(1, keepdims=True); return z - np.log(np.exp(z).sum(1, keepdims=True))
kl = (np.exp(lsm(logits)) * (lsm(logits) - lsm(c))).sum(1)
print('top1 agree %.1f%%  mean KL %.4f  max KL %.4f  corr %.5f' % ((logits.argmax(1) == c.argmax(1)).mean() * 100, kl.mean(), kl.max(), np.corrcoef(logits.ravel(), c.ravel())[0, 1]))
if len(sys.argv) > 4:
    cb = np.fromfile(sys.argv[4], np.float32)[:logits.shape[1]]
    z = lsm(logits[-1:]); zb = lsm(cb[None]); print('batched prefill last token: KL vs ref %.4f, argmax %d vs ref %d' % ((np.exp(z) * (z - zb)).sum(), cb.argmax(), logits[-1].argmax()))
