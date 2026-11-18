# independent fp32 numpy reference for Qwen3.5 / Qwen3.8 dense text models (model_type qwen3_5):
# hybrid layers (Gated DeltaNet linear attention, every `full_attention_interval`-th layer full
# attention with an output gate and partial RoPE), zero-centered RMSNorm (x * (1 + w)).
#   ref_qwen35.py <hf dir> <ids file> <our logits (N, vocab) f32> [batched last-token logits]
import sys, json, os, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'scripts'))
from export_hf_q8 import load_safetensors
md = sys.argv[1]
cfg = json.load(open(md + '/config.json')); cfg = cfg.get('text_config', cfg)
T = {}
for f in sorted(x for x in os.listdir(md) if x.endswith('.safetensors')): T.update(load_safetensors(os.path.join(md, f)))
P = 'model.language_model.' if 'model.language_model.embed_tokens.weight' in T else 'model.'
g = lambda n: T[n].astype(np.float32)
ids = [int(x) for x in open(sys.argv[2]).read().split()]
N = len(ids)
D, H, KV, L = cfg['hidden_size'], cfg['num_attention_heads'], cfg['num_key_value_heads'], cfg['num_hidden_layers']
hd, eps = cfg['head_dim'], cfg['rms_norm_eps']
rp = cfg.get('rope_parameters', {})
theta = rp.get('rope_theta', cfg.get('rope_theta'))
rot = int(hd * rp.get('partial_rotary_factor', cfg.get('partial_rotary_factor', 1.0)))
inv = 1.0 / theta ** (np.arange(0, rot, 2) / rot)
ang = np.arange(N)[:, None] * inv[None, :]
cos, sin = np.cos(np.concatenate([ang, ang], 1)), np.sin(np.concatenate([ang, ang], 1))
nk, nv, dk, dv, K = (cfg['linear_num_key_heads'], cfg['linear_num_value_heads'], cfg['linear_key_head_dim'],
                     cfg['linear_value_head_dim'], cfg['linear_conv_kernel_dim'])
def rms0(x, w): return x / np.sqrt((x * x).mean(-1, keepdims=True) + eps) * (1.0 + w)
def ropef(x):   # (N, heads, hd): rotate-half on the first `rot` dims
    xr, xp = x[..., :rot], x[..., rot:]
    x1, x2 = xr[..., :rot // 2], xr[..., rot // 2:]
    xr = xr * cos[:, None] + np.concatenate([-x2, x1], -1) * sin[:, None]
    return np.concatenate([xr, xp], -1)
def l2n(x): return x / np.sqrt((x * x).sum(-1, keepdims=True) + 1e-6)
silu = lambda a: a / (1 + np.exp(-a))
x = g(P + 'embed_tokens.weight')[ids]
mask = np.triu(np.full((N, N), -np.inf, np.float32), 1)
for l in range(L):
    p = f'{P}layers.{l}.'
    h = rms0(x, g(p + 'input_layernorm.weight'))
    if cfg['layer_types'][l] == 'full_attention':
        qg = (h @ g(p + 'self_attn.q_proj.weight').T).reshape(N, H, 2 * hd)
        q, gate = qg[..., :hd], qg[..., hd:].reshape(N, H * hd)
        q = ropef(rms0(q, g(p + 'self_attn.q_norm.weight')))
        k = ropef(rms0((h @ g(p + 'self_attn.k_proj.weight').T).reshape(N, KV, hd), g(p + 'self_attn.k_norm.weight')))
        v = (h @ g(p + 'self_attn.v_proj.weight').T).reshape(N, KV, hd)
        k = np.repeat(k, H // KV, 1); v = np.repeat(v, H // KV, 1)
        s = np.einsum('qhd,khd->hqk', q, k) / np.sqrt(hd) + mask
        s = np.exp(s - s.max(-1, keepdims=True)); s /= s.sum(-1, keepdims=True)
        o = np.einsum('hqk,khd->qhd', s, v).reshape(N, H * hd) * (1 / (1 + np.exp(-gate)))
        x = x + o @ g(p + 'self_attn.o_proj.weight').T
    else:
        a_ = p + 'linear_attn.'
        qkv = h @ g(a_ + 'in_proj_qkv.weight').T                 # (N, 2 nk dk + nv dv)
        w = g(a_ + 'conv1d.weight').reshape(qkv.shape[1], K)
        pad = np.concatenate([np.zeros((K - 1, qkv.shape[1]), np.float32), qkv], 0)
        qkv = silu(sum(w[:, j] * pad[j:j + N] for j in range(K)))
        q = qkv[:, :nk * dk].reshape(N, nk, dk); k = qkv[:, nk * dk:2 * nk * dk].reshape(N, nk, dk)
        v = qkv[:, 2 * nk * dk:].reshape(N, nv, dv)
        z = (h @ g(a_ + 'in_proj_z.weight').T).reshape(N, nv, dv)
        beta = 1 / (1 + np.exp(-(h @ g(a_ + 'in_proj_b.weight').T)))
        gg = -np.exp(g(a_ + 'A_log')) * np.log1p(np.exp(h @ g(a_ + 'in_proj_a.weight').T + g(a_ + 'dt_bias')))
        q = np.repeat(l2n(q), nv // nk, 1) / np.sqrt(dk); k = np.repeat(l2n(k), nv // nk, 1)
        S = np.zeros((nv, dk, dv), np.float32); o = np.zeros((N, nv, dv), np.float32)
        for t in range(N):
            S = S * np.exp(gg[t])[:, None, None]
            kv = np.einsum('hkv,hk->hv', S, k[t])
            delta = (v[t] - kv) * beta[t][:, None]
            S = S + k[t][:, :, None] * delta[:, None, :]
            o[t] = np.einsum('hkv,hk->hv', S, q[t])
        o = o / np.sqrt((o * o).mean(-1, keepdims=True) + eps) * g(a_ + 'norm.weight') * silu(z)
        x = x + o.reshape(N, nv * dv) @ g(a_ + 'out_proj.weight').T
    h = rms0(x, g(p + 'post_attention_layernorm.weight'))
    a = h @ g(p + 'mlp.gate_proj.weight').T; b = h @ g(p + 'mlp.up_proj.weight').T
    x = x + (silu(a) * b) @ g(p + 'mlp.down_proj.weight').T
logits = rms0(x, g(P + 'norm.weight')) @ g('lm_head.weight').T
def lsm(z): z = z - z.max(-1, keepdims=True); return z - np.log(np.exp(z).sum(-1, keepdims=True))
c = np.fromfile(sys.argv[3], np.float32).reshape(N, -1)[:, :logits.shape[1]]
R, C = lsm(logits), lsm(c)
kl = (np.exp(R) * (R - C)).sum(1)
print('top1 agree %.1f%%  mean KL %.4f  max KL %.4f  corr %.5f' % ((logits.argmax(1) == c.argmax(1)).mean() * 100, kl.mean(), kl.max(), np.corrcoef(logits.ravel(), c.ravel())[0, 1]))
if len(sys.argv) > 4:
    zb = lsm(np.fromfile(sys.argv[4], np.float32)[:logits.shape[1]][None])
    print('batched prefill last token: KL vs ref %.4f, argmax %d vs ref %d' % ((np.exp(R[-1:]) * (R[-1:] - zb)).sum(), zb.argmax(), logits[-1].argmax()))
