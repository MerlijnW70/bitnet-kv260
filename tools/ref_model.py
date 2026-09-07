"""The hybrid reference for BitNet b1.58 2B4T on the KV260: a numpy implementation, token by token
with a KV cache, of exactly what runtime/bitnet_kria.c will compute -- integer matvecs where the
ternary engines run, the integer FFN of reference.ffn_int (spec v2) where the glue runs, float32
on the A53 for everything else -- so the board can be checked stage by stage against it.

usage
  python ref_model.py --self-test                     the exactness and glue checks, no HF needed
  python ref_model.py --check --hf DIR [--report F]   against hf_dump.py's tensors in DIR
  python ref_model.py --dump DIR [--dump-tokens 8]    the stage dumps the C runtime checks against
  python ref_model.py --verify DIR                    stages.bin read back == the .npy files
  python ref_model.py --ask "..." [--new 64]          greedy generation, the reference alone
options: --prompt TEXT, --new N, --cache-dtype f32|bf16, --head int8|bf16, --snapshot DIR,
         --tokens "1502,25,..." (skip the tokenizer and give the ids), --max-layers N (debug)

what one token costs, layer by layer (h the residual stream, float32 [2560])
  x  = rmsnorm(h, input_layernorm, eps);       xq, sx = absmax_int8(x)
  q  = matvec_int(W_q, xq) ws_q / sx           k, v likewise           (the four engines)
  q, k rotated by RoPE at this position, k and v appended to the cache
  grouped-query attention, 4 query heads a key/value head, softmax in float32   (the A53)
  a  = rmsnorm(attn, attn_sub_norm, eps);      aq, sa = absmax_int8(a)
  h += matvec_int(W_o, aq) ws_o / sa
  x2 = rmsnorm(h, post_attention_layernorm, eps); x2q, sx2 = absmax_int8(x2)
  g  = matvec_int(W_gate, x2q), u = matvec_int(W_up, x2q)               (the engines)
  the glue: s_g, s_u, kg, G, s_S, s_T, s_V, h_int, hmax, m, q8 as reference.ffn_int fixes them
  d  = matvec_int(W_down, q8)                                           (the engines)
  h += d ws_down absmax_y / 127                 the scale below
then the final norm and the head.

the scale bookkeeping of the FFN, the one place the integers do not carry their own scale
  the glue's q8 is an absmax int8 quantisation of h_int, and h_int is proportional to the true
  ffn_sub_norm output y, so q8 IS the int8 quantisation of y and the float output of the FFN is
      d ws_down (max|y| / 127).
  max|y| comes out of the integers: with P_i = relu(g_i)^2 u_i (an exact integer, g and u the
  engines' int32 sums) and Kc = ws_gate^2 ws_up / sx2^3, the float intermediate is z = Kc P, so
      y = gamma z / sqrt(mean(z^2) + eps) = gamma P / sqrt(mean(P^2) + eps / Kc^2)
      max|y| = max_i |gamma_i P_i| / sqrt(mean(P^2) + eps / Kc^2)
  and Kc cancels out of everything but that eps term (which is ~1e-25 of the sum in practice).
  check_scales() recomputes max|y| along the float path and reports the relative departure.

matvecs are float32 BLAS on ternary weights held as float32, which is EXACT: every operand is an
integer, every partial sum of at most 6912 terms of at most 128 in modulus is at most 884736 <
2^24, so every intermediate is representable and every addition is exact whatever order BLAS
picks. self_test() checks that against an int64 matvec on sampled rows.

The weights are read straight from the local Hugging Face cache (model.safetensors, the packed
U8 BitLinear layout: 4 weights a byte as w+1, packed row r bit pair i holding weight row
i*(N/4) + r), never re-downloaded.
"""
import glob
import json
import os
import struct
import sys
import time

import numpy as np

import reference

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.expanduser("~/.cache/huggingface/hub/models--microsoft--bitnet-b1.58-2B-4T/snapshots")
PROMPT = "What is the capital of France, and what river runs through it? Explain in three sentences."
PROJ = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
NORMS = ("input_layernorm", "post_attention_layernorm", "attn_sub_norm", "ffn_sub_norm")


def find_snapshot(path=None):
    if path:
        return path
    cands = sorted(glob.glob(os.path.join(CACHE, "*")))
    for c in cands:
        if os.path.exists(os.path.join(c, "model.safetensors")):
            return c
    raise SystemExit(f"no model.safetensors under {CACHE}; pass --snapshot DIR")


def bf16_to_f32(raw):
    return (np.frombuffer(raw, dtype="<u2").astype(np.uint32) << 16).view(np.float32)


def round_bf16(x):
    """x float32 rounded to bfloat16 precision, half to even, kept as float32."""
    u = np.ascontiguousarray(x, dtype=np.float32).view(np.uint32)
    return ((u + 0x7FFF + ((u >> 16) & 1)) & 0xFFFF0000).view(np.float32)


def rmsnorm(x, w, eps):
    """LlamaRMSNorm in float32: w * x / sqrt(mean(x^2) + eps)."""
    v = np.float32(np.mean(np.square(x, dtype=np.float32), dtype=np.float32))
    return (x * np.float32(1.0 / np.sqrt(v + eps))) * w


def absmax_int8(x):
    """BitNet's ActQuant: (int8, scale) with scale = 127 / max(max|x|, 1e-5), round half to even."""
    amax = np.float32(max(float(np.abs(x).max()), 1e-5))
    scale = np.float32(127.0) / amax
    return np.clip(np.round(x * scale), -128, 127).astype(np.int8), scale


def matvec_i32(W, q):
    """The exact int32 W @ q for a ternary float32 W [N, K] and an int8 q [K]."""
    return np.rint(W @ q.astype(np.float32)).astype(np.int32)


def softmax(s):
    e = np.exp(s - s.max(), dtype=np.float32)
    return e / e.sum(dtype=np.float32)


def ffn_glue(g, u, gamma):
    """The FFN glue of spec v2 on the engines' int32 sums: everything reference.ffn_int computes
    between the gate/up matvecs and the down matvec. Returns a dict."""
    g64, u64 = g.astype(np.int64), u.astype(np.int64)
    relu_g, abs_u = np.maximum(g64, 0), np.abs(u64)
    s_g = max(0, int(relu_g.max()).bit_length() - 15)
    s_u = max(0, int(abs_u.max()).bit_length() - 15)
    a, b = relu_g >> s_g, abs_u >> s_u
    kg, G = reference.gamma_fixed(gamma)
    G64 = G.astype(np.int64)
    S = a * a
    s_S = max(0, int(S.max()).bit_length() - 15)
    S1 = S >> s_S
    T = S1 * b
    s_T = max(0, int(T.max()).bit_length() - 15)
    T1 = T >> s_T
    V = T1 * np.abs(G64)
    s_V = max(0, int(V.max()).bit_length() - 15)
    V1 = V >> s_V
    h = np.sign(u64) * np.sign(G64) * V1
    abs_h = np.abs(h)
    hmax = int(abs_h.max())
    m = 65535 if hmax == 0 else min(65535, (127 << 15) // hmax)
    reference.assert_16x16(a, b, S1, T1, V1, np.abs(G64), abs_h, m)
    q8 = (np.sign(h) * np.minimum(127, (abs_h * m + (1 << 14)) >> 15)).astype(np.int8)
    return {"s_g": s_g, "s_u": s_u, "kg": kg, "G": G, "s_S": s_S, "s_T": s_T, "s_V": s_V,
            "h": h.astype(np.int16), "hmax": hmax, "m": m, "q8": q8}


def ffn_absmax(g, u, gamma, ws_gate, ws_up, sx, eps):
    """max|y| of the true ffn_sub_norm output, from the engines' integers alone, with the parts
    the check needs: P = relu(g)^2 u exactly, Kc = ws_gate^2 ws_up / sx^3, and
    max|y| = max|gamma P| / sqrt(mean(P^2) + eps / Kc^2)."""
    P = np.square(np.maximum(g.astype(np.float64), 0.0)) * u.astype(np.float64)
    Kc = float(ws_gate) ** 2 * float(ws_up) / float(sx) ** 3
    denom = np.sqrt(np.mean(np.square(P)) + eps / Kc ** 2)
    return float(np.abs(gamma.astype(np.float64) * P).max() / denom), P, Kc, denom


class Ref:
    """The model: ternary weights as float32 [N, K], norms and the embedding as float32."""

    def __init__(self, snapshot=None, head="int8", cache_dtype="f32", max_layers=None, verbose=True):
        self.snapshot = find_snapshot(snapshot)
        self.head_mode = head
        self.cache_dtype = cache_dtype
        self.verbose = verbose
        cfg = json.load(open(os.path.join(self.snapshot, "config.json")))
        self.cfg = cfg
        self.hidden = cfg["hidden_size"]
        self.inter = cfg["intermediate_size"]
        self.n_layers = cfg["num_hidden_layers"] if max_layers is None else min(max_layers, cfg["num_hidden_layers"])
        self.n_q = cfg["num_attention_heads"]
        self.n_kv = cfg["num_key_value_heads"]
        self.head_dim = cfg.get("head_dim") or self.hidden // self.n_q
        self.groups = self.n_q // self.n_kv
        self.eps = float(cfg["rms_norm_eps"])
        self.theta = float(cfg["rope_theta"])
        self.vocab = cfg["vocab_size"]
        self.scaling = np.float32(self.head_dim ** -0.5)
        self.inv_freq = (1.0 / (self.theta ** (np.arange(0, self.head_dim, 2, dtype=np.float64) / self.head_dim))).astype(np.float64)
        self._load()

    def _log(self, *a):
        if self.verbose:
            print(*a, flush=True)

    def _load(self):
        t0 = time.time()
        path = os.path.join(self.snapshot, "model.safetensors")
        f = open(path, "rb")
        (hl,) = struct.unpack("<Q", f.read(8))
        header = json.loads(f.read(hl))
        base = 8 + hl
        mm = np.memmap(path, dtype=np.uint8, mode="r")

        def raw(name):
            meta = header[name]
            a, b = meta["data_offsets"]
            return mm[base + a:base + b], meta

        def ternary(name):
            data, meta = raw(name)
            assert meta["dtype"] == "U8", (name, meta["dtype"])
            rows, k = meta["shape"]
            packed = np.asarray(data).reshape(rows, k)
            out = np.empty((rows * 4, k), dtype=np.float32)
            for i in range(4):
                out[i * rows:(i + 1) * rows] = (packed >> (2 * i)) & 3
            out -= 1.0
            return out

        def bf16(name):
            data, meta = raw(name)
            assert meta["dtype"] == "BF16", (name, meta["dtype"])
            return bf16_to_f32(np.asarray(data).tobytes()).reshape(meta["shape"]).copy()

        self.layers = []
        for l in range(self.n_layers):
            p = f"model.layers.{l}"
            lay = {"W": {}, "ws": {}}
            for proj in PROJ:
                sub = "mlp" if proj in ("gate_proj", "up_proj", "down_proj") else "self_attn"
                lay["W"][proj] = ternary(f"{p}.{sub}.{proj}.weight")
                lay["ws"][proj] = float(bf16(f"{p}.{sub}.{proj}.weight_scale")[0])
            lay["input_layernorm"] = bf16(f"{p}.input_layernorm.weight")
            lay["post_attention_layernorm"] = bf16(f"{p}.post_attention_layernorm.weight")
            lay["attn_sub_norm"] = bf16(f"{p}.self_attn.attn_sub_norm.weight")
            lay["ffn_sub_norm"] = bf16(f"{p}.mlp.ffn_sub_norm.weight")
            self.layers.append(lay)
            if self.verbose and (l + 1) % 10 == 0:
                self._log(f"  {l + 1}/{self.n_layers} layers in {time.time() - t0:.0f} s")
        self.final_norm = bf16("model.norm.weight")
        self.embed = bf16("model.embed_tokens.weight")
        self.load_seconds = time.time() - t0
        gb = sum(w.nbytes for lay in self.layers for w in lay["W"].values()) / 2 ** 30
        self._log(f"loaded {self.n_layers} layers ({gb:.2f} GiB of float32 ternary) + embed "
                  f"{self.embed.shape} in {self.load_seconds:.0f} s from {self.snapshot}")
        if self.head_mode == "int8":
            t1 = time.time()
            amax = np.abs(self.embed).max(axis=1)
            self.head_scale = np.where(amax > 0, amax / 127.0, 1.0).astype(np.float32)
            self.head_f = np.clip(np.round(self.embed / self.head_scale[:, None]), -127, 127)
            self.head_i8 = self.head_f.astype(np.int8)
            self._log(f"int8 head: {self.head_i8.shape} in {time.time() - t1:.0f} s, "
                      f"row scale range [{self.head_scale.min():.6g}, {self.head_scale.max():.6g}]")

    def rope(self, pos):
        ang = pos * self.inv_freq
        c = np.concatenate([np.cos(ang), np.cos(ang)]).astype(np.float32)
        s = np.concatenate([np.sin(ang), np.sin(ang)]).astype(np.float32)
        return c, s

    @staticmethod
    def rotate_half(x):
        half = x.shape[-1] // 2
        return np.concatenate([-x[..., half:], x[..., :half]], axis=-1)

    def step(self, token, pos, cache, trace=None):
        """One token through the whole model. cache is a list of [K, V] per layer, each grown in
        place. trace, if a dict, collects layer 0's intermediates."""
        h = self.embed[token].astype(np.float32).copy()
        if trace is not None:
            trace["hidden"] = [h.copy()]
        for l, lay in enumerate(self.layers):
            W, ws = lay["W"], lay["ws"]
            t = trace if (trace is not None and l == 0) else None
            x = rmsnorm(h, lay["input_layernorm"], self.eps)
            xq, sx = absmax_int8(x)
            qi = matvec_i32(W["q_proj"], xq)
            ki = matvec_i32(W["k_proj"], xq)
            vi = matvec_i32(W["v_proj"], xq)
            q = (qi.astype(np.float32) * np.float32(ws["q_proj"] / sx)).reshape(self.n_q, self.head_dim)
            k = (ki.astype(np.float32) * np.float32(ws["k_proj"] / sx)).reshape(self.n_kv, self.head_dim)
            v = (vi.astype(np.float32) * np.float32(ws["v_proj"] / sx)).reshape(self.n_kv, self.head_dim)
            c, s = self.rope(pos)
            q = q * c + self.rotate_half(q) * s
            k = k * c + self.rotate_half(k) * s
            if self.cache_dtype == "bf16":
                k, v = round_bf16(k), round_bf16(v)
            cache[l][0].append(k)
            cache[l][1].append(v)
            K = np.stack(cache[l][0], axis=1)
            V = np.stack(cache[l][1], axis=1)
            attn = np.empty(self.hidden, dtype=np.float32)
            for hq in range(self.n_q):
                gidx = hq // self.groups
                sc = (K[gidx] @ q[hq]) * self.scaling
                attn[hq * self.head_dim:(hq + 1) * self.head_dim] = softmax(sc) @ V[gidx]
            a = rmsnorm(attn, lay["attn_sub_norm"], self.eps)
            aq, sa = absmax_int8(a)
            oi = matvec_i32(W["o_proj"], aq)
            o = oi.astype(np.float32) * np.float32(ws["o_proj"] / sa)
            h = h + o
            if t is not None:
                t.update(attn_in=x.copy(), xq=xq.copy(), sx=float(sx), qi=qi, ki=ki, vi=vi,
                         q=q.copy(), k=k.copy(), v=v.copy(), attn_out=attn.copy(),
                         attn_sub=a.copy(), aq=aq.copy(), sa=float(sa), oi=oi, o=o.copy())
            r = h
            x2 = rmsnorm(h, lay["post_attention_layernorm"], self.eps)
            x2q, sx2 = absmax_int8(x2)
            gi = matvec_i32(W["gate_proj"], x2q)
            ui = matvec_i32(W["up_proj"], x2q)
            gl = ffn_glue(gi, ui, lay["ffn_sub_norm"])
            di = matvec_i32(W["down_proj"], gl["q8"])
            absmax_y, P, Kc, denom = ffn_absmax(gi, ui, lay["ffn_sub_norm"], ws["gate_proj"], ws["up_proj"], sx2, self.eps)
            ffn = di.astype(np.float32) * np.float32(ws["down_proj"] * absmax_y / 127.0)
            h = r + ffn
            if t is not None:
                t.update(mlp_in=x2.copy(), x2q=x2q.copy(), sx2=float(sx2), gi=gi, ui=ui, di=di,
                         glue=gl, absmax_y=absmax_y, Kc=Kc, denom=denom, P=P, ffn=ffn.copy())
            if trace is not None:
                trace["hidden"].append(h.copy())
        fin = rmsnorm(h, self.final_norm, self.eps)
        if trace is not None:
            trace["final_norm"] = fin.copy()
        return h, fin

    def logits(self, fin, mode=None):
        """The head. "bf16" is HF's own: the float embedding matrix times the final hidden state.
        "int8" is the board's: the per-row int8 head times the int8-quantised hidden state, the
        int32 sums scaled by the row scale and the activation scale."""
        mode = mode or self.head_mode
        if mode == "bf16":
            return self.embed @ fin
        q, s = absmax_int8(fin)
        acc = np.rint(self.head_f @ q.astype(np.float32))
        return acc.astype(np.float32) * self.head_scale * np.float32(1.0 / s)

    def run(self, ids, new=0, dump_tokens=0, eos=(128001, 128009), progress=False):
        """The prompt then `new` greedy tokens. Returns (hidden [31, T, 2560] for the dumped
        positions, per-token traces, generated ids, timings)."""
        cache = [[[], []] for _ in range(self.n_layers)]
        traces, hidden, gen, fins = [], [], [], []
        t0 = time.time()
        for i, tid in enumerate(ids):
            trace = {} if i < dump_tokens else None
            h, fin = self.step(int(tid), i, cache, trace)
            if trace is not None:
                hidden.append(np.stack(trace.pop("hidden")))
                traces.append(trace)
            fins.append(fin)
            if progress and (i + 1) % 8 == 0:
                self._log(f"  prompt {i + 1}/{len(ids)} in {time.time() - t0:.0f} s")
        prompt_seconds = time.time() - t0
        pos = len(ids)
        last = fins[-1]
        t1 = time.time()
        for _ in range(new):
            nxt = int(np.argmax(self.logits(last)))
            gen.append(nxt)
            if nxt in eos:
                break
            _, last = self.step(nxt, pos, cache, None)
            pos += 1
        gen_seconds = time.time() - t1
        return {"hidden": np.stack(hidden) if hidden else None, "traces": traces, "fins": fins,
                "gen": gen, "prompt_seconds": prompt_seconds, "gen_seconds": gen_seconds}


DT = {"float32": 0, "int32": 1, "int16": 2, "int8": 3}
MAGIC = b"REFSTG\x01\x00"


def write_stage_bin(path, arrays):
    """A flat binary the C runtime reads without a JSON parser:
       magic char[8] "REFSTG\\x01\\x00"; uint32 count; uint32 data_start; then `count` entries of
       72 bytes: char name[32] (nul padded), uint32 dtype (0 f32, 1 i32, 2 i16, 3 i8),
       uint32 ndim, uint32 dims[4], uint64 offset, uint64 nbytes; then the arrays, each aligned
       to 64 bytes, little endian, C order. data_start is 4096-aligned."""
    entries, blobs, off = [], [], 0
    for name, a in arrays:
        a = np.ascontiguousarray(a)
        assert a.ndim <= 4 and len(name) < 32, (name, a.shape)
        off = (off + 63) & ~63
        entries.append((name, DT[a.dtype.name], a.ndim, list(a.shape) + [0] * (4 - a.ndim), off, a.nbytes))
        blobs.append((off, a))
        off += a.nbytes
    head = struct.pack("<8sII", MAGIC, len(entries), 0)
    for name, dt, nd, dims, o, nb in entries:
        head += struct.pack("<32sII4IQQ", name.encode(), dt, nd, *dims, o, nb)
    data_start = (len(head) + 4095) & ~4095
    head = struct.pack("<8sII", MAGIC, len(entries), data_start) + head[16:]
    with open(path, "wb") as f:
        f.write(head)
        f.write(b"\0" * (data_start - len(head)))
        end = 0
        for o, a in blobs:
            f.write(b"\0" * (o - end))
            f.write(a.tobytes())
            end = o + a.nbytes
    return data_start + off


def read_stage_bin(path):
    """stages.bin back as {name: ndarray}, by the same rules a C reader follows."""
    names = {v: k for k, v in DT.items()}
    raw = open(path, "rb").read()
    magic, count, data_start = struct.unpack_from("<8sII", raw, 0)
    assert magic == MAGIC, magic
    out = {}
    for i in range(count):
        name, dt, nd, d0, d1, d2, d3, off, nb = struct.unpack_from("<32sII4IQQ", raw, 16 + 72 * i)
        shape = [d0, d1, d2, d3][:nd]
        a = np.frombuffer(raw, dtype=names[dt], count=nb // np.dtype(names[dt]).itemsize,
                          offset=data_start + off).reshape(shape)
        out[name.rstrip(b"\0").decode()] = a
    return out


def verify_dump(out_dir):
    """Every array in stages.bin equals its .npy, read back the way the C reader will."""
    got = read_stage_bin(os.path.join(out_dir, "stages.bin"))
    index = json.load(open(os.path.join(out_dir, "ref_dump.json")))
    lines = []
    for entry in index["arrays"]:
        n = entry["name"]
        want = np.load(os.path.join(out_dir, f"{n}.npy"))
        a = got[n]
        assert a.dtype == want.dtype and list(a.shape) == list(want.shape) == entry["shape"], (n, a.shape, want.shape)
        assert np.array_equal(a, want), n
        lines.append(f"  {n:14s} {a.dtype.name:8s} {tuple(a.shape)} identical in stages.bin and {n}.npy")
    assert set(got) == {e["name"] for e in index["arrays"]}, set(got) ^ {e["name"] for e in index["arrays"]}
    return lines


def dump_stages(model, ids, out_dir, ptokens=8, new=0):
    """Every array the runtime's bring-up compares against, for the first `ptokens` prompt tokens,
    as .npy and in one stages.bin, with ref_dump.json naming them."""
    os.makedirs(out_dir, exist_ok=True)
    res = model.run(ids, new=new, dump_tokens=ptokens, progress=True)
    tr, P = res["traces"], len(res["traces"])
    arrays = [("tokens", np.array(ids[:P], dtype=np.int32)),
              ("hidden", res["hidden"].transpose(1, 0, 2).copy()),
              ("l0_attn_in", np.stack([t["attn_in"] for t in tr])),
              ("l0_xq", np.stack([t["xq"] for t in tr])),
              ("l0_sx", np.array([t["sx"] for t in tr], dtype=np.float32)),
              ("l0_qi", np.stack([t["qi"] for t in tr])),
              ("l0_ki", np.stack([t["ki"] for t in tr])),
              ("l0_vi", np.stack([t["vi"] for t in tr])),
              ("l0_q", np.stack([t["q"] for t in tr])),
              ("l0_k", np.stack([t["k"] for t in tr])),
              ("l0_v", np.stack([t["v"] for t in tr])),
              ("l0_attn_out", np.stack([t["attn_out"] for t in tr])),
              ("l0_attn_sub", np.stack([t["attn_sub"] for t in tr])),
              ("l0_aq", np.stack([t["aq"] for t in tr])),
              ("l0_sa", np.array([t["sa"] for t in tr], dtype=np.float32)),
              ("l0_oi", np.stack([t["oi"] for t in tr])),
              ("l0_o", np.stack([t["o"] for t in tr])),
              ("l0_mlp_in", np.stack([t["mlp_in"] for t in tr])),
              ("l0_x2q", np.stack([t["x2q"] for t in tr])),
              ("l0_sx2", np.array([t["sx2"] for t in tr], dtype=np.float32)),
              ("l0_g", np.stack([t["gi"] for t in tr])),
              ("l0_u", np.stack([t["ui"] for t in tr])),
              ("l0_h", np.stack([t["glue"]["h"] for t in tr])),
              ("l0_q8", np.stack([t["glue"]["q8"] for t in tr])),
              ("l0_d", np.stack([t["di"] for t in tr])),
              ("l0_shifts", np.array([[t["glue"]["s_g"], t["glue"]["s_u"], t["glue"]["kg"], t["glue"]["s_S"],
                                       t["glue"]["s_T"], t["glue"]["s_V"], t["glue"]["hmax"], t["glue"]["m"]]
                                      for t in tr], dtype=np.int32)),
              ("l0_absmax_y", np.array([t["absmax_y"] for t in tr], dtype=np.float32)),
              ("l0_ffn", np.stack([t["ffn"] for t in tr])),
              ("final_norm", np.stack(res["fins"][:P])),
              ("logits", np.stack([model.logits(f) for f in res["fins"][:P]]))]
    arrays.append(("logits_bf16", np.stack([model.logits(f, "bf16") for f in res["fins"][:P]])))
    arrays.append(("argmax", np.array([int(np.argmax(a)) for a in dict(arrays)["logits"]], dtype=np.int32)))
    for name, a in arrays:
        np.save(os.path.join(out_dir, f"{name}.npy"), a)
    total = write_stage_bin(os.path.join(out_dir, "stages.bin"), arrays)
    index = {"format": "REFSTG\\x01\\x00, see ref_model.write_stage_bin", "bytes": total,
             "prompt_token_ids": [int(v) for v in ids], "dumped_positions": list(range(P)),
             "head": model.head_mode, "cache_dtype": model.cache_dtype,
             "eps": model.eps, "rope_theta": model.theta,
             "arrays": [{"name": n, "dtype": a.dtype.name, "shape": list(a.shape)} for n, a in arrays],
             "generated_ids": res["gen"],
             "seconds": {"prompt": res["prompt_seconds"], "generate": res["gen_seconds"]}}
    json.dump(index, open(os.path.join(out_dir, "ref_dump.json"), "w"), indent=1)
    return res, arrays, index


def cosine(a, b):
    a, b = np.asarray(a, dtype=np.float64).ravel(), np.asarray(b, dtype=np.float64).ravel()
    n = np.linalg.norm(a) * np.linalg.norm(b)
    return float(a @ b / n) if n else 1.0


def self_test(model=None, seed=5):
    """The claims the reference rests on, none of them needing HF."""
    out = []
    rng = np.random.default_rng(seed)
    for n in range(6):
        Wg = reference.random_ternary(rng, (32, 2560))
        Wu = reference.random_ternary(rng, (32, 2560))
        Wd = reference.random_ternary(rng, (16, 32))
        gamma = (rng.standard_normal(32) * 1.5).astype(np.float32)
        x = (np.full(2560, 127) if n == 0 else np.full(2560, -128) if n == 1 else
             np.zeros(2560) if n == 2 else rng.integers(-128, 128, 2560)).astype(np.int8)
        want = reference.ffn_int(Wg, Wu, Wd, gamma, x)
        got = ffn_glue(reference.matvec_int(Wg, x), reference.matvec_int(Wu, x), gamma)
        for f in ("s_g", "s_u", "kg", "s_S", "s_T", "s_V", "hmax", "m"):
            assert int(getattr(want, f)) == int(got[f]), (n, f, getattr(want, f), got[f])
        assert np.array_equal(want.h, got["h"]) and np.array_equal(want.q, got["q8"])
        assert np.array_equal(want.G, got["G"])
    out.append("ffn_glue == reference.ffn_int's glue on 6 small layers (corners included)")
    for n in range(4):
        W = reference.random_ternary(rng, (257, 6912))
        x = rng.integers(-128, 128, 6912).astype(np.int8)
        got = matvec_i32(W.astype(np.float32), x)
        want = W.astype(np.int64) @ x.astype(np.int64)
        assert np.array_equal(got.astype(np.int64), want), n
    out.append("float32 BLAS matvec == int64 matvec, 4 x [257, 6912] ternary times int8")
    for n in range(4):
        x = (rng.standard_normal(2560) * 3).astype(np.float32)
        q, s = absmax_int8(x)
        assert int(np.abs(q).max()) == 127 and q.dtype == np.int8
        assert abs(float(s) - 127.0 / float(np.abs(x).max())) < 1e-3
    out.append("absmax_int8 puts the largest activation on 127 with scale 127/max|x|")
    v = np.array([1.0, 1.0 + 2 ** -9, 3.14159265, -1e-7, 65504.0], dtype=np.float32)
    assert np.all(np.abs(round_bf16(v) - v) <= np.abs(v) * 2 ** -8 + 1e-12)
    out.append("round_bf16 keeps 8 mantissa bits")
    if model is not None:
        lay = model.layers[0]
        x = (rng.standard_normal(model.hidden) * 0.5).astype(np.float32)
        x2q, sx2 = absmax_int8(x)
        gi = matvec_i32(lay["W"]["gate_proj"], x2q)
        ui = matvec_i32(lay["W"]["up_proj"], x2q)
        gamma = lay["ffn_sub_norm"]
        wsg, wsu = lay["ws"]["gate_proj"], lay["ws"]["up_proj"]
        amax_i, _, Kc, _ = ffn_absmax(gi, ui, gamma, wsg, wsu, sx2, model.eps)
        gf = gi.astype(np.float32) * np.float32(wsg / sx2)
        uf = ui.astype(np.float32) * np.float32(wsu / sx2)
        zf = np.square(np.maximum(gf, 0.0)) * uf
        yf = gamma * zf * np.float32(1.0 / np.sqrt(np.mean(np.square(zf.astype(np.float64))) + model.eps))
        amax_f = float(np.abs(yf).max())
        rel = abs(amax_i - amax_f) / amax_f
        assert rel < 1e-5, (amax_i, amax_f, rel)
        gl = ffn_glue(gi, ui, gamma)
        q8f, _ = absmax_int8(yf)
        same = float(np.mean(gl["q8"] == q8f))
        within = float(np.mean(np.abs(gl["q8"].astype(np.int32) - q8f.astype(np.int32)) <= 1))
        out.append(f"max|y| from the integers vs the float path: relative departure {rel:.3e} (Kc {Kc:.6g}); "
                   f"the glue's q8 vs quantising the float y: {same:.4f} identical, {within:.4f} within one")
        if model.head_mode == "int8":
            fin = (rng.standard_normal(model.hidden) * 0.3).astype(np.float32)
            qh, _ = absmax_int8(fin)
            raw = model.head_f @ qh.astype(np.float32)
            err = float(np.abs(raw - np.rint(raw)).max())
            assert err < 0.25, err
            out.append(f"the int8 head's float32 accumulation lands on integers: max |acc - round(acc)| {err:.3g} "
                       f"(so the reference's logits are the runtime's int32 sums exactly)")
    return out


def top_k(a, k):
    idx = np.argpartition(-a, k)[:k]
    return set(int(i) for i in idx[np.argsort(-a[idx])])


def check(model, hf_dir, report=None, new=32, dump_dir=None, ptokens=8):
    hf = json.load(open(os.path.join(hf_dir, "hf_dump.json"), encoding="utf-8"))
    ids = hf["token_ids"]
    T = len(ids)
    lines = []

    def say(s=""):
        print(s, flush=True)
        lines.append(s)

    say(f"ref_model.py (numpy, float32 on the A53's side, exact int32 matvecs where the engines run) against")
    say(f"transformers {hf['transformers']} (torch {hf['torch']}, {hf['device']}, attn {hf['attn_implementation']}) on {hf['model']}")
    say(f"prompt ({T} tokens): {hf['prompt']!r}")
    say(f"chat text: {hf['chat_text']!r}")
    say(f"token ids: {ids}")
    say(f"reference: head {model.head_mode}, KV cache {model.cache_dtype}, eps {model.eps}, rope_theta {model.theta}, "
        f"weights float32 from {os.path.basename(model.snapshot)} in {model.load_seconds:.0f} s")
    say(f"HF self-checks: attention rebuilt from its own q,k,v max |diff| {hf['checks']['attention_rebuilt_max_abs_diff']:.4g}, "
        f"down_proj from its int8 input max |diff| {hf['checks']['down_from_int8_max_abs_diff']:.4g}")
    say()
    say("COMMANDS (from the repository root, the model read out of the local Hugging Face cache)")
    say(f"  python tools/hf_dump.py --out {hf_dir}")
    say(f"  python tools/ref_model.py --check --hf {hf_dir} "
        f"--report model-agreement.txt --dump refdump --new {new}   (run inside tools/)")
    say(f"  python tools/ref_model.py --self-test")
    say(f"  python tools/ref_model.py --verify refdump")
    say()
    say("SELF-TEST")
    for s in self_test(model):
        say(f"  {s}")
    say()

    t0 = time.time()
    res = model.run(ids, new=0, dump_tokens=T, progress=True)
    say(f"the {T}-token prompt through the reference in {res['prompt_seconds']:.1f} s "
        f"({res['prompt_seconds'] / T * 1000:.0f} ms a token, numpy on the PC)")
    hidden = res["hidden"]
    hf_hidden = np.load(os.path.join(hf_dir, "hf_hidden.npy"))
    say()
    say(f"HIDDEN STATE COSINE, reference vs HF, over all {T} positions "
        f"(layer 0 is the embedding output, layer l+1 the residual stream after layer l)")
    say("  layer   min cosine    mean cosine   max |diff|   ref |h| max")
    per_layer = []
    for l in range(hidden.shape[1]):
        cs = [cosine(hidden[p, l], hf_hidden[l, p]) for p in range(T)]
        md = max(float(np.abs(hidden[p, l] - hf_hidden[l, p]).max()) for p in range(T))
        per_layer.append((l, min(cs), float(np.mean(cs)), md))
        say(f"  {l:5d}   {min(cs):.8f}   {np.mean(cs):.8f}   {md:10.4g}   {float(np.abs(hidden[:, l]).max()):9.4g}")
    say(f"  worst layer cosine {min(c for _, c, _, _ in per_layer):.8f} at layer "
        f"{min(per_layer, key=lambda r: r[1])[0]}")

    say()
    fins = res["fins"]
    hf_fin = np.load(os.path.join(hf_dir, "hf_final_norm.npy"))
    hf_log = np.load(os.path.join(hf_dir, "hf_logits.npy"))
    fin_cos = [cosine(fins[p], hf_fin[p]) for p in range(T)]
    say(f"FINAL NORM cosine over all {T} positions: min {min(fin_cos):.8f} mean {np.mean(fin_cos):.8f}")
    head_numbers = {}
    for mode in ("int8", "bf16") if model.head_mode == "int8" else ("bf16",):
        lg = np.stack([model.logits(f, mode) for f in fins])
        cs = [cosine(lg[p], hf_log[p]) for p in range(T)]
        top1 = float(np.mean([int(np.argmax(lg[p])) == int(np.argmax(hf_log[p])) for p in range(T)]))
        top5 = float(np.mean([len(top_k(lg[p], 5) & top_k(hf_log[p], 5)) / 5 for p in range(T)]))
        top1_5 = float(np.mean([int(np.argmax(lg[p])) in top_k(hf_log[p], 5) for p in range(T)]))
        head_numbers[mode] = (min(cs), top1, top5, top1_5)
        say(f"LOGITS ({mode} head) over all {T} positions: cosine min {min(cs):.8f} mean {np.mean(cs):.8f}; "
            f"top-1 agreement {top1:.4f} ({round(top1 * T)}/{T}); top-5 set overlap {top5:.4f}; "
            f"top-1 inside HF's top-5 {top1_5:.4f}")
        for p in range(T):
            a, b = int(np.argmax(lg[p])), int(np.argmax(hf_log[p]))
            if a != b:
                mine, theirs = np.sort(hf_log[p])[::-1][:2]
                say(f"    position {p} (token {ids[p]}) differs: ref {a}, HF {b}; HF's top two logits "
                    f"{mine:.4f} and {theirs:.4f}, a margin of {mine - theirs:.4f}; HF's logit for the "
                    f"reference's pick {float(hf_log[p][a]):.4f}")
    if model.head_mode == "int8":
        a8 = [int(np.argmax(model.logits(hf_fin[p], "int8"))) for p in range(T)]
        af = [int(np.argmax(model.logits(hf_fin[p], "bf16"))) for p in range(T)]
        ahf = [int(np.argmax(hf_log[p])) for p in range(T)]
        say(f"the int8 head vs the float head on HF's OWN final hidden states: argmax equal "
            f"{sum(int(a == b) for a, b in zip(a8, af))}/{T}; vs HF's own logits {sum(int(a == b) for a, b in zip(a8, ahf))}/{T} "
            f"(int8) and {sum(int(a == b) for a, b in zip(af, ahf))}/{T} (float)")

    say()
    P = min(ptokens, T)
    say(f"LAYER 0, STAGE BY STAGE, the first {P} positions (the reference's integers vs HF's floats)")
    tr = res["traces"][:P]
    l0 = {n: np.load(os.path.join(hf_dir, f"hf_l0_{n}.npy")) for n in
          ("attn_in", "q", "k", "v", "attn_out", "attn_sub", "o", "mlp_in", "g", "u", "h", "q8", "q8_scale", "d")}
    lay = model.layers[0]
    say("  pos  attn_in      q        k        v      attn_out  attn_sub    o       mlp_in      g        u        h      q8 same  q8 <=1     d")
    stage_min = {}
    for p in range(P):
        t = tr[p]
        gf = t["gi"].astype(np.float32) * np.float32(lay["ws"]["gate_proj"] / t["sx2"])
        uf = t["ui"].astype(np.float32) * np.float32(lay["ws"]["up_proj"] / t["sx2"])
        hf_h = l0["h"][p]
        row = {"attn_in": cosine(t["attn_in"], l0["attn_in"][p]), "q": cosine(t["q"], l0["q"][p]),
               "k": cosine(t["k"], l0["k"][p]), "v": cosine(t["v"], l0["v"][p]),
               "attn_out": cosine(t["attn_out"], l0["attn_out"][p]), "attn_sub": cosine(t["attn_sub"], l0["attn_sub"][p]),
               "o": cosine(t["o"], l0["o"][p]), "mlp_in": cosine(t["mlp_in"], l0["mlp_in"][p]),
               "g": cosine(gf, l0["g"][p]), "u": cosine(uf, l0["u"][p]),
               "h": cosine(t["glue"]["h"].astype(np.float64), hf_h),
               "d": cosine(t["ffn"], l0["d"][p])}
        same = float(np.mean(t["glue"]["q8"] == l0["q8"][p]))
        within = float(np.mean(np.abs(t["glue"]["q8"].astype(np.int32) - l0["q8"][p].astype(np.int32)) <= 1))
        for kk, vv in row.items():
            stage_min[kk] = min(stage_min.get(kk, 1.0), vv)
        stage_min["q8_same"] = min(stage_min.get("q8_same", 1.0), same)
        stage_min["q8_within1"] = min(stage_min.get("q8_within1", 1.0), within)
        say(f"  {p:3d} " + " ".join(f"{row[k]:.6f}" for k in
            ("attn_in", "q", "k", "v", "attn_out", "attn_sub", "o", "mlp_in", "g", "u", "h")) +
            f"  {same:.4f}  {within:.4f}  {row['d']:.6f}")
    say("  worst over those positions: " + ", ".join(f"{k} {v:.6f}" for k, v in stage_min.items()))
    say()
    say("  the FFN scale bookkeeping, position by position (max|y| from the integers vs the float path,")
    say("  and the reference's own float FFN output against HF's down_proj output)")
    worst_scale = 0.0
    for p in range(P):
        t = tr[p]
        gf = t["gi"].astype(np.float32) * np.float32(lay["ws"]["gate_proj"] / t["sx2"])
        uf = t["ui"].astype(np.float32) * np.float32(lay["ws"]["up_proj"] / t["sx2"])
        zf = np.square(np.maximum(gf, 0.0)) * uf
        yf = lay["ffn_sub_norm"] * zf * np.float32(1.0 / np.sqrt(np.mean(np.square(zf.astype(np.float64))) + model.eps))
        amax_f, amax_hf = float(np.abs(yf).max()), float(np.abs(l0["h"][p]).max())
        rel_f = abs(t["absmax_y"] - amax_f) / amax_f
        rel_hf = abs(t["absmax_y"] - amax_hf) / amax_hf
        worst_scale = max(worst_scale, rel_f)
        gl = t["glue"]
        say(f"   pos {p}: s_g {gl['s_g']} s_u {gl['s_u']} kg {gl['kg']} (s_S,s_T,s_V) {gl['s_S']},{gl['s_T']},{gl['s_V']} "
            f"hmax {gl['hmax']} m {gl['m']}; max|y| int {t['absmax_y']:.6f} float {amax_f:.6f} HF {amax_hf:.6f}; "
            f"rel vs float {rel_f:.3e}, vs HF {rel_hf:.3e}; ffn out cos {cosine(t['ffn'], l0['d'][p]):.8f}")

    say()
    hf_gen = np.load(os.path.join(hf_dir, "hf_gen_ids.npy")).tolist()
    n_cmp = min(new, len(hf_gen))
    t1 = time.time()
    gres = model.run(ids, new=new, dump_tokens=0)
    ref_gen = gres["gen"]
    match = 0
    first_diff = -1
    for i in range(min(len(ref_gen), n_cmp)):
        if ref_gen[i] == hf_gen[i]:
            match += 1
        else:
            first_diff = i
            break
    else:
        match = min(len(ref_gen), n_cmp)
    say(f"GREEDY GENERATION, {new} tokens asked, HF gave {len(hf_gen)} before EOS, comparing {n_cmp}")
    say(f"  HF : {hf_gen}")
    say(f"  ref: {ref_gen}")
    say(f"  {match}/{n_cmp} tokens identical" + (f", first difference at position {first_diff}" if first_diff >= 0
                                                 else ", no difference"))
    say(f"  HF text : {hf['generated_text'].strip()!r}")
    dec = decoder(model.snapshot)
    if dec:
        say(f"  ref text: {dec(ref_gen).strip()!r}")
        ids_again = tokenize(hf["prompt"], model.snapshot)
        say(f"  the tokenizers library on tokenizer.json with the chat text built by hand reproduces HF's prompt ids: "
            f"{ids_again == ids} ({len(ids_again)} ids)")
    say(f"  ({gres['gen_seconds']:.1f} s for {len(ref_gen)} reference tokens, {time.time() - t1:.1f} s including the prompt)")

    keep = model.cache_dtype
    model.cache_dtype = "bf16"
    bres = model.run(ids, new=new, dump_tokens=P)
    model.cache_dtype = keep
    bgen = bres["gen"]
    bmatch = next((i for i in range(min(len(bgen), n_cmp)) if bgen[i] != hf_gen[i]), -1)
    say(f"  with a bf16 KV cache instead of float32: {min(len(bgen), n_cmp) if bmatch < 0 else bmatch}/{n_cmp} "
        f"identical to HF" + (f", first difference at position {bmatch}" if bmatch >= 0 else ", no difference") +
        f"; against the float32-cache reference: "
        f"{'identical' if bgen == ref_gen else 'first difference at ' + str(next(i for i in range(min(len(bgen), len(ref_gen))) if bgen[i] != ref_gen[i]))}")
    bh = bres["hidden"]
    worst = min(min(cosine(bh[p, l], hidden[p, l]) for p in range(P)) for l in range(bh.shape[1]))
    rel = max(max(float(np.abs(bh[p, l] - hidden[p, l]).max() / max(1e-9, np.abs(hidden[p, l]).max()))
                  for p in range(P)) for l in range(bh.shape[1]))
    say(f"  the same two references against each other over the {P} dumped positions and all "
        f"{bh.shape[1]} stages: worst cosine {worst:.8f}, largest |diff| relative to that stage's own "
        f"max |h| {rel:.3e} (the tolerance a bf16 cache costs a --stage-check)")

    if dump_dir:
        say()
        _, arrays, index = dump_stages(model, ids, dump_dir, ptokens=ptokens, new=0)
        say(f"STAGE DUMPS for the runtime in {dump_dir}: stages.bin ({index['bytes']} bytes), ref_dump.json, and "
            f"one .npy an array, {len(arrays)} arrays over prompt positions {index['dumped_positions']}")
        say("  stages.bin: magic char[8] \"REFSTG\\x01\\x00\"; uint32 count; uint32 data_start; then count entries of")
        say("  72 bytes {char name[32]; uint32 dtype (0 float32, 1 int32, 2 int16, 3 int8); uint32 ndim; uint32 dims[4];")
        say("  uint64 offset; uint64 nbytes}; the arrays follow from data_start (4096-aligned), each at data_start +")
        say("  offset, 64-byte aligned, little endian, C order. runtime/refstages.h reads it.")
        for n, a in arrays:
            say(f"  {n:14s} {a.dtype.name:8s} {tuple(a.shape)}")
        verify_dump(dump_dir)
        say(f"  read back through the binary's own rules: all {len(arrays)} arrays identical to their .npy")
    say()
    say("SUMMARY")
    say(f"  hidden state cosine vs HF over {T} positions and 30 layers: worst {min(c for _, c, _, _ in per_layer):.6f} "
        f"(layer {min(per_layer, key=lambda r: r[1])[0]}), best layer's worst "
        f"{max(c for l, c, _, _ in per_layer if l):.6f}")
    hn = head_numbers[model.head_mode]
    say(f"  logits ({model.head_mode} head): cosine min {hn[0]:.6f}, top-1 {round(hn[1] * T)}/{T}, "
        f"top-5 overlap {hn[2]:.4f}, top-1 inside HF's top-5 {hn[3]:.4f}")
    say(f"  greedy generation: {match}/{n_cmp} tokens identical to HF's" +
        ("" if first_diff < 0 else f", first difference at {first_diff}"))
    say(f"  layer 0 worst stage cosine over 8 positions: " +
        ", ".join(f"{k} {v:.6f}" for k, v in stage_min.items() if not k.startswith("q8_")))
    say(f"  the FFN's max|y| from the integers vs the float path: worst relative departure {worst_scale:.3e}")
    if report:
        open(report, "w", encoding="utf-8").write("\n".join(lines) + "\n")
        print(f"\nwrote {report}")
    return lines


def tokenize(prompt, snapshot):
    """The chat-templated prompt as ids, via tokenizers if it is installed. The template is the
    checkpoint's own: "User: " + the trimmed content + "<|eot_id|>" + "Assistant: ", and no BOS."""
    from tokenizers import Tokenizer
    tk = Tokenizer.from_file(os.path.join(snapshot, "tokenizer.json"))
    text = f"User: {prompt.strip()}<|eot_id|>Assistant: "
    return tk.encode(text, add_special_tokens=False).ids


def decoder(snapshot):
    """A decode(ids) -> str, or None when the tokenizers library is not installed."""
    try:
        from tokenizers import Tokenizer
    except ImportError:
        return None
    tk = Tokenizer.from_file(os.path.join(snapshot, "tokenizer.json"))
    return lambda ids: tk.decode([int(v) for v in ids], skip_special_tokens=True)


def main(argv):
    opts = {"--prompt": PROMPT, "--new": "32", "--hf": None, "--report": None, "--dump": None,
            "--dump-tokens": "8", "--cache-dtype": "f32", "--head": "int8", "--snapshot": None,
            "--tokens": None, "--max-layers": None, "--ask": None}
    flags = set()
    i = 0
    while i < len(argv):
        if argv[i] == "--verify":
            for s in verify_dump(argv[i + 1]):
                print(s)
            return 0
        if argv[i] in ("--self-test", "--check"):
            flags.add(argv[i])
            i += 1
        elif argv[i] in opts:
            opts[argv[i]] = argv[i + 1]
            i += 2
        else:
            raise SystemExit(__doc__)
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    model = Ref(opts["--snapshot"], head=opts["--head"], cache_dtype=opts["--cache-dtype"],
                max_layers=int(opts["--max-layers"]) if opts["--max-layers"] else None)
    if "--self-test" in flags or not (flags or opts["--dump"] or opts["--ask"]):
        for s in self_test(model):
            print(f"  {s}")
    if "--check" in flags:
        if not opts["--hf"]:
            raise SystemExit("--check needs --hf DIR (run hf_dump.py first)")
        check(model, opts["--hf"], report=opts["--report"], new=int(opts["--new"]),
              dump_dir=opts["--dump"], ptokens=int(opts["--dump-tokens"]))
        return 0
    if opts["--dump"]:
        ids = ([int(v) for v in opts["--tokens"].split(",")] if opts["--tokens"]
               else tokenize(opts["--prompt"], model.snapshot))
        _, arrays, index = dump_stages(model, ids, opts["--dump"], ptokens=int(opts["--dump-tokens"]))
        print(f"stage dumps in {opts['--dump']}: {len(arrays)} arrays, {index['bytes']} bytes in stages.bin")
    if opts["--ask"]:
        ids = tokenize(opts["--ask"], model.snapshot)
        res = model.run(ids, new=int(opts["--new"]), progress=True)
        dec = decoder(model.snapshot)
        print(f"Q: {opts['--ask']}")
        print(f"A: {dec(res['gen']).strip() if dec else res['gen']}")
        print(f"ids in ({len(ids)}) {ids}\nids out ({len(res['gen'])}) {res['gen']}")
        print(f"{res['prompt_seconds']:.1f} s prompt, {res['gen_seconds']:.1f} s for {len(res['gen'])} tokens")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
