"""The reference for BitNet b1.58 2B4T's layer 0 MLP on the KV260: the exact integer products the
grids in hardware/grown/ and the ternary engines must reproduce, BitNet's activation quantisation
that feeds them, and the integer FFN (ffn_int) that the glue block and the driver are checked
against.

usage: python reference.py            self-test against numpy's int64 dot and a scalar loop

matvec_int(W, x): the exact int32 W @ x for W int8 in {-1, 0, +1} of shape [out, in] and x int8
of shape [in]; with in <= 6912 and |x| <= 128 every sum is under 2^21, so int32 holds it.
quantize_activations(x): BitNet's per-token absmax quantisation as transformers' ActQuant does
it, scale = 127 / max|x| (clamped at 1e-5), x_q = round(x * scale) (half to even) clipped to
[-128, 127] as int8, returning (x_q, scale). bitlinear(W, weight_scale, x): the layer's float
output, matvec_int(W, x_q) / (scale * weight_scale).

ffn_int(W_gate, W_up, W_down, gamma, x): the integer FFN of one token, spec v2, x int8 [2560]:
  g = W_gate @ x, u = W_up @ x                                  int32 [6912], the engines' sums
  s_g = max(0, bitlen(max relu(g)) - 15), s_u = max(0, bitlen(max |u|) - 15)   (A53)
  a = relu(g) >> s_g, b = |u| >> s_u, both in 0..32767; sgn_u = sign(u)         (shell)
  kg = the largest k in 0..15 with round(max|gamma| 2^k) <= 32767; G = round(gamma 2^kg) int16
  S = a a;     s_S = max(0, bitlen(max S) - 15); S' = S >> s_S                  (grown 16x16, A53)
  T = S' b;    s_T = max(0, bitlen(max T) - 15); T' = T >> s_T
  V = T' |G|;  s_V = max(0, bitlen(max V) - 15); V' = V >> s_V
  h = sgn_u sign(G) V'                                          int16, |h| < 2^15; hmax = max |h|
  m = 65535 if hmax == 0 else min(65535, floor(127 2^15 / hmax))                (A53)
  P = |h| m; q = sign(h) min(127, (P + 2^14) >> 15)             int8 [6912]     (grown 16x16)
  d = W_down @ q                                                int32 [2560], the engines
returning FfnInt(g, u, s_g, s_u, kg, G, s_S, s_T, s_V, h, hmax, m, q, d). Every shift truncates
(floor), every round is half to even, as numpy's, and bitlen(0) = 0. Every multiply of the chain
is a 16 x 16 unsigned one: a, b, S', T', V', |G| and |h| are all under 2^15 and m under 2^16,
asserted on every call.

ffn_int_fixed(...): the old design, the three stage shifts fixed at 15 (its FfnInt carries
s_S = s_T = s_V = 15); compare.py reports both. ffn_int_scaled is ffn_int.
"""
import os
import sys
from typing import NamedTuple

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
HIDDEN = 2560
INTER = 6912


class FfnInt(NamedTuple):
    g: np.ndarray
    u: np.ndarray
    s_g: int
    s_u: int
    kg: int
    G: np.ndarray
    s_S: int
    s_T: int
    s_V: int
    h: np.ndarray
    hmax: int
    m: int
    q: np.ndarray
    d: np.ndarray


FfnIntScaled = FfnInt


def load(here=HERE):
    W = np.load(os.path.join(here, "W_gate_proj.npy"))
    scale = np.load(os.path.join(here, "scale.npy"))
    return W, float(scale[0])


def load_ffn(here=HERE):
    """(W_gate [6912, 2560], W_up [6912, 2560], W_down [2560, 6912], gamma float32 [6912]) as fetch.py wrote them."""
    W_gate = np.load(os.path.join(here, "W_gate_proj.npy"))
    W_up = np.load(os.path.join(here, "W_up_proj.npy"))
    W_down = np.load(os.path.join(here, "W_down_proj.npy"))
    gamma = np.load(os.path.join(here, "gamma.npy"))
    assert W_gate.shape == W_up.shape == (INTER, HIDDEN) and W_down.shape == (HIDDEN, INTER), (W_gate.shape, W_up.shape, W_down.shape)
    assert gamma.shape == (INTER,) and gamma.dtype == np.float32, (gamma.shape, gamma.dtype)
    return W_gate, W_up, W_down, gamma


def matvec_int(W, x):
    """The exact int32 W @ x for int8 ternary W [out, in] and int8 x [in]."""
    if W.dtype != np.int8 or x.dtype != np.int8:
        raise TypeError("matvec_int wants int8 W and int8 x")
    if x.shape != (W.shape[1],):
        raise ValueError(f"x has shape {x.shape}, W has {W.shape[1]} columns")
    return W.astype(np.int32) @ x.astype(np.int32)


def quantize_activations(x_float, num_bits=8):
    """BitNet's per-token absmax int8 quantisation: (x_q int8, scale) with x ~= x_q / scale."""
    x = np.asarray(x_float, dtype=np.float32)
    qn, qp = -(2 ** (num_bits - 1)), 2 ** (num_bits - 1) - 1
    absmax = np.maximum(np.abs(x).max(axis=-1, keepdims=True), 1e-5)
    scale = qp / absmax
    x_q = np.clip(np.round(x * scale), qn, qp).astype(np.int8)
    return x_q, scale


def bitlinear(W, weight_scale, x_float):
    """BitLinear.forward on one token: the float output of the layer for a float input row."""
    x_q, scale = quantize_activations(x_float)
    return matvec_int(W, x_q).astype(np.float32) / (scale * weight_scale)


def bitlen(v):
    """The number of bits of a non-negative integer, bitlen(0) = 0."""
    v = int(v)
    if v < 0:
        raise ValueError("bitlen of a negative")
    return v.bit_length()


def gamma_fixed(gamma):
    """(kg, G int16): kg the largest k in 0..15 with round(max|gamma| 2^k) <= 32767, G = round(gamma 2^kg), half to even."""
    amax = float(np.abs(np.asarray(gamma, dtype=np.float64)).max())
    kg = 0
    for k in range(16):
        if round(amax * 2 ** k) <= 32767:
            kg = k
    G = np.round(np.asarray(gamma, dtype=np.float64) * 2 ** kg).astype(np.int64)
    assert np.abs(G).max() <= 32767
    return kg, G.astype(np.int16)


def stage_shift(values, scaled=True):
    """The shift of one product stage: bitlen(max) - 15 floored at 0 (spec v2), 15 in the old fixed design."""
    return max(0, bitlen(int(values.max())) - 15) if scaled else 15


def assert_16x16(a, b, S1, T1, V1, abs_G, abs_h, m):
    """Every multiply of the chain (a a, S' b, T' |G|, |h| m) is 16 x 16 unsigned: the named
    operands under 2^15, m under 2^16."""
    for name, v in (("a", a), ("b", b), ("S'", S1), ("T'", T1), ("V'", V1), ("|G|", abs_G), ("|h|", abs_h)):
        lo, hi = (min(v), max(v)) if isinstance(v, list) else (int(v.min()), int(v.max()))
        assert 0 <= lo and hi < 1 << 15, (name, lo, hi)
    assert 0 <= m < 1 << 16, m


def _ffn(W_gate, W_up, W_down, gamma, x, scaled):
    if x.dtype != np.int8 or x.shape != (W_gate.shape[1],):
        raise TypeError(f"ffn_int wants int8 x of shape ({W_gate.shape[1]},), got {x.dtype} {x.shape}")
    g = matvec_int(W_gate, x)
    u = matvec_int(W_up, x)
    g64, u64 = g.astype(np.int64), u.astype(np.int64)
    relu_g = np.maximum(g64, 0)
    abs_u = np.abs(u64)
    s_g = max(0, bitlen(relu_g.max()) - 15)
    s_u = max(0, bitlen(abs_u.max()) - 15)
    a = relu_g >> s_g
    b = abs_u >> s_u
    kg, G = gamma_fixed(gamma)
    G64 = G.astype(np.int64)
    abs_G = np.abs(G64)
    S = a * a
    s_S = stage_shift(S, scaled)
    S1 = S >> s_S
    T = S1 * b
    s_T = stage_shift(T, scaled)
    T1 = T >> s_T
    V = T1 * abs_G
    s_V = stage_shift(V, scaled)
    V1 = V >> s_V
    h = np.sign(u64) * np.sign(G64) * V1
    abs_h = np.abs(h)
    hmax = int(abs_h.max())
    m = 65535 if hmax == 0 else min(65535, (127 << 15) // hmax)
    assert_16x16(a, b, S1, T1, V1, abs_G, abs_h, m)
    P = abs_h * m
    q = (np.sign(h) * np.minimum(127, (P + (1 << 14)) >> 15)).astype(np.int8)
    d = matvec_int(W_down, q)
    return FfnInt(g, u, s_g, s_u, kg, G, s_S, s_T, s_V, h.astype(np.int16), hmax, m, q, d)


def ffn_int(W_gate, W_up, W_down, gamma, x):
    """Spec v2: the integer FFN of one token as the docstring fixes it; every intermediate in int64."""
    return _ffn(W_gate, W_up, W_down, gamma, x, True)


def ffn_int_fixed(W_gate, W_up, W_down, gamma, x):
    """The old design: the three stage shifts fixed at 15 (s_S = s_T = s_V = 15 in the result)."""
    return _ffn(W_gate, W_up, W_down, gamma, x, False)


ffn_int_scaled = ffn_int


def ffn_int_scalar(W_gate, W_up, W_down, gamma, x, rows=None, scaled=True):
    """The same FFN (spec v2, or the old fixed design with scaled=False) by a scalar loop over
    Python ints (the matvecs over every row unless `rows` names the rows of each projection to
    compute: {"gate": [...], "up": [...], "down": [...]} with the rest taken from ffn_int, which
    is then only checked on those rows)."""
    xs = [int(v) for v in x]
    fast = _ffn(W_gate, W_up, W_down, gamma, x, scaled) if rows else None

    def dot(row, vec):
        return sum(int(w) * v for w, v in zip(row, vec))

    def matvec(W, vec, key, fallback):
        if rows is None:
            return [dot(W[i], vec) for i in range(W.shape[0])]
        out = [int(v) for v in fallback]
        for i in rows[key]:
            out[i] = dot(W[i], vec)
        return out

    def shift(values):
        return max(0, max(values).bit_length() - 15) if scaled else 15

    g = matvec(W_gate, xs, "gate", fast.g if fast else None)
    u = matvec(W_up, xs, "up", fast.u if fast else None)
    n = len(g)
    relu_g = [v if v > 0 else 0 for v in g]
    abs_u = [abs(v) for v in u]
    s_g = max(0, max(relu_g).bit_length() - 15)
    s_u = max(0, max(abs_u).bit_length() - 15)
    amax = max(abs(float(v)) for v in gamma)
    kg = max(k for k in range(16) if round(amax * 2 ** k) <= 32767)
    G = [int(round(float(v) * 2 ** kg)) for v in gamma]
    a = [relu_g[i] >> s_g for i in range(n)]
    b = [abs_u[i] >> s_u for i in range(n)]
    S = [a[i] * a[i] for i in range(n)]
    s_S = shift(S)
    S1 = [v >> s_S for v in S]
    T = [S1[i] * b[i] for i in range(n)]
    s_T = shift(T)
    T1 = [v >> s_T for v in T]
    V = [T1[i] * abs(G[i]) for i in range(n)]
    s_V = shift(V)
    V1 = [v >> s_V for v in V]
    h = []
    for i in range(n):
        sgn_u = (u[i] > 0) - (u[i] < 0)
        sgn_G = (G[i] > 0) - (G[i] < 0)
        h.append(sgn_u * sgn_G * V1[i])
    hmax = max(abs(v) for v in h)
    m = 65535 if hmax == 0 else min(65535, (127 * 32768) // hmax)
    assert_16x16(a, b, S1, T1, V1, [abs(v) for v in G], [abs(v) for v in h], m)
    q = []
    for v in h:
        P = abs(v) * m
        mag = min(127, (P + 16384) >> 15)
        q.append(mag if v > 0 else -mag if v < 0 else 0)
    d = matvec(W_down, q, "down", fast.d if fast else None)
    return FfnInt(np.array(g, dtype=np.int32), np.array(u, dtype=np.int32), s_g, s_u, kg, np.array(G, dtype=np.int16),
                  s_S, s_T, s_V, np.array(h, dtype=np.int16), hmax, m, np.array(q, dtype=np.int8), np.array(d, dtype=np.int32))


def same_ffn(fast, slow):
    """Every field of two FfnInt equal, with the dtypes ffn_int promises."""
    assert type(fast) is type(slow), (type(fast), type(slow))
    assert fast.g.dtype == np.int32 and fast.u.dtype == np.int32 and fast.d.dtype == np.int32, (fast.g.dtype, fast.u.dtype, fast.d.dtype)
    assert fast.G.dtype == np.int16 and fast.h.dtype == np.int16 and fast.q.dtype == np.int8, (fast.G.dtype, fast.h.dtype, fast.q.dtype)
    for name in fast._fields:
        a, b = getattr(fast, name), getattr(slow, name)
        if isinstance(a, np.ndarray):
            assert np.array_equal(a.astype(np.int64), b.astype(np.int64)), f"{name} differs at {np.flatnonzero(a.astype(np.int64) != b.astype(np.int64))[:8]}"
        else:
            assert int(a) == int(b), (name, a, b)


def random_ternary(rng, shape, density=0.62):
    W = rng.integers(-1, 2, size=shape, dtype=np.int64)
    W[rng.random(shape) > density] = 0
    return W.astype(np.int8)


def self_test(trials=8, seed=7):
    W, weight_scale = load()
    assert W.shape == (INTER, HIDDEN) and W.dtype == np.int8, (W.shape, W.dtype)
    assert set(np.unique(W).tolist()) <= {-1, 0, 1}
    rng = np.random.default_rng(seed)
    for n in range(trials):
        x = rng.integers(-128, 128, size=W.shape[1], dtype=np.int64).astype(np.int8)
        got = matvec_int(W, x)
        want = W.astype(np.int64) @ x.astype(np.int64)
        assert got.dtype == np.int32 and np.array_equal(got.astype(np.int64), want), f"trial {n} disagrees"
        row = int(rng.integers(W.shape[0]))
        by_hand = sum(int(w) * int(v) for w, v in zip(W[row], x))
        assert by_hand == int(got[row]), (row, by_hand, int(got[row]))
    x_f = rng.standard_normal(W.shape[1]).astype(np.float32) * 3
    x_q, scale = quantize_activations(x_f)
    assert x_q.dtype == np.int8 and x_q.min() >= -128 and x_q.max() <= 127
    assert int(np.abs(x_q).max()) == 127, "absmax quantisation puts the largest activation on 127"
    assert np.allclose(x_q / scale, x_f, atol=0.5 / scale)
    y = bitlinear(W, weight_scale, x_f)
    y_slow = (W.astype(np.float32) / weight_scale) @ (x_q.astype(np.float32) / scale)
    assert np.allclose(y, y_slow, rtol=1e-4, atol=1e-3)
    return W, weight_scale, trials


def ffn_self_test(seed=11, small_trials=6):
    """ffn_int (spec v2) and ffn_int_fixed against the scalar loop: whole on small random layers
    with 2560 inputs (so the sums reach the shifts), and on the real layer 0 for a random x and
    the corner activations, the matvecs checked on sampled rows. Returns what was checked, for
    the printout."""
    assert ffn_int_scaled is ffn_int and FfnIntScaled is FfnInt
    rng = np.random.default_rng(seed)
    checked = []
    for n in range(small_trials):
        W_gate = random_ternary(rng, (32, HIDDEN))
        W_up = random_ternary(rng, (32, HIDDEN))
        W_down = random_ternary(rng, (16, 32))
        gamma = (rng.standard_normal(32) * 1.5).astype(np.float32)
        gamma[rng.integers(32)] = 0.0
        if n == 0:
            x = np.full(HIDDEN, 127, dtype=np.int8)
            W_gate[:] = 1
            W_up[:] = -1
        elif n == 1:
            x = np.full(HIDDEN, -128, dtype=np.int8)
        elif n == 2:
            x = np.zeros(HIDDEN, dtype=np.int8)
        else:
            x = rng.integers(-128, 128, size=HIDDEN, dtype=np.int64).astype(np.int8)
        fast = ffn_int(W_gate, W_up, W_down, gamma, x)
        slow = ffn_int_scalar(W_gate, W_up, W_down, gamma, x)
        same_ffn(fast, slow)
        assert fast.hmax == 0 or fast.hmax >= 16384 or fast.s_V == 0, (fast.hmax, fast.s_V)
        assert 0 <= fast.s_S <= 15 and 0 <= fast.s_T <= 15 and 0 <= fast.s_V <= 15, (fast.s_S, fast.s_T, fast.s_V)
        checked.append(("small v2", n, fast.s_g, fast.s_u, fast.kg, fast.s_S, fast.s_T, fast.s_V, fast.hmax, fast.m))
        fast = ffn_int_fixed(W_gate, W_up, W_down, gamma, x)
        slow = ffn_int_scalar(W_gate, W_up, W_down, gamma, x, scaled=False)
        same_ffn(fast, slow)
        assert (fast.s_S, fast.s_T, fast.s_V) == (15, 15, 15)
        checked.append(("small fixed", n, fast.s_g, fast.s_u, fast.kg, fast.s_S, fast.s_T, fast.s_V, fast.hmax, fast.m))
    assert gamma_fixed(np.array([4.5625], dtype=np.float32)) == (12, np.array([18688], dtype=np.int16))
    kg, G = gamma_fixed(np.array([1.0, -0.5, 0.00001], dtype=np.float32))
    assert kg == 14 and G.tolist() == [16384, -8192, 0], (kg, G)
    kg, G = gamma_fixed(np.array([0.0, 0.0], dtype=np.float32))
    assert kg == 15 and G.tolist() == [0, 0], (kg, G)
    kg, G = gamma_fixed(np.array([32767.4], dtype=np.float32))
    assert kg == 0 and G.tolist() == [32767], (kg, G)
    assert gamma_fixed(np.array([0.5 * 2 ** -14 * 3], dtype=np.float32))[0] == 15
    kg, G = gamma_fixed(np.array([2.5 / 2 ** 13, 3.5 / 2 ** 13, 1.0], dtype=np.float32))
    assert kg == 14 and G.tolist() == [5, 7, 16384], (kg, G)
    kg, G = gamma_fixed(np.array([2.5 / 2 ** 14, 3.5 / 2 ** 14, 1.0], dtype=np.float32))
    assert kg == 14 and G.tolist() == [2, 4, 16384], (kg, G)
    for ffn in (ffn_int, ffn_int_fixed):
        small = ffn(np.ones((4, 8), dtype=np.int8), np.ones((4, 8), dtype=np.int8), np.ones((2, 4), dtype=np.int8),
                    np.array([1, 1, 1, 1], dtype=np.float32), np.zeros(8, dtype=np.int8))
        assert small.hmax == 0 and small.m == 65535 and small.q.tolist() == [0, 0, 0, 0] and small.d.tolist() == [0, 0]
        assert (small.s_S, small.s_T, small.s_V) == ((0, 0, 0) if ffn is ffn_int else (15, 15, 15))
    W1 = np.ones((1, 8), dtype=np.int8)
    one = ffn_int(W1, W1, np.ones((1, 1), dtype=np.int8), np.array([1.0], dtype=np.float32), np.full(8, 127, dtype=np.int8))
    assert one.g.tolist() == [1016] and one.u.tolist() == [1016] and one.s_g == 0 and one.s_u == 0 and one.kg == 14
    assert (one.s_S, one.s_T, one.s_V) == (5, 10, 14), (one.s_S, one.s_T, one.s_V)
    assert one.h.tolist() == [32005] and one.hmax == 32005 and one.m == 130 and one.q.tolist() == [127] and one.d.tolist() == [127]
    fixed = ffn_int_fixed(W1, W1, np.ones((1, 1), dtype=np.int8), np.array([1.0], dtype=np.float32), np.full(8, 127, dtype=np.int8))
    assert fixed.h.tolist() == [0] and fixed.hmax == 0 and fixed.m == 65535 and fixed.q.tolist() == [0] and fixed.d.tolist() == [0], fixed.h
    if all(os.path.exists(os.path.join(HERE, f)) for f in ("W_gate_proj.npy", "W_up_proj.npy", "W_down_proj.npy", "gamma.npy")):
        W_gate, W_up, W_down, gamma = load_ffn()
        rows = {"gate": [0, 28, 1550, 6911], "up": [1, 28, 1550, 6911], "down": [0, 751, 1700, 2559]}
        for n, x in enumerate((rng.integers(-128, 128, size=HIDDEN, dtype=np.int64).astype(np.int8),
                               np.full(HIDDEN, 127, dtype=np.int8), np.full(HIDDEN, -128, dtype=np.int8))):
            fast = ffn_int(W_gate, W_up, W_down, gamma, x)
            slow = ffn_int_scalar(W_gate, W_up, W_down, gamma, x, rows)
            same_ffn(fast, slow)
            assert fast.g.tolist() == matvec_int(W_gate, x).tolist()
            assert fast.hmax == 0 or fast.hmax >= 16384 or fast.s_V == 0, (fast.hmax, fast.s_V)
            checked.append(("layer0 v2", n, fast.s_g, fast.s_u, fast.kg, fast.s_S, fast.s_T, fast.s_V, fast.hmax, fast.m))
            fast = ffn_int_fixed(W_gate, W_up, W_down, gamma, x)
            slow = ffn_int_scalar(W_gate, W_up, W_down, gamma, x, rows, scaled=False)
            same_ffn(fast, slow)
            checked.append(("layer0 fixed", n, fast.s_g, fast.s_u, fast.kg, fast.s_S, fast.s_T, fast.s_V, fast.hmax, fast.m))
    return checked


def main():
    W, weight_scale, trials = self_test()
    counts = {v: int((W == v).sum()) for v in (-1, 0, 1)}
    dens = (W != 0).sum(axis=1) / W.shape[1]
    print(f"W_gate_proj {W.shape} int8; weight_scale {weight_scale}; matvec_int agrees with int64 dot on {trials} random x")
    print(f"histogram -1: {counts[-1]}  0: {counts[0]}  +1: {counts[1]}")
    print(f"nonzero density per row: mean {dens.mean():.4f} min {dens.min():.4f} max {dens.max():.4f}")
    x = np.random.default_rng(1).integers(-128, 128, size=W.shape[1], dtype=np.int64).astype(np.int8)
    y = matvec_int(W, x)
    print(f"one random int8 x: y range [{y.min()}, {y.max()}], y[:8] = {y[:8].tolist()}")
    checked = ffn_self_test()
    print(f"ffn_int (spec v2) and ffn_int_fixed agree with the scalar loop on {len(checked)} cases (kind, n, s_g, s_u, kg, s_S, s_T, s_V, hmax, m):")
    for c in checked:
        print(f"  {c}")


if __name__ == "__main__":
    sys.exit(main())
