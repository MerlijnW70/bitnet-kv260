"""Test vectors for hardware/ternary_glue.v, the FFN glue block, in the beat formats it eats.

usage: python gen_glue_vectors.py        (from anywhere)

Writes hardware/vectors/glue_<set>_{a_in,a_h,b_in,b_q,words}.hex for these sets:
  rand    6912 elements: a random ternary layer (W_gate, W_up 6912 x 2560, W_down 2560 x 6912, gamma
          normal, one zero; x random int8; seed 20260908) through reference.ffn_int, spec v2
  tok0    6912 elements: layer 0 of BitNet b1.58 2B4T on the DATA agent's token 0 (x_0.bin, the
          .npy weights, gamma.npy) through reference.ffn_int, checked against expect_0.json; written
          only when those files are there
  small8  the first 8 elements of rand with rand's shifts and m (the glue applies what it is given)
  one     the first element of rand alone (pass A of N = 1) and the first four (pass B of N = 4)
  ext<i>  64 elements each, i = 0..EXT-1, shifts and m from EXT_SETS below: every shift 0; every
          shift 31; one stage at 14, 15, 16, 17 or 31 with the others 0 (15 = the first shift whose
          capture window needs the zero fill); s_g = s_u = 16; random shifts; m = 0 and m = 1. The
          operands are drawn so the shifted operands stay alive: a and b random 16-bit values placed
          at bits s_g and s_u of g and |u|, G random int16, elements 0 and 1 the corners 0 and (1, -1, 1).
The ext sets exercise the shell with shifts the spec never produces; their expected values come from
glue_model below, which is checked against reference.ffn_int element by element on rand and tok0.

Formats (fixed, shared with the glue and the driver):
  pass A input  one element a beat, 128 bits: bits 31..0 g (int32), 63..32 u (int32), 79..64 G (int16),
                the rest nought
  pass A output one element a beat, 32 bits: h sign-extended to int32
  pass B input  four elements a beat, 128 bits: element 4j + e in bits 32e + 31 .. 32e (h as int32), the
                pass A output buffer read back as is
  pass B output four elements a beat, 32 bits: q of element 4j + e as int8 in bits 8e + 7 .. 8e
  params        bits 4..0 s_g, 9..5 s_u, 14..10 s_S, 19..15 s_T, 24..20 s_V;  mval bits 15..0 m
  words file    four lines: N, the params word, m, hmax & 0x7fff (what status bits 14..0 show)
The .hex files hold one beat a line, most significant digit first, so $readmemh gives the beat directly.

glue_model(g, u, G, shifts, m): what the shell computes element by element, with the truncations the
fabric makes (16-bit operands, |u| and |G| as 31- and 15-bit negations):
  a = (relu(g) >> s_g) & 0xffff,  b = ((|u| & 0x7fffffff) >> s_u) & 0xffff,  |G| & 0x7fff
  S = a a, S' = (S >> s_S) & 0xffff;  T = S' b, T' = (T >> s_T) & 0xffff;  V = T' |G|, V' = (V >> s_V) & 0xffff
  h = -V' if (u < 0) xor (G < 0) else V';  hmax = max V'
  P = V' m;  q = sign(h) min(127, (P + 2^14) >> 15)
On operands the spec produces (everything under 2^15) this is reference.ffn_int's h, hmax and q exactly.
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "vectors")
LAYER0 = os.environ.get("BITNET_LAYER0_DIR", os.path.join(HERE, "..", "tools"))
sys.path.insert(0, os.path.join(HERE, "..", "tools"))
sys.path.insert(0, HERE)
import reference

N_FULL = 6912
M16 = 0xFFFF


def glue_model(g, u, G, shifts, m):
    """(h int64 [n], q int64 [n], hmax) as the shell computes them for int64 g, u, G and shifts (s_g, s_u, s_S, s_T, s_V)."""
    s_g, s_u, s_S, s_T, s_V = shifts
    g = np.asarray(g, dtype=np.int64)
    u = np.asarray(u, dtype=np.int64)
    G = np.asarray(G, dtype=np.int64)
    a = (np.maximum(g, 0) >> s_g) & M16
    b = ((np.abs(u) & 0x7FFFFFFF) >> s_u) & M16
    absG = np.abs(G) & 0x7FFF
    neg = (u < 0) ^ (G < 0)
    S1 = ((a * a) >> s_S) & M16
    T1 = ((S1 * b) >> s_T) & M16
    V1 = ((T1 * absG) >> s_V) & M16
    h = np.where(neg, -V1, V1)
    qm = np.minimum(127, ((V1 * m) + (1 << 14)) >> 15)
    q = np.where(neg, -qm, qm)
    return h, q, int(V1.max()) if len(V1) else 0


def params_word(shifts):
    s_g, s_u, s_S, s_T, s_V = shifts
    for s in shifts:
        assert 0 <= s <= 31, shifts
    return s_g | (s_u << 5) | (s_S << 10) | (s_T << 15) | (s_V << 20)


def a_beats(g, u, G):
    return [f"{((int(Gi) & 0xFFFF) << 64) | ((int(ui) & 0xFFFFFFFF) << 32) | (int(gi) & 0xFFFFFFFF):032x}" for gi, ui, Gi in zip(g, u, G)]


def a_out(h):
    return [f"{int(v) & 0xFFFFFFFF:08x}" for v in h]


def b_beats(h):
    lines = []
    for j in range(0, len(h), 4):
        word = 0
        for e, v in enumerate(h[j:j + 4]):
            word |= (int(v) & 0xFFFFFFFF) << (32 * e)
        lines.append(f"{word:032x}")
    return lines


def b_out(q):
    lines = []
    for j in range(0, len(q), 4):
        word = 0
        for e, v in enumerate(q[j:j + 4]):
            word |= (int(v) & 0xFF) << (8 * e)
        lines.append(f"{word:08x}")
    return lines


def write_set(name, g, u, G, shifts, m, h=None, q=None, hmax=None):
    """Write the five files of one set; h, q, hmax from glue_model unless given (then checked against it)."""
    mh, mq, mhmax = glue_model(g, u, G, shifts, m)
    if h is not None:
        assert np.array_equal(mh, np.asarray(h, dtype=np.int64)), f"{name}: glue_model's h differs from the reference at {np.flatnonzero(mh != h)[:8]}"
        assert np.array_equal(mq, np.asarray(q, dtype=np.int64)), f"{name}: glue_model's q differs from the reference at {np.flatnonzero(mq != q)[:8]}"
        assert mhmax == hmax, (name, mhmax, hmax)
    h, q, hmax = mh, mq, mhmax
    n = len(g)
    os.makedirs(OUT, exist_ok=True)
    files = {
        "a_in": a_beats(g, u, G),
        "a_h": a_out(h),
        "b_in": b_beats(h),
        "b_q": b_out(q),
        "words": [f"{n:08x}", f"{params_word(shifts):08x}", f"{m:08x}", f"{hmax & 0x7FFF:08x}"],
    }
    for suffix, lines in files.items():
        with open(os.path.join(OUT, f"glue_{name}_{suffix}.hex"), "w") as f:
            f.write("\n".join(lines) + "\n")
    print(f"{name}: N {n}, shifts {shifts}, m {m}, hmax {hmax}, h in [{h.min()}, {h.max()}], q in [{q.min()}, {q.max()}], "
          f"q nonzero {int((q != 0).sum())}")
    return h, q


def from_reference(name, ffn, G, n=None):
    """A set from an FfnInt result: its g, u, G, shifts and m, its h, q and hmax checked against glue_model."""
    shifts = (ffn.s_g, ffn.s_u, ffn.s_S, ffn.s_T, ffn.s_V)
    sl = slice(0, n)
    return write_set(name, ffn.g[sl], ffn.u[sl], G[sl], shifts, ffn.m,
                     ffn.h[sl].astype(np.int64), ffn.q[sl].astype(np.int64), int(np.abs(ffn.h[sl].astype(np.int64)).max()))


EXT_SETS = (
    ("every shift 0", (0, 0, 0, 0, 0), 65535),
    ("every shift 31", (31, 31, 31, 31, 31), 1),
    ("s_S 14", (0, 0, 14, 0, 0), None), ("s_S 15", (0, 0, 15, 0, 0), None), ("s_S 16", (0, 0, 16, 0, 0), None),
    ("s_S 17", (0, 0, 17, 0, 0), None), ("s_S 31", (0, 0, 31, 0, 0), None),
    ("s_T 14", (0, 0, 0, 14, 0), None), ("s_T 15", (0, 0, 0, 15, 0), None), ("s_T 16", (0, 0, 0, 16, 0), None),
    ("s_T 17", (0, 0, 0, 17, 0), None), ("s_T 31", (0, 0, 0, 31, 0), None),
    ("s_V 14", (0, 0, 0, 0, 14), None), ("s_V 15", (0, 0, 0, 0, 15), None), ("s_V 16", (0, 0, 0, 0, 16), None),
    ("s_V 17", (0, 0, 0, 0, 17), None), ("s_V 31", (0, 0, 0, 0, 31), None),
    ("s_g s_u 16", (16, 16, 0, 0, 0), None),
    ("random", None, None), ("random", None, None), ("random", None, None),
    ("m 0", (0, 0, 0, 0, 0), 0), ("m 1", (0, 0, 2, 2, 2), 1),
)


def placed_elements(rng, n, s_g, s_u):
    """n elements whose shifted operands are alive: a and b random 16-bit values at bits s_g of g and
    s_u of |u| (masked to fit int32), G random int16, u's sign random, corners in the first two."""
    a = rng.integers(0, 1 << 16, size=n, dtype=np.int64) & ((1 << max(0, 31 - s_g)) - 1)
    b = rng.integers(0, 1 << 16, size=n, dtype=np.int64) & ((1 << max(0, 31 - s_u)) - 1)
    g = a << s_g
    u = np.where(rng.random(n) < 0.5, -(b << s_u), b << s_u)
    G = rng.integers(-32768, 32768, size=n, dtype=np.int64)
    g[0], u[0], G[0] = 0, 0, 0
    g[1], u[1], G[1] = 1, -1, 1
    return g, u, G


def ext_sets(rng, n=64):
    for i, (label, shifts, m) in enumerate(EXT_SETS):
        if shifts is None:
            shifts = tuple(int(v) for v in rng.integers(0, 32, size=5))
        if m is None:
            m = int(rng.integers(0, 65536)) if rng.random() < 0.5 else int(rng.integers(0, 400))
        g, u, G = placed_elements(rng, n, shifts[0], shifts[1])
        print(f"ext{i} ({label}): ", end="")
        write_set(f"ext{i}", g, u, G, shifts, m)


def main():
    rng = np.random.default_rng(20260908)
    W_gate = reference.random_ternary(rng, (N_FULL, reference.HIDDEN))
    W_up = reference.random_ternary(rng, (N_FULL, reference.HIDDEN))
    W_down = reference.random_ternary(rng, (reference.HIDDEN, N_FULL))
    gamma = (rng.standard_normal(N_FULL) * 1.5).astype(np.float32)
    gamma[17] = 0.0
    x = rng.integers(-128, 128, size=reference.HIDDEN, dtype=np.int64).astype(np.int8)
    ffn = reference.ffn_int(W_gate, W_up, W_down, gamma, x)
    G = ffn.G.astype(np.int64)
    from_reference("rand", ffn, G)
    from_reference("small8", ffn, G, 8)
    from_reference("one", ffn, G, 4)
    from_reference("one1", ffn, G, 1)

    needed = ["W_gate_proj.npy", "W_up_proj.npy", "W_down_proj.npy", "gamma.npy", "x_0.bin"]
    if all(os.path.exists(os.path.join(LAYER0, f)) for f in needed):
        Wg, Wu, Wd, gm = reference.load_ffn(LAYER0)
        x0 = np.fromfile(os.path.join(LAYER0, "x_0.bin"), dtype=np.int8)
        assert x0.shape == (reference.HIDDEN,), x0.shape
        tok = reference.ffn_int(Wg, Wu, Wd, gm, x0)
        expect = os.path.join(LAYER0, "expect_0.json")
        if os.path.exists(expect):
            want = json.load(open(expect))
            for key in ("s_g", "s_u", "kg", "s_S", "s_T", "s_V", "hmax", "m"):
                assert int(getattr(tok, key)) == int(want[key]), (key, getattr(tok, key), want[key])
            print(f"tok0 agrees with expect_0.json on s_g s_u kg s_S s_T s_V hmax m")
        from_reference("tok0", tok, tok.G.astype(np.int64))
    else:
        print("tok0 skipped: " + ", ".join(f for f in needed if not os.path.exists(os.path.join(LAYER0, f)))
              + f" missing from {LAYER0}. Get the weights with:  python3 ../tools/fetch.py ../tools\n"
                "(x_0.bin is a research artefact and is not in this repository; the tok0 vectors\n"
                " already in vectors/ are left untouched, so the testbench still runs them)")

    ext_sets(np.random.default_rng(20260909))
    print(f"wrote {os.path.abspath(OUT)}: {len(EXT_SETS)} ext sets")
    return 0


if __name__ == "__main__":
    sys.exit(main())
