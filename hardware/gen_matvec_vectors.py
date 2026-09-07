"""Test vectors for hardware/ternary_matvec.v, in the stream formats the engine eats.

usage: python gen_matvec_vectors.py             (from anywhere)
       python gen_matvec_vectors.py --k 6912

Writes hardware/vectors/<set>_act.hex, <set>_w.hex, <set>_y.hex for four sets:
  rand32   32 neurons of a random ternary matrix and one random int8 x   (seed 20260905)
  gate64   rows 0..63 of BitNet's layer-0 W_gate_proj and one random int8 x  (seed 20260906)
  batch32  8 neurons of a random ternary matrix, K = 2560, and four random int8 x (seed 20260908)
  bk6912   4 neurons of a random ternary matrix, K = 6912, and four random int8 x (seed 20260909)
or, with --k K, for one set of K inputs a neuron (6912 is down_proj's):
  k<K>     8 neurons of a random ternary matrix with K inputs and one random int8 x (seed 20260907)
and the expected sums from tools/reference.py's matvec_int (exact int32 W @ x).

Stream formats (fixed, shared with the engine and the software driving it):
  weights      row-major over neurons, five weights a byte in base 3: byte = sum over j of
               (w_j + 1) * 3^j for j = 0..4 (0..242; a byte of five zero weights is 121), the
               lowest power the first weight. One beat = 16 bytes = 80 consecutive weights of one
               neuron, so weight 5j + t of the beat is trit t of byte j and byte j is at bits
               [8j +: 8]; a neuron is B = ceil(K / 80) beats (32 for K = 2560, 87 for K = 6912)
               with its last 80B - K weights padded with zero trits
  activations  int8 bytes in order, 16 a beat, 5 beats an 80-activation word, 5B beats a vector
               (160 for B = 32, 435 for B = 87) with the padding bytes zero, and the batch's
               vectors back to back
  results      one 32-bit little-endian two's complement sum a neuron a vector, the batch's sums
               of a neuron back to back in vector order
The .hex files hold one beat a line (32 hex digits, most significant first, so $readmemh gives the
128-bit beat directly) and one 8-digit sum a line.
"""
import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "vectors")
LAYER0 = os.environ.get("BITNET_LAYER0_DIR", os.path.join(HERE, "..", "tools"))
sys.path.insert(0, os.path.join(HERE, "..", "tools"))
import reference

IN = 2560
LANES = 80
TRITS_PER_BYTE = 5
BEAT_BYTES = 16
ACT_BEATS_PER_WORD = LANES // BEAT_BYTES


def beats_per_neuron(k):
    return (k + LANES - 1) // LANES


def pack_weights(W):
    """int8 W [N, K] in {-1, 0, +1} -> N * ceil(K / 80) beat hex lines, five trits a byte."""
    assert W.dtype == np.int8
    n, k = W.shape
    beats = beats_per_neuron(k)
    codes = np.ones((n, beats * LANES), dtype=np.uint8)
    codes[:, :k] = (W.astype(np.int16) + 1).astype(np.uint8)
    c = codes.reshape(n, beats * BEAT_BYTES, TRITS_PER_BYTE).astype(np.uint16)
    b = (c[..., 0] + 3 * c[..., 1] + 9 * c[..., 2] + 27 * c[..., 3] + 81 * c[..., 4]).astype(np.uint8)
    return [bytes(beat[::-1]).hex() for beat in b.reshape(-1, BEAT_BYTES)]


def pack_activations(xs, k):
    """a list of int8 x [K] -> 5 * ceil(K / 80) beat hex lines a vector, one after another."""
    beats = beats_per_neuron(k)
    lines = []
    for x in xs:
        assert x.ndim == 1 and x.dtype == np.int8 and x.shape[0] == k
        padded = np.zeros(beats * LANES, dtype=np.int8)
        padded[:k] = x
        lines += [bytes(beat[::-1]).hex() for beat in padded.view(np.uint8).reshape(-1, BEAT_BYTES)]
    return lines


def pack_sums(y):
    return [f"{int(v) & 0xFFFFFFFF:08x}" for v in y.reshape(-1)]


def write_set(name, W, xs):
    """y is [N, batch]: the sums of a neuron back to back in vector order, as they come out."""
    y = np.stack([reference.matvec_int(W, x) for x in xs], axis=1)
    os.makedirs(OUT, exist_ok=True)
    n, k = W.shape
    beats = beats_per_neuron(k)
    for suffix, lines in (("act", pack_activations(xs, k)), ("w", pack_weights(W)), ("y", pack_sums(y))):
        with open(os.path.join(OUT, f"{name}_{suffix}.hex"), "w") as f:
            f.write("\n".join(lines) + "\n")
    print(f"{name}: {n} neurons x {k} inputs, batch {len(xs)}, {beats} beats a neuron "
          f"({beats * LANES - k} padding weights), {n * beats} weight beats, "
          f"{len(xs) * ACT_BEATS_PER_WORD * beats} activation beats, "
          f"y range [{y.min()}, {y.max()}], y[0] = {y[0].tolist()}")
    return y


def check_roundtrip(W):
    """Every trit of the base-3 stream unpacks back to the weight it came from."""
    n, k = W.shape
    beats = beats_per_neuron(k)
    raw = np.frombuffer(bytes.fromhex("".join(line for line in pack_weights(W))), dtype=np.uint8)
    raw = raw.reshape(-1, BEAT_BYTES)[:, ::-1].reshape(n, beats * BEAT_BYTES)
    got = np.empty((n, beats * LANES), dtype=np.int8)
    v = raw.astype(np.int32)
    for t in range(TRITS_PER_BYTE):
        got[:, t::TRITS_PER_BYTE] = (v % 3).astype(np.int8) - 1
        v //= 3
    if not np.array_equal(got[:, :k], W):
        raise SystemExit("the base-3 stream does not unpack to its weights")
    if got[:, k:].any():
        raise SystemExit("a padding weight is not zero")
    return int(got[:, k:].size)


def main():
    parser = argparse.ArgumentParser(description="test vectors for hardware/ternary_matvec.v")
    parser.add_argument("--k", type=int, default=None, help="write the k<K> set: 8 neurons of K inputs")
    args = parser.parse_args()

    if args.k is not None:
        if args.k <= 0:
            print("--k must be positive", file=sys.stderr)
            return 2
        rng = np.random.default_rng(20260907)
        W = rng.integers(-1, 2, size=(8, args.k), dtype=np.int64).astype(np.int8)
        x = rng.integers(-128, 128, size=args.k, dtype=np.int64).astype(np.int8)
        print(f"  {check_roundtrip(W)} padding weights checked zero")
        write_set(f"k{args.k}", W, [x])
        print(f"wrote {OUT}")
        return 0

    rng = np.random.default_rng(20260905)
    W = rng.integers(-1, 2, size=(32, IN), dtype=np.int64).astype(np.int8)
    x = rng.integers(-128, 128, size=IN, dtype=np.int64).astype(np.int8)
    check_roundtrip(W)
    write_set("rand32", W, [x])

    if os.path.exists(os.path.join(LAYER0, "W_gate_proj.npy")):
        Wg, _ = reference.load(LAYER0)
        rng = np.random.default_rng(20260906)
        xg = rng.integers(-128, 128, size=IN, dtype=np.int64).astype(np.int8)
        check_roundtrip(Wg[:64])
        write_set("gate64", Wg[:64], [xg])
    else:
        print(f"gate64 skipped: no {os.path.join(LAYER0, 'W_gate_proj.npy')}. "
              f"Get it with:  python3 ../tools/fetch.py ../tools\n"
              f"(the gate64 vectors already in vectors/ are left untouched)", file=sys.stderr)

    rng = np.random.default_rng(20260908)
    Wb = rng.integers(-1, 2, size=(8, IN), dtype=np.int64).astype(np.int8)
    xb = [rng.integers(-128, 128, size=IN, dtype=np.int64).astype(np.int8) for _ in range(4)]
    write_set("batch32", Wb, xb)

    rng = np.random.default_rng(20260909)
    Wk = rng.integers(-1, 2, size=(4, 6912), dtype=np.int64).astype(np.int8)
    xk = [rng.integers(-128, 128, size=6912, dtype=np.int64).astype(np.int8) for _ in range(4)]
    print(f"  bk6912: {check_roundtrip(Wk)} padding weights checked zero")
    write_set("bk6912", Wk, xk)
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
