"""Pack layer 0's gate_proj of BitNet b1.58 2B4T into the stream format of the KV260 ternary
matvec accelerator, and make one fixed random activation vector with its exact reference product.

pack_model.py imports this module for pack_weights / unpack_weights / matvec_int, which is why it
is here. Running it as a script writes the TWO-BIT single-matrix test vectors of the earlier
engine and needs W_gate_proj.npy beside it (get it with `python3 fetch.py .`); the driver those
vectors were for is not part of this repository, and the shipped bitstream streams base 3 only.

usage: python pack.py [out_dir] [--seed N]        (default out_dir: this directory, seed 1)

Weight stream (fixed, shared with the hardware): row-major over neurons; each weight two bits,
value w+1 (0, 1, 2 for -1, 0, +1); four weights a byte, the lowest two bits the first weight;
2560 weights = 640 bytes a neuron; the DMA stream is 128 bits wide, one beat = 16 bytes = 64
consecutive weights of one neuron, so a neuron is 40 beats and the layer 6912*40 = 276,480 beats.
Activation stream: int8 bytes in order, 16 activations a beat, 160 beats for 2560.
Result stream: one 32-bit little-endian two's complement sum per neuron, 6912 words.

Writes into out_dir:
  weights.bin  6912*640 = 4,423,680 bytes, the weight stream
  x.bin        2560 bytes, int8 activations from numpy's default_rng(seed)
  expect.bin   6912*4 = 27,648 bytes, int32 little-endian reference.matvec_int(W, x)
then reads the three back, unpacks the weights and checks them against W, recomputes the product,
and prints the sizes, the first beat and the first eight sums.
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from reference import matvec_int

IN_FEATURES = 2560
OUT_FEATURES = 6912
BYTES_PER_NEURON = IN_FEATURES // 4
BEAT_BYTES = 16
DEFAULT_SEED = 1


def pack_weights(W):
    """int8 ternary W [out, in] -> uint8 [out, in/4]: weight k of a row sits in bits 2(k%4)..2(k%4)+1 of byte k//4 as w+1."""
    if W.dtype != np.int8:
        raise TypeError("pack_weights wants int8 W")
    out, cols = W.shape
    if cols % 4:
        raise ValueError(f"{cols} columns is not a multiple of four")
    v = (W.astype(np.int16) + 1)
    if v.min() < 0 or v.max() > 2:
        raise ValueError("W holds a value outside {-1, 0, +1}")
    v = v.astype(np.uint8).reshape(out, cols // 4, 4)
    packed = v[:, :, 0] | (v[:, :, 1] << 2) | (v[:, :, 2] << 4) | (v[:, :, 3] << 6)
    return np.ascontiguousarray(packed, dtype=np.uint8)


def unpack_weights(packed, cols):
    """The inverse of pack_weights: uint8 [out, cols/4] -> int8 [out, cols]."""
    out = packed.shape[0]
    v = np.empty((out, cols // 4, 4), dtype=np.uint8)
    for k in range(4):
        v[:, :, k] = (packed >> (2 * k)) & 3
    if v.max() > 2:
        raise ValueError("a two-bit field of 3 appeared")
    return (v.reshape(out, cols).astype(np.int16) - 1).astype(np.int8)


def activations(seed, n=IN_FEATURES):
    return np.random.default_rng(seed).integers(-128, 128, size=n, dtype=np.int64).astype(np.int8)


def write(out_dir, W, x):
    packed = pack_weights(W)
    y = matvec_int(W, x)
    paths = {name: os.path.join(out_dir, name) for name in ("weights.bin", "x.bin", "expect.bin")}
    with open(paths["weights.bin"], "wb") as f:
        f.write(packed.tobytes(order="C"))
    with open(paths["x.bin"], "wb") as f:
        f.write(x.tobytes())
    with open(paths["expect.bin"], "wb") as f:
        f.write(y.astype("<i4").tobytes())
    return paths, packed, y


def check(paths, W, x, y):
    packed = np.fromfile(paths["weights.bin"], dtype=np.uint8)
    assert packed.size == W.shape[0] * W.shape[1] // 4, packed.size
    packed = packed.reshape(W.shape[0], W.shape[1] // 4)
    assert np.array_equal(unpack_weights(packed, W.shape[1]), W), "weights.bin does not unpack to W"
    x_back = np.fromfile(paths["x.bin"], dtype=np.int8)
    assert np.array_equal(x_back, x), "x.bin differs"
    y_back = np.fromfile(paths["expect.bin"], dtype="<i4")
    assert np.array_equal(y_back, y), "expect.bin differs"
    assert np.array_equal(matvec_int(W, x_back).astype(np.int64), W.astype(np.int64) @ x_back.astype(np.int64))
    by_hand = bytes(sum((int(W[0, 4 * b + k]) + 1) << (2 * k) for k in range(4)) for b in range(BEAT_BYTES))
    assert packed[0, :BEAT_BYTES].tobytes() == by_hand, "first beat differs from the by-hand packing"
    return packed


def main(argv):
    out_dir, seed = HERE, DEFAULT_SEED
    args = list(argv)
    if "--seed" in args:
        i = args.index("--seed")
        seed = int(args[i + 1])
        del args[i:i + 2]
    if args:
        out_dir = args[0]
    os.makedirs(out_dir, exist_ok=True)
    W = np.load(os.path.join(HERE, "W_gate_proj.npy"))
    assert W.shape == (OUT_FEATURES, IN_FEATURES) and W.dtype == np.int8, (W.shape, W.dtype)
    x = activations(seed)
    paths, packed, y = write(out_dir, W, x)
    check(paths, W, x, y)
    for name, path in paths.items():
        print(f"{path}  {os.path.getsize(path)} bytes")
    print(f"seed {seed}; {OUT_FEATURES} neurons x {IN_FEATURES} weights; {BYTES_PER_NEURON} bytes = {BYTES_PER_NEURON // BEAT_BYTES} beats a neuron; "
          f"{OUT_FEATURES * BYTES_PER_NEURON // BEAT_BYTES} weight beats, {IN_FEATURES // BEAT_BYTES} activation beats, {OUT_FEATURES} result words")
    print(f"first weight beat (neuron 0, weights 0..63): {packed[0, :BEAT_BYTES].tobytes().hex()}")
    print(f"first activation beat: {x[:BEAT_BYTES].tobytes().hex()}")
    print(f"y range [{y.min()}, {y.max()}], y[:8] = {y[:8].tolist()}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
