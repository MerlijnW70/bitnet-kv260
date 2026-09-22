"""hostfmt.py <stem> <groups> <slots> <dim> <layers>: the byte stream the runtime writes, for the RTL to read back"""

import random
import sys

import fxmodel
import genshape


def shape(kv, nq, fd):
    rows = fd // 16
    scb = (kv + 7) // 8
    cells = rows * kv
    qfb = (nq * 16 + 127) // 128
    topb = (nq * 24 + 127) // 128
    return {"kv": kv, "nq": nq, "fd": fd, "rows": rows, "scb": scb, "cells": cells,
            "qfb": qfb, "topb": topb,
            "block": (2 * scb + 2 * cells) * 16,
            "header": (1 + qfb + topb + nq * rows) * 16,
            "off_top": (1 + qfb) * 16,
            "off_qq": (1 + qfb + topb) * 16}


def le(buf, off, value, n):
    for i in range(n):
        buf[off + i] = (value >> (8 * i)) & 0xFF


def position_block(s, k16, v16, k, v):
    """One position, laid out exactly as bitnet_kria.c's CACHE_FAB write path does."""
    b = bytearray(s["block"])
    kbase = s["scb"] * 32
    for g in range(s["kv"]):
        sk = (g // 8) * 16 + (g % 8) * 2
        le(b, sk, k16[g], 2)
        le(b, s["scb"] * 16 + sk, v16[g], 2)
        for d in range(s["fd"]):
            b[kbase + g * s["fd"] + d] = k[g][d] & 0xFF
            b[kbase + (s["kv"] + g) * s["fd"] + d] = v[g][d] & 0xFF
    return b


def header(s, pas, T, qf, top, qq):
    """The header attn_fabric builds, with the offsets the shape gives."""
    h = bytearray(s["header"])
    h[0], h[1] = 0x7E, 0xA7
    h[2] = pas
    le(h, 4, T, 2)
    for i in range(s["nq"]):
        le(h, 16 + 2 * i, qf[i], 2)
        le(h, s["off_top"] + 3 * i, top[i] & 0xFFFFFF, 3)
    for i in range(s["nq"] * s["fd"]):
        h[s["off_qq"] + i] = qq[i] & 0xFF
    return h


def beats(buf):
    return ["".join("%02x" % buf[o + 15 - i] for i in range(16)) for o in range(0, len(buf), 16)]


def main():
    stem = sys.argv[1]
    groups, slots, dim, n = (int(v) for v in sys.argv[2:6])
    nq = groups * slots
    s = shape(groups, nq, dim)
    rng = random.Random(20260923)
    layers = [genshape.random_layer(rng, groups, slots, dim) for _ in range(n)]

    with open(stem + ".beats", "w") as f, open(stem + ".want", "w") as w:
        for layer in layers:
            T, gs = layer["T"], layer["groups"]
            qf = [g["qf"][j] for g in gs for j in range(slots)]
            qq = [x for g in gs for x in g["q"]]
            top = [g["top"][j] for g in gs for j in range(slots)]
            tot = [g["sum"][j] for g in gs for j in range(slots)]
            acc = [x for g in gs for x in g["acc"]]
            f.write("LAYER %d\n" % T)
            for pas in (0, 1):
                f.write("PASS %d\n" % pas)
                for line in beats(header(s, pas, T, qf, top if pas else [0] * nq, qq)):
                    f.write(line + "\n")
                for t in range(T):
                    blk = position_block(s,
                                         [g["k16"][t] for g in gs],
                                         [g["v16"][t] for g in gs],
                                         [g["k"][t * dim:(t + 1) * dim] for g in gs],
                                         [g["v"][t * dim:(t + 1) * dim] for g in gs])
                    for line in beats(blk):
                        f.write(line + "\n")
            w.write("TOP %s\n" % " ".join(str(x) for x in top))
            w.write("SUM %s\n" % " ".join(str(x) for x in tot))
            w.write("ACC %s\n" % " ".join(str(x) for x in acc))
        f.write("END\n")
        w.write("END\n")
    print("%s.beats and %s.want: %d layers, %d positions, block %d B, header %d B"
          % (stem, stem, len(layers), sum(l["T"] for l in layers), s["block"], s["header"]))


if __name__ == "__main__":
    main()
