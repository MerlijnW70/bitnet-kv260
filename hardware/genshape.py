"""genshape.py <out> <groups> <slots> <dim> <random-layers>: test calls for any block shape, with fxmodel's answers"""

import random
import sys

import fxmodel


def layer_from(T, rows_k, rows_v, qs, k16, v16, qf, groups, slots, dim):
    gs = []
    for g in range(groups):
        call = {"T": T, "heads": slots, "dim": dim, "qf": qf[g], "q": qs[g],
                "k16": k16[g], "k": rows_k[g], "v16": v16[g], "v": rows_v[g]}
        call["top"], call["sum"], call["acc"] = fxmodel.fx(call)
        gs.append(call)
    return {"T": T, "groups": gs}


def random_layer(rng, groups, slots, dim):
    T = rng.choice([1, 2, 9, rng.randint(1, 120)])
    edge = rng.random() < 0.25
    pick = (lambda: rng.choice([-128, 127, 0, -1, 1])) if edge else (lambda: rng.randint(-128, 127))
    s16 = lambda: rng.choice([0, 65535, rng.randint(0, 65535)])
    return layer_from(T,
                      [[pick() for _ in range(T * dim)] for _ in range(groups)],
                      [[pick() for _ in range(T * dim)] for _ in range(groups)],
                      [[pick() for _ in range(slots * dim)] for _ in range(groups)],
                      [[s16() for _ in range(T)] for _ in range(groups)],
                      [[s16() for _ in range(T)] for _ in range(groups)],
                      [[rng.choice([0, 1, 65535, rng.randint(1, 65535)]) for _ in range(slots)]
                       for _ in range(groups)],
                      groups, slots, dim)


def write(out, layers, dim):
    with open(out, "w") as f:
        for layer in layers:
            T, gs = layer["T"], layer["groups"]
            f.write("LAYER %d\n" % T)
            f.write("QF %s\n" % " ".join(str(x) for g in gs for x in g["qf"]))
            f.write("Q %s\n" % " ".join(str(x) for g in gs for x in g["q"]))
            for t in range(T):
                f.write("P %s %s %s %s\n" % (
                    " ".join(str(g["k16"][t]) for g in gs),
                    " ".join(str(g["v16"][t]) for g in gs),
                    " ".join(str(x) for g in gs for x in g["k"][t * dim:(t + 1) * dim]),
                    " ".join(str(x) for g in gs for x in g["v"][t * dim:(t + 1) * dim])))
            f.write("TOP %s\n" % " ".join(str(x) for g in gs for x in g["top"]))
            f.write("SUM %s\n" % " ".join(str(x) for g in gs for x in g["sum"]))
            f.write("ACC %s\n" % " ".join(str(x) for g in gs for x in g["acc"]))
        f.write("END\n")


def main():
    out = sys.argv[1]
    groups, slots, dim, n = (int(v) for v in sys.argv[2:6])
    rng = random.Random(20260922)
    layers = [random_layer(rng, groups, slots, dim) for _ in range(n)]
    write(out, layers, dim)
    print("%s: %d groups of %d heads of %d, %d layers, %d positions"
          % (out, groups, slots, dim, len(layers), sum(l["T"] for l in layers)))


if __name__ == "__main__":
    main()
