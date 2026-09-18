"""genlayers.py <out> <random-layers> <dump...>: whole-layer test calls (5 groups of 4 heads) with fxmodel's answers"""

import random
import sys

import fxmodel

GROUPS, SLOTS, DIM = 5, 4, 128


def layer_from(rng, T, rows_k, rows_v, qs, k16, v16, qf):
    groups = []
    for g in range(GROUPS):
        call = {"T": T, "heads": SLOTS, "dim": DIM, "qf": qf[g], "q": qs[g],
                "k16": k16[g], "k": rows_k[g], "v16": v16[g], "v": rows_v[g]}
        call["top"], call["sum"], call["acc"] = fxmodel.fx(call)
        groups.append(call)
    return {"T": T, "groups": groups}


def random_layer(rng):
    T = rng.choice([1, 2, 9, rng.randint(1, 120)])
    edge = rng.random() < 0.25
    pick = (lambda: rng.choice([-128, 127, 0, -1, 1])) if edge else (lambda: rng.randint(-128, 127))
    return layer_from(rng, T,
                      [[pick() for _ in range(T * DIM)] for _ in range(GROUPS)],
                      [[pick() for _ in range(T * DIM)] for _ in range(GROUPS)],
                      [[pick() for _ in range(SLOTS * DIM)] for _ in range(GROUPS)],
                      [[rng.choice([0, 65535, rng.randint(0, 65535)]) for _ in range(T)] for _ in range(GROUPS)],
                      [[rng.choice([0, 65535, rng.randint(0, 65535)]) for _ in range(T)] for _ in range(GROUPS)],
                      [[rng.choice([0, 1, 65535, rng.randint(1, 65535)]) for _ in range(SLOTS)] for _ in range(GROUPS)])


def real_layer(rng, calls, T):
    donors = [c for c in calls if c["T"] >= T]
    rows_k, rows_v, k16, v16, qs, qf = [], [], [], [], [], []
    for g in range(GROUPS):
        c = donors[g % len(donors)]
        shift = rng.randint(0, c["T"] - T)
        rows_k.append(c["k"][shift * DIM:(shift + T) * DIM])
        rows_v.append(c["v"][shift * DIM:(shift + T) * DIM])
        k16.append(c["k16"][shift:shift + T])
        v16.append(c["v16"][shift:shift + T])
        q, f = [], []
        for s in range(SLOTS):
            h = (g + s) % c["heads"]
            q.extend(c["q"][h * DIM:(h + 1) * DIM])
            f.append(c["qf"][h])
        qs.append(q)
        qf.append(f)
    return layer_from(rng, T, rows_k, rows_v, qs, k16, v16, qf)


def write(out, layers):
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
                    " ".join(str(x) for g in gs for x in g["k"][t * DIM:(t + 1) * DIM]),
                    " ".join(str(x) for g in gs for x in g["v"][t * DIM:(t + 1) * DIM])))
            f.write("TOP %s\n" % " ".join(str(x) for g in gs for x in g["top"]))
            f.write("SUM %s\n" % " ".join(str(x) for g in gs for x in g["sum"]))
            f.write("ACC %s\n" % " ".join(str(x) for g in gs for x in g["acc"]))
        f.write("END\n")


def main():
    out, n = sys.argv[1], int(sys.argv[2])
    rng = random.Random(20260918)
    calls = []
    for path in sys.argv[3:]:
        calls.extend(fxmodel.read_calls(path))
    layers = []
    if calls:
        for T in (1, 40, 300, 1000):
            if any(c["T"] >= T for c in calls):
                layers.append(real_layer(rng, calls, T))
    layers.extend(random_layer(rng) for _ in range(n))
    write(out, layers)
    print("%s: %d layers, %d positions" % (out, len(layers), sum(l["T"] for l in layers)))


if __name__ == "__main__":
    main()
