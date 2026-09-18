"""fxmodel.py <dump...>: recompute every dumped fx_group call in integers and compare with what the board wrote"""

import math
import sys

FRAC = 12
CUT = 20
ONE = (1 << 20) - 1
EXPF = [int(round(ONE * math.exp(-f / (1 << FRAC)))) for f in range(1 << FRAC)]
EXPI = [int(round(ONE * math.exp(-i))) for i in range(CUT + 1)]


def read_calls(path):
    calls, cur = [], None
    for line in open(path):
        word = line.split()
        if not word:
            continue
        if word[0] == "call":
            cur = {"T": int(word[3]), "heads": int(word[5]), "dim": int(word[7])}
            calls.append(cur)
        else:
            cur[word[0]] = [int(x) for x in word[1:]]
    return calls


def dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def fx(call):
    T, nh, hd = call["T"], call["heads"], call["dim"]
    q = [call["q"][h * hd:(h + 1) * hd] for h in range(nh)]
    k = [call["k"][t * hd:(t + 1) * hd] for t in range(T)]
    v = [call["v"][t * hd:(t + 1) * hd] for t in range(T)]
    score = [[0] * T for _ in range(nh)]
    top = [None] * nh
    for t in range(T):
        for h in range(nh):
            s = (dot(k[t], q[h]) * call["k16"][t] * call["qf"][h]) >> (36 - FRAC)
            s = max(-(1 << 23), min((1 << 23) - 1, s))
            score[h][t] = s
            top[h] = s if top[h] is None else max(top[h], s)
    weight = [[0] * T for _ in range(nh)]
    total = [0] * nh
    for h in range(nh):
        for t in range(T):
            d = top[h] - score[h][t]
            whole = d >> FRAC
            w = 0 if whole > CUT else (EXPF[d & ((1 << FRAC) - 1)] * EXPI[whole]) >> 20
            weight[h][t] = w
            total[h] += w
    acc = [[0] * hd for _ in range(nh)]
    for t in range(T):
        for h in range(nh):
            w = weight[h][t]
            if not w:
                continue
            u = (w * call["v16"][t]) >> 16
            for d in range(hd):
                acc[h][d] += u * v[t][d]
    return top, total, [x for row in acc for x in row]


def main():
    bad = 0
    count = 0
    for path in sys.argv[1:]:
        for call in read_calls(path):
            top, total, acc = fx(call)
            same = (top == call["top"], total == call["sum"], acc == call["acc"])
            count += 1
            if not all(same):
                bad += 1
            print("%s call T=%d heads=%d: top %s, sum %s, acc %s"
                  % (path, call["T"], call["heads"], *("same" if s else "DIFFERENT" for s in same)))
    print("%d calls, %d different" % (count, bad))
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
