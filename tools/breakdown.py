"""Weighted mean of the runtime's per-forward line over a provebench.py or headcheck.py json.

usage (on the board)
  python3 breakdown.py prove-after.json prove-baseline.json

Each run's line is a mean over that run's own forwards, so the runs are weighted by their forward
count (and the head by its call count) exactly as kria-model-results.txt weighted its 15 bench runs.
"""
import json
import re
import sys

N = r"(\d+\.\d+)"
FIELDS = [("engines qkv", "engines qkv " + N), ("engines o", ", o " + N),
          ("engines gate+up", r"gate\+up " + N), ("engines down", "down " + N + r" \("),
          ("engines together", r"\(" + N + r" together\)"),
          ("activation loads", "activation loads " + N),
          ("glue pass A", "glue A " + N), ("glue pass B", " B " + N + ";"),
          ("shift scan", "shift scan " + N), ("sums out", "sums out " + N),
          ("attention", "attention " + N),
          ("norms, quant, RoPE, residuals", "residuals " + N),
          ("waiting on the side thread", "waiting on the side thread " + N),
          ("forward total", "forward total " + N),
          ("side thread busy", "side thread busy " + N)]
HEAD = [("head a call", "head " + N + " ms a call"),
        ("  stage 1 ternary stream", "stage 1 ternary stream " + N),
        ("  its activation load", "its activation load " + N),
        ("  sync, scale and top-K", r"sync and top-\d+ " + N),
        ("  stage 2 exact rescore", "stage 2 exact rescore " + N)]


def one(path):
    d = json.load(open(path, encoding="utf-8"))
    acc = {k: 0.0 for k, _ in FIELDS + HEAD}
    fwd = calls = 0
    ptok = pms = gtok = gms = 0
    for r in d["runs"]:
        per = r["per_forward"]
        m = re.search(r"per forward over (\d+)", per)
        if not m:
            continue
        n = int(m.group(1))
        fwd += n
        for k, pat in FIELDS:
            mm = re.search(pat, per)
            if mm:
                acc[k] += float(mm.group(1)) * n
        mc = re.search(r"head [\d.]+ ms a call over (\d+) calls", per)
        if mc:
            c = int(mc.group(1))
            calls += c
            for k, pat in HEAD:
                mm = re.search(pat, per)
                if mm:
                    acc[k] += float(mm.group(1)) * c
        ms = re.match(r"prompt (\d+) tokens ([\d.]+) ms .*generated (\d+) ([\d.]+) ms", r["stats"])
        if ms:
            ptok += int(ms.group(1))
            pms += float(ms.group(2))
            gtok += int(ms.group(3))
            gms += float(ms.group(4))
    return acc, fwd, calls, (ptok, pms, gtok, gms)


def main():
    cols = [(p, *one(p)) for p in sys.argv[1:]]
    print(f"{'':32s}" + "".join(f"{p.split('/')[-1]:>22s}" for p, *_ in cols))
    print(f"{'forwards / head calls':32s}" +
          "".join(f"{str(f) + ' / ' + str(c):>22s}" for _, _, f, c, _ in cols))
    for k, _ in FIELDS + HEAD:
        row = f"{k:32s}"
        for _, acc, f, c, _ in cols:
            n = c if (k.startswith("head") or k.startswith("  ")) else f
            row += f"{(acc[k] / n if n else 0):22.3f}"
        print(row)
    for name, i, j in (("prompt tok/s", 0, 1), ("generation tok/s", 2, 3)):
        row = f"{name:32s}"
        for _, _, _, _, t in cols:
            row += f"{t[i] / (t[j] / 1e3):22.2f}"
        print(row)
    for name, i, j in (("a prompt token, ms", 0, 1), ("a generated token, ms", 2, 3)):
        row = f"{name:32s}"
        for _, _, _, _, t in cols:
            row += f"{t[j] / t[i]:22.3f}"
        print(row)
    return 0


if __name__ == "__main__":
    sys.exit(main())
