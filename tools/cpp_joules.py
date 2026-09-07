"""Joules an answer for the bitnet.cpp baseline, from the INA260 log taken around that run.

quality_bitnetcpp.py records the absolute t0 and t1 of every llama-cli invocation; ina260-sample.py
writes "epoch_s uW mA mV" every 50 ms through the whole thing. This trapezoid-integrates the power
over each answer's own window, exactly as edge_monitor.py does on the board, so the two joules
figures are computed the same way and can be put side by side. The idle stretches the launcher
leaves before and after the run give the idle draw to subtract.

usage:
  python cpp_joules.py edge-cpp-answers.json edge-cpp-power.log edge-cpp.window
"""
import argparse
import json
import sys


def load_log(path):
    out = []
    for line in open(path, encoding="utf-8"):
        f = line.split()
        if len(f) >= 4:
            try:
                out.append((float(f[0]), int(f[1]) / 1e6))
            except ValueError:
                pass
    out.sort()
    return out


def integrate(s, t0, t1):
    """Joules and mean/min/max watts over [t0, t1], trapezoid, interpolating at the edges."""
    if t1 <= t0 or len(s) < 2:
        return None
    j = 0.0
    for (ta, wa), (tb, wb) in zip(s, s[1:]):
        lo, hi = max(ta, t0), min(tb, t1)
        if hi <= lo or tb <= ta:
            continue
        wlo = wa + (wb - wa) * (lo - ta) / (tb - ta)
        whi = wa + (wb - wa) * (hi - ta) / (tb - ta)
        j += 0.5 * (wlo + whi) * (hi - lo)
    ws = [w for (t, w) in s if t0 <= t <= t1]
    if not ws:
        return None
    return {"joules": round(j, 3), "mean_w": round(j / (t1 - t0), 4),
            "min_w": round(min(ws), 4), "max_w": round(max(ws), 4), "samples": len(ws)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("answers")
    ap.add_argument("log")
    ap.add_argument("window", nargs="?", default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    doc = json.load(open(a.answers, encoding="utf-8"))
    s = load_log(a.log)
    print(f"{len(s)} INA260 samples spanning {s[-1][0] - s[0][0]:.1f} s")

    if a.window:
        w = [float(x) for x in open(a.window, encoding="utf-8").read().split()]
        if len(w) >= 2:
            before = integrate(s, s[0][0], w[0])
            during = integrate(s, w[0], w[1])
            after = integrate(s, w[1], s[-1][0])
            print(f"idle before the run: {before}")
            print(f"the whole run      : {during}")
            print(f"idle after the run : {after}")

    js, ms = [], []
    for r in doc["runs"]:
        if "t0" not in r:
            continue
        p = integrate(s, r["t0"], r["t1"])
        r["power"] = p
        if p:
            js.append(p["joules"])
            ms.append(r["wall_s"] * 1e3)
    if js:
        print(f"\n{len(js)} answers with a power window")
        print(f"  joules an answer  mean {sum(js)/len(js):.3f}  min {min(js):.3f}  "
              f"max {max(js):.3f}  total {sum(js):.1f} J")
        print(f"  ms an answer      mean {sum(ms)/len(ms):.1f}  min {min(ms):.1f}  "
              f"max {max(ms):.1f}")
    if a.out:
        json.dump(doc, open(a.out, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
        print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
