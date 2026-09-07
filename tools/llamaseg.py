"""Cut a bitnet.cpp power trace into load, prompt eval and generation using llama's own milliseconds.

usage (on the board)
  python3 llamaseg.py /home/ubuntu/bitnet/power-prove-bitnetcpp2.log \
                      /home/ubuntu/bitnet/power-prove-bitnetcpp2-run.err

The whole-window mean of a llama run is not its generation power: the window holds a quiet start and
model load, then the prompt eval, then the generation.  llama prints how many milliseconds each of the
last two took, so the boundaries are found by counting those backwards from the end of the high-power
region (the last sample above idle + 0.4 W).  The same cut kria-model-results.txt used.
"""
import re
import sys


def main():
    log, err = sys.argv[1], sys.argv[2]
    text = open(err, encoding="utf-8", errors="replace").read()
    ms = {}
    for key, pat in (("prompt", r"prompt eval time =\s*([\d.]+) ms /\s*(\d+) tokens"),
                     ("eval", r"eval time\s*=\s*([\d.]+) ms /\s*(\d+) runs"),
                     ("load", r"load time =\s*([\d.]+) ms")):
        m = re.search(pat, text)
        if m:
            ms[key] = (float(m.group(1)), int(m.group(2)) if m.lastindex and m.lastindex > 1 else 0)
    rows = []
    for line in open(log, encoding="utf-8"):
        f = line.split()
        if len(f) >= 4:
            rows.append((float(f[0]), int(f[1]) / 1e6, int(f[2]), int(f[3])))
    idle = sorted(p for _, p, _, _ in rows)[len(rows) // 20]
    hot = [t for t, p, _, _ in rows if p > idle + 0.4]
    end = hot[-1]
    gen0 = end - ms["eval"][0] / 1e3
    pr0 = gen0 - ms["prompt"][0] / 1e3

    def band(a, b, name, tokens):
        sel = [(p, c, v) for t, p, c, v in rows if a <= t < b]
        if not sel:
            print(f"{name}: no samples")
            return
        mean = sum(p for p, _, _ in sel) / len(sel)
        secs = b - a
        line = (f"{name:<12s} {len(sel):4d} samples  power mean {mean:.3f} W  min "
                f"{min(p for p, _, _ in sel):.3f} W  max {max(p for p, _, _ in sel):.3f} W  over "
                f"{secs:7.3f} s")
        if tokens:
            line += (f"  {tokens:4d} tokens = {tokens / secs:5.2f} tok/s, "
                     f"{mean * secs / tokens:.3f} J a token, "
                     f"{(mean - idle) * secs / tokens:.3f} J over idle")
        print(line)

    print(f"idle (5th percentile of the whole log) {idle:.3f} W; {len(rows)} samples")
    band(rows[0][0], pr0, "start+load", 0)
    band(pr0, gen0, "prompt eval", ms["prompt"][1])
    band(gen0, end, "generation", ms["eval"][1])
    band(end, rows[-1][0], "after", 0)
    return 0


if __name__ == "__main__":
    sys.exit(main())
