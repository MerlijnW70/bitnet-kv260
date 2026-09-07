"""Where the triage answers actually land: the label the ground truth asks for against the label
that came back, for every way, over all forty lines.

The single accuracy number hides the shape of the failure. A model that answered at random would
miss differently from one that has collapsed onto one label, and the fix for each is different.

usage:
  python edge_confusion.py edge-scores.json
"""
import argparse
import json
import sys

CATEGORIES = ("mechanical", "electrical", "thermal", "sensor", "network", "software", "none")
SEVERITIES = ("critical", "warning", "info")


def table(rows, want, got, vocab, title):
    print(f"\n--- {title}: rows are the truth, columns what came back")
    print(f"    {'':12s}" + "".join(f"{c[:9]:>10s}" for c in vocab) + f"{'total':>10s}")
    for t in vocab:
        line = [sum(1 for r in rows if r[want] == t and r[got] == g) for g in vocab]
        n = sum(1 for r in rows if r[want] == t)
        hit = sum(1 for r in rows if r[want] == t and r[got] == t)
        print(f"    {t:12s}" + "".join(f"{v:>10d}" for v in line) + f"{n:>10d}"
              + (f"   {hit}/{n} right" if n else ""))
    print(f"    {'answered':12s}" + "".join(
        f"{sum(1 for r in rows if r[got] == g):>10d}" for g in vocab)
        + f"{len(rows):>10d}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scores")
    a = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    d = json.load(open(a.scores, encoding="utf-8"))
    for name, rows in d["rows"].items():
        print(f"\n================ {name} ({len(rows)} answers, all forty lines)")
        table(rows, "want_category", "got_category", CATEGORIES, "category")
        table(rows, "want_severity", "got_severity", SEVERITIES, "severity")
        crit = [r for r in rows if r["want_severity"] == "critical"]
        print(f"\n    of the {len(crit)} lines whose truth is critical, "
              f"{sum(1 for r in crit if r['got_severity'] == 'critical')} came back critical, "
              f"{sum(1 for r in crit if r['got_severity'] == 'warning')} warning, "
              f"{sum(1 for r in crit if r['got_severity'] == 'info')} info")
    return 0


if __name__ == "__main__":
    sys.exit(main())
