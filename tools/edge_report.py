"""Pull the pieces of an edge_monitor run out of its JSON: transcript, power, net, self-reports.

usage:
  python edge_report.py edge-run.json --transcript 12 > pieces.txt
"""
import argparse
import json
import sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("json")
    ap.add_argument("--transcript", type=int, default=12)
    ap.add_argument("--only", default=None)
    a = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    d = json.load(open(a.json, encoding="utf-8"))
    runs = d["runs"]
    tri = [r for r in runs if r["job"] == "triage" and not r.get("mode")]
    self_ = [r for r in runs if r["job"] == "selftest"]

    print(f"=== setup: {d['cmd']}")
    print(f"    setup_s {d['setup_s']}, max_new {d['max_new']}, context {d['context']}, "
          f"threads {d['threads']}, cache {d['cache_dtype']}, greedy {d['greedy']}")
    print(f"    run_s {d['run_s']}, {len(runs)} answers ({len(tri)} triage, {len(self_)} self)")
    print(f"=== idle before: {d.get('idle_before')}")
    print(f"=== app summary (NOTE: folds the --saving pass in): {json.dumps(d['summary'])}")
    print(f"=== saving: {json.dumps(d.get('saving'))}")

    print("\n=== power over the triage answers")
    pw = [r["power"] for r in tri if r.get("power")]
    if pw:
        j = [p["joules"] for p in pw]
        w = [p["mean_w"] for p in pw]
        mx = [p["max_w"] for p in pw]
        n = [p["samples"] for p in pw]
        print(f"  joules  mean {sum(j)/len(j):.3f}  min {min(j):.3f}  max {max(j):.3f}  "
              f"total {sum(j):.1f} J over {len(j)} answers")
        print(f"  watts   mean {sum(w)/len(w):.3f}  min {min(w):.3f}  max {max(w):.3f}")
        print(f"  peak W  max over answers {max(mx):.3f}")
        print(f"  INA260 samples an answer: mean {sum(n)/len(n):.0f}, min {min(n)}, max {max(n)}")

    print("\n=== /proc/net/dev during the answers themselves")
    tot = {}
    nz = 0
    for r in tri:
        nd = r.get("net_delta") or {}
        if any(v["rx_bytes"] or v["tx_bytes"] for v in nd.values()):
            nz += 1
        for k, v in nd.items():
            t = tot.setdefault(k, {"rx_bytes": 0, "tx_bytes": 0, "rx_packets": 0, "tx_packets": 0})
            for m in t:
                t[m] += v[m]
    for k, v in sorted(tot.items()):
        print(f"  {k:6s} summed over the {len(tri)} answer windows: "
              f"rx {v['rx_bytes']} B / {v['rx_packets']} pkt, "
              f"tx {v['tx_bytes']} B / {v['tx_packets']} pkt")
    print(f"  answers whose own window saw ANY byte on ANY interface: {nz} of {len(tri)}")

    print("\n=== whole-run /proc/net/dev")
    for k in sorted(d["net_before"]):
        b, af = d["net_before"][k], d["net_after"][k]
        print(f"  {k:6s} rx {af['rx_bytes']-b['rx_bytes']:>9d} B / "
              f"{af['rx_packets']-b['rx_packets']:>6d} pkt   "
              f"tx {af['tx_bytes']-b['tx_bytes']:>9d} B / {af['tx_packets']-b['tx_packets']:>6d} pkt")

    print(f"\n=== self-reports ({len(self_)})")
    for r in self_:
        s = r["readings"]
        print(f"--- {r['id']}: {r['answer_ms']:.0f} ms, {r['prompt_ids']} prompt ids, "
              f"{r['gen_ids']} gen ids, {r['sentences']} sentence ends, {r['words']} words, "
              f"verdict {r['verdict']!r}"
              + (f", {r['power']['joules']:.3f} J at {r['power']['mean_w']:.2f} W"
                 if r.get("power") else ""))
        print(f"    sensors: {s['power_w']} W, {s['current_ma']} mA, {s['voltage_v']} V, "
              f"PS {s['temp_ps_lpd_c']}/{s['temp_ps_fpd_c']} C, PL {s['temp_pl_c']} C, "
              f"cpu {s['cpu_mhz']} MHz, load {s['loadavg']}, "
              f"mem {s['mem_available_mb']}/{s['mem_total_mb']} MB, fpga {s['fpga_state']}, "
              f"up {s['uptime_s']/86400:.2f} d")
        print("    PROMPT:")
        for ln in r["prompt"].split("\n"):
            print(f"      {ln}")
        print("    ANSWER:")
        for ln in r["answer"].strip().split("\n"):
            print(f"      {ln}")

    print(f"\n=== triage transcript (first {a.transcript})")
    want = set(a.only.split(",")) if a.only else None
    shown = 0
    for r in tri:
        if want is not None and r["id"] not in want:
            continue
        if want is None and shown >= a.transcript:
            break
        shown += 1
        print(f"--- {r['id']} [{'AMBIG' if r['ambiguous'] else 'clear'}] "
              f"want {r['want_category']}/{r['want_severity']}")
        print(f"  > {r['prompt']}")
        print(f"  {r['answer'].strip()}")
        pwr = r.get("power")
        print(f"    strict json {r['json_strict_ok']}, got {r['got_category']}/{r['got_severity']} "
              f"(category {'OK' if r['category_ok'] else 'no'}, "
              f"severity {'OK' if r['severity_ok'] else 'no'}), "
              f"{r['answer_ms']:.0f} ms, {r.get('prompt_tokens', r['prompt_ids'])} prompt, "
              f"{r.get('gen_tokens', r['gen_ids'])} gen"
              + (f", {pwr['joules']:.2f} J at {pwr['mean_w']:.2f} W" if pwr else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
