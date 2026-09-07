"""Run the recorded prompts through this board and compare every generated token id.

usage (on the board, as root, or through ../selftest.sh which handles that)
  python3 check_ids.py [--dir /home/ubuntu/bitnet-kria] [--exe PATH] [--head fabric|arm]
                       [--expected expected_ids.json] [--out ids.json]

selftest/expected_ids.json holds, for six short prompts, the exact prompt token ids and generated
token ids that the reference KV260 produced with the shipped base-3 bitstream and runtime.  The
same ids came out of four independent recorded runs (b3v2, pv2, ids-b3 and spec-ids-after), so they
are not a single lucky sample.  Decoding is greedy, so a correct board must reproduce them exactly:
one different id anywhere means the fabric, the weights or the runtime is not what it should be.

It also prints this board's prompt and generation rates beside the reference board's, so a slow
board is visible in the same output as a wrong one.  A rate well under the reference usually means
the fabric is at 100 MHz (the overlay was not loaded), the ARM head is in use, or a core is busy --
run ../doctor.sh, which tests all three.

exit 0 every id matched, 1 an id differed, 2 the run could not be made.
"""
import argparse
import json
import os
import re
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DIR = os.environ.get("BITNET_DIR", "/home/ubuntu/bitnet-kria")
sys.path.insert(0, DEFAULT_DIR)
sys.path.insert(0, os.path.join(HERE, "..", "runtime"))
import bitnet_chat as bc


def rates(stats):
    """'prompt 22 tokens 789.5 ms (27.87 tok/s), generated 8 401.5 ms (19.92 tok/s)' -> numbers."""
    m = re.match(r"prompt (\d+) tokens ([\d.]+) ms .*generated (\d+) ([\d.]+) ms", stats or "")
    if not m:
        return None
    return (int(m.group(1)), float(m.group(2)), int(m.group(3)), float(m.group(4)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=DEFAULT_DIR)
    ap.add_argument("--exe", default=None)
    ap.add_argument("--head", default="auto", choices=("auto", "fabric", "arm"))
    ap.add_argument("--expected", default=os.path.join(HERE, "expected_ids.json"))
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    doc = json.load(open(a.expected, encoding="utf-8"))
    cases = doc["cases"]
    exe = a.exe or os.path.join(a.dir, "bitnet_kria")
    if not os.path.exists(exe):
        print(f"check_ids: no {exe}; run  sudo ./setup.sh --build", file=sys.stderr)
        return 2
    head = a.head if a.head != "auto" else bc.pick_head(a.dir)
    tok = bc.Tok(os.path.join(a.dir, "tokenizer.json"))

    print(f"expected ids recorded {doc['recorded_on']}")
    print(f"  bitstream md5 {doc['bitstream_md5']}  model3.bin md5 {doc['model3_md5']}")
    print(f"  agreed across {len(doc['agreed_across'])} independent recorded runs")
    print(f"running {len(cases)} prompts on this board, --head {head}\n")

    groups = {}
    for c in cases:
        groups.setdefault(c["max_new"], []).append(c)

    bad = ids_seen = 0
    out_runs = []
    pt = pm = gt = gm = 0.0
    print(f"{'prompt':20s} {'ids':>5s}  {'this board':>22s}  {'reference board':>22s}  verdict")
    for max_new in sorted(groups):
        cmd = [exe, "--dir", a.dir, "--max-new", str(max_new), "--context", "1024",
               "--threads", "4", "--cache-dtype", "f32", "--head", head, "--timing"]
        if os.geteuid() != 0:
            cmd = ["sudo"] + cmd
        eng = bc.Engine(cmd, quiet=True)
        try:
            for c in groups[max_new]:
                ids = tok.encode(bc.chat_text([{"role": "system", "content": c["system"]},
                                               {"role": "user", "content": c["user"]}]))
                t0 = time.time()
                gen, stats, per_forward = eng.generate(ids)
                wall = time.time() - t0
                out_runs.append({"id": c["id"], "prompt_ids": ids, "gen_ids": gen,
                                 "stats": stats, "per_forward": per_forward, "wall_s": wall})
                r, rr = rates(stats), rates(c["recorded_stats"])
                if r:
                    pt += r[0]; pm += r[1]; gt += r[2]; gm += r[3]
                mine = f"{r[2]/(r[3]/1e3):5.2f} gen {r[0]/(r[1]/1e3):5.2f} pr" if r else "  no stats"
                theirs = f"{rr[2]/(rr[3]/1e3):5.2f} gen {rr[0]/(rr[1]/1e3):5.2f} pr" if rr else ""
                if ids != c["prompt_ids"]:
                    verdict = "PROMPT IDS DIFFER -- the tokenizer is not the checkpoint's"
                    bad += 1
                elif gen != c["gen_ids"]:
                    n = next((i for i in range(min(len(gen), len(c["gen_ids"])))
                              if gen[i] != c["gen_ids"][i]), min(len(gen), len(c["gen_ids"])))
                    got = gen[n] if n < len(gen) else None
                    want = c["gen_ids"][n] if n < len(c["gen_ids"]) else None
                    verdict = f"IDS DIFFER at {n}: {got} not {want} ({len(gen)} vs {len(c['gen_ids'])} ids)"
                    bad += 1
                else:
                    verdict = "identical"
                    ids_seen += len(gen)
                print(f"{c['id']:20s} {len(c['gen_ids']):5d}  {mine:>22s}  {theirs:>22s}  {verdict}")
                if verdict != "identical" and c.get("answer"):
                    print(f"{'':20s}   expected: {c['answer']!r}")
        finally:
            eng.close()

    print()
    if gm:
        print(f"this board       prompt {pt/(pm/1e3):6.2f} tok/s   generation {gt/(gm/1e3):6.2f} tok/s")
    print("reference board  prompt  28.78 tok/s   generation  16.78 tok/s"
          "   (five bench prompts x3, context 512, results/kria-speed2-results.txt)")
    print("                 a generated token 59.599 ms, of which the fabric holds about 77%")

    if a.out:
        json.dump({"head": head, "runs": out_runs}, open(a.out, "w", encoding="utf-8"), indent=1)
        print(f"\nwrote {a.out}")

    if bad:
        print(f"\n{bad} of {len(cases)} prompts DIFFER from the recorded ids. This board is not "
              f"reproducing the reference board.\nRun ../doctor.sh: a wrong bitstream, a wrong "
              f"model3.bin or a wrong tokenizer.json all land here.")
        return 1
    print(f"\nall {len(cases)} prompts identical, {ids_seen} generated ids, every one the "
          f"reference board's.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
