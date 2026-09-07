"""The five bench prompts, three times each, through one long-lived bitnet_kria of your choosing.

usage (on the board)
  python3 provebench.py --exe ./bitnet_kria --head fabric --out prove-after.json
  python3 provebench.py --exe baseline/bitnet_kria --no-head-flag --out prove-before.json

Same prompts, repeats, context and summary as runbench.py, with the head mode and the binary chosen on
the command line so that a baseline binary (which has no --head) and today's can be measured back to
back.  One engine is held open for the whole run, so the 521 MB of weights are read once; every prompt
line resets the KV cache and the timing accumulators, so a per-forward line belongs to its own prompt.
Nothing here judges the ids: they are written to the json and the caller compares them.
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "/home/ubuntu/bitnet-kria")
import bitnet_chat as bc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="/home/ubuntu/bitnet-kria")
    ap.add_argument("--prompts", default=None)
    ap.add_argument("--exe", default="./bitnet_kria")
    ap.add_argument("--out", default=None)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--context", type=int, default=512)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--cache-dtype", default="f32")
    ap.add_argument("--head", default="fabric")
    ap.add_argument("--head-k", type=int, default=0)
    ap.add_argument("--no-head-flag", action="store_true")
    ap.add_argument("--compare", nargs=2)
    a = ap.parse_args()

    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if a.compare:
        x = json.load(open(a.compare[0], encoding="utf-8"))["runs"]
        y = json.load(open(a.compare[1], encoding="utf-8"))["runs"]
        keys = sorted({(r["rep"], r["name"]) for r in x} | {(r["rep"], r["name"]) for r in y})
        dx = {(r["rep"], r["name"]): r["gen_ids"] for r in x}
        dy = {(r["rep"], r["name"]): r["gen_ids"] for r in y}
        ids = bad = 0
        for k in keys:
            u, v = dx.get(k), dy.get(k)
            ids += max(len(u or []), len(v or []))
            if u == v and u is not None:
                print(f"rep {k[0]} {k[1]:9s} {len(u):4d} ids identical")
            else:
                first = next((i for i in range(min(len(u or []), len(v or []))) if u[i] != v[i]), -1)
                print(f"rep {k[0]} {k[1]:9s} DIFFER, first at {first}")
                bad += 1
        print(f"{len(keys)} runs, {ids} generated ids compared: "
              f"{'ALL IDENTICAL' if not bad else str(bad) + ' RUNS DIFFER'}")
        return 1 if bad else 0

    ppath = a.prompts or os.path.join(a.dir, "prompts.json")
    if not os.path.exists(ppath):
        raise SystemExit(
            f"provebench: no {ppath}.\n"
            f"The five bench prompts this repository's speed numbers were taken on lived in a\n"
            f"prompts.json on the reference board and are not shipped here. Write your own -- a\n"
            f"JSON list of {{\"name\", \"user\", \"system\", \"max_new\"}} -- and pass it with --prompts,\n"
            f"or use headcheck.py, whose 26-prompt battery IS in tools/quality_prompts.json.")
    prompts = json.load(open(ppath, encoding="utf-8"))
    tok = bc.Tok(os.path.join(a.dir, "tokenizer.json"))
    max_new = max(p.get("max_new", 128) for p in prompts)
    cmd = [a.exe, "--dir", a.dir, "--max-new", str(max_new), "--context", str(a.context),
           "--threads", str(a.threads), "--cache-dtype", a.cache_dtype, "--timing"]
    if not a.no_head_flag:
        cmd += ["--head", a.head]
    if a.head_k:
        cmd += ["--head-k", str(a.head_k)]
    if os.geteuid() != 0:
        cmd = (["sudo", "-A"] if os.environ.get("SUDO_ASKPASS") else ["sudo", "-n"]) + cmd

    t_open = time.time()
    engine = bc.Engine(cmd)
    print(f"engine open in {time.time() - t_open:.2f} s: {' '.join(cmd)}")
    banner = list(engine.banner)
    for line in banner:
        print("  | " + line)

    runs = []
    try:
        for rep in range(a.repeats):
            for p in prompts:
                msgs = []
                if p.get("system"):
                    msgs.append({"role": "system", "content": p["system"]})
                msgs.append({"role": "user", "content": p["user"]})
                ids = tok.encode(bc.chat_text(msgs))
                t0 = time.time()
                gen, stats, per_forward = engine.generate(ids)
                wall = time.time() - t0
                r = {"rep": rep, "name": p["name"], "user": p["user"], "prompt_ids": ids,
                     "gen_ids": gen, "answer": tok.decode([t for t in gen if t not in bc.EOS]),
                     "wall_s": wall, "stats": stats, "per_forward": per_forward}
                r.update(bc.parse_stats(stats) or {})
                runs.append(r)
                print(f"=== rep {rep} {p['name']}: {len(ids)} prompt ids, {len(gen)} generated, "
                      f"wall {wall:.2f} s")
                print("  " + stats)
                print("  " + per_forward.replace("\n", "\n  "))
                sys.stdout.flush()
    finally:
        engine.close()

    if a.out:
        json.dump({"exe": a.exe, "head": None if a.no_head_flag else a.head, "banner": banner,
                   "runs": runs}, open(a.out, "w", encoding="utf-8"), indent=1)
        print("wrote " + a.out)

    print("\n=== SUMMARY (mean / min over repeats)")
    for p in prompts:
        rs = [r for r in runs if r["name"] == p["name"]]
        pt = [r["prompt_tps"] for r in rs if "prompt_tps" in r]
        gt = [r["gen_tps"] for r in rs if "gen_tps" in r]
        same = len({tuple(r["gen_ids"]) for r in rs}) == 1
        print(f"{p['name']:9s} {rs[0]['prompt_tokens']:4d} prompt ids, {rs[0]['gen_tokens']:4d} "
              f"generated: prompt mean {sum(pt)/len(pt):6.2f} min {min(pt):6.2f} tok/s; generation "
              f"mean {sum(gt)/len(gt):5.2f} min {min(gt):5.2f} tok/s; identical ids across repeats: "
              f"{same}")
    allp = [r["prompt_tps"] for r in runs if "prompt_tps" in r]
    allg = [r["gen_tps"] for r in runs if "gen_tps" in r]
    ptok = sum(r["prompt_tokens"] for r in runs)
    pms = sum(r["prompt_ms"] for r in runs)
    gtok = sum(r["gen_tokens"] for r in runs)
    gms = sum(r["gen_ms"] for r in runs)
    print(f"over {len(runs)} runs: prompt mean {sum(allp)/len(allp):.2f} min {min(allp):.2f} tok/s; "
          f"generation mean {sum(allg)/len(allg):.2f} min {min(allg):.2f} tok/s")
    print(f"aggregate: {ptok} prompt tokens in {pms/1e3:.3f} s = {ptok/(pms/1e3):.2f} tok/s; "
          f"{gtok} generated tokens in {gms/1e3:.3f} s = {gtok/(gms/1e3):.2f} tok/s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
