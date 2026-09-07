"""Run every battery and bench prompt through bitnet_kria in one head mode and keep the ids.

usage (on the board)
  python3 headcheck.py out.json --head arm
  python3 headcheck.py out.json --head fabric
  python3 headcheck.py --compare arm.json fabric.json
  python3 headcheck.py --numbers fabric.json

The 26 prompts of quality_prompts.json (with its system message) and the 5 of prompts.json are run
greedily through one long-lived bitnet_kria a max_new group, exactly as quality_board.py and
runbench.py run them. Only the generated ids and the runtime's own timing lines are kept; --compare
puts two such files side by side, prompt by prompt and id by id.
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "/home/ubuntu/bitnet-kria")
import bitnet_chat as bc


def load_prompts(dirname):
    """The 26 battery prompts, and the 5 bench prompts when a prompts.json is beside them.

    quality_prompts.json ships in tools/; prompts.json does not, because its five prompts were only
    ever a file on the reference board. Without it this runs the 26 battery prompts alone and says
    so, and every 'battery:' id still lines up with the recorded runs in results/.
    """
    out = []
    spec_path = os.path.join(dirname, "quality_prompts.json")
    if not os.path.exists(spec_path):
        spec_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "quality_prompts.json")
    spec = json.load(open(spec_path, encoding="utf-8"))
    system = spec.get("system")
    for p in spec["prompts"]:
        out.append({"id": "battery:" + p["id"], "user": p["prompt"], "system": system,
                    "max_new": int(p["max_new"])})
    bench = os.path.join(dirname, "prompts.json")
    if os.path.exists(bench):
        for p in json.load(open(bench, encoding="utf-8")):
            out.append({"id": "bench:" + p["name"], "user": p["user"], "system": p.get("system"),
                        "max_new": int(p.get("max_new", 128))})
    else:
        print(f"[no {bench}: running the 26 battery prompts only. A prompts.json is a list of "
              f"{{name, user, system, max_new}}; the five the recorded runs used are not in this "
              f"repository.]", file=sys.stderr)
    return out


def compare(a_path, b_path):
    a = json.load(open(a_path, encoding="utf-8"))
    b = json.load(open(b_path, encoding="utf-8"))
    ra = {r["id"]: r for r in a["runs"]}
    rb = {r["id"]: r for r in b["runs"]}
    keys = sorted(set(ra) | set(rb))
    ids = 0
    bad = 0
    for k in keys:
        if k not in ra or k not in rb:
            print(f"{k:28s} MISSING from {'a' if k not in ra else 'b'}")
            bad += 1
            continue
        x, y = ra[k]["gen_ids"], rb[k]["gen_ids"]
        ids += max(len(x), len(y))
        if x == y:
            print(f"{k:28s} {len(x):4d} ids identical")
        else:
            first = next((i for i in range(min(len(x), len(y))) if x[i] != y[i]), min(len(x), len(y)))
            print(f"{k:28s} {len(x):4d} vs {len(y):4d} ids DIFFER, first at {first}: "
                  f"{x[first] if first < len(x) else None} vs {y[first] if first < len(y) else None}")
            bad += 1
    print(f"{len(keys)} prompts, {ids} generated ids compared: "
          f"{'ALL IDENTICAL' if not bad else str(bad) + ' PROMPTS DIFFER'}")
    return 1 if bad else 0


def numbers(path):
    import re
    d = json.load(open(path, encoding="utf-8"))
    pt = pm = gt = gm = 0
    head_ms = head_calls = 0.0
    stage = {"s1": 0.0, "act": 0.0, "top": 0.0, "s2": 0.0}
    for r in d["runs"]:
        m = re.match(r"prompt (\d+) tokens ([\d.]+) ms .*generated (\d+) ([\d.]+) ms", r["stats"])
        if m:
            pt += int(m.group(1))
            pm += float(m.group(2))
            gt += int(m.group(3))
            gm += float(m.group(4))
        m = re.search(r"head ([\d.]+) ms a call over (\d+) calls", r["per_forward"])
        if m:
            head_ms += float(m.group(1)) * int(m.group(2))
            head_calls += int(m.group(2))
        m = re.search(r"stage 1 ternary stream ([\d.]+), its activation load ([\d.]+), sync and "
                      r"top-\d+ ([\d.]+), stage 2 exact rescore ([\d.]+), head total ([\d.]+)",
                      r.get("per_forward", ""))
        if m:
            n = int(re.search(r"over (\d+) calls", r["per_forward"]).group(1))
            for i, k in enumerate(("s1", "act", "top", "s2")):
                stage[k] += float(m.group(i + 1)) * n
    print(f"{path}: head {d['head']}")
    print(f"  prompt     {pt:5d} tokens in {pm / 1e3:8.3f} s = {pt / (pm / 1e3):6.2f} tok/s")
    print(f"  generated  {gt:5d} tokens in {gm / 1e3:8.3f} s = {gt / (gm / 1e3):6.2f} tok/s")
    if head_calls:
        print(f"  head       {head_calls:5.0f} calls, {head_ms / head_calls:.3f} ms a call, "
              f"{head_ms / 1e3:.3f} s in all")
        if stage["s1"]:
            print(f"  head stages a call: stage 1 stream {stage['s1'] / head_calls:.3f}, its activation "
                  f"load {stage['act'] / head_calls:.3f}, sync and top-K {stage['top'] / head_calls:.3f}, "
                  f"stage 2 rescore {stage['s2'] / head_calls:.3f}")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out", nargs="?")
    ap.add_argument("--compare", nargs=2)
    ap.add_argument("--numbers", default=None)
    ap.add_argument("--head", default="arm")
    ap.add_argument("--dir", default="/home/ubuntu/bitnet-kria")
    ap.add_argument("--exe", default=None)
    ap.add_argument("--context", type=int, default=1024)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--cache-dtype", default="f32")
    ap.add_argument("--k", type=int, default=0)
    ap.add_argument("--no-head-flag", action="store_true", help="the baseline binary has no --head")
    a = ap.parse_args()

    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if a.numbers:
        return numbers(a.numbers)
    if a.compare:
        return compare(a.compare[0], a.compare[1])
    if not a.out:
        ap.error("an output path is needed")

    exe = a.exe or os.path.join(a.dir, "bitnet_kria")
    prompts = load_prompts(a.dir)
    tok = bc.Tok(os.path.join(a.dir, "tokenizer.json"))
    groups = {}
    for p in prompts:
        groups.setdefault(p["max_new"], []).append(p)

    runs, banner = [], None
    for max_new in sorted(groups):
        cmd = [exe, "--dir", a.dir, "--max-new", str(max_new), "--context", str(a.context),
               "--threads", str(a.threads), "--cache-dtype", a.cache_dtype, "--timing"]
        if not a.no_head_flag:
            cmd += ["--head", a.head]
        if a.k:
            cmd += ["--head-k", str(a.k)]
        if os.geteuid() != 0:
            cmd = (["sudo", "-A"] if os.environ.get("SUDO_ASKPASS") else ["sudo", "-n"]) + cmd
        engine = bc.Engine(cmd, quiet=(banner is not None))
        if banner is None:
            banner = list(engine.banner)
        print(f"--- engine for max_new={max_new} ({len(groups[max_new])} prompts)", flush=True)
        try:
            for p in groups[max_new]:
                msgs = []
                if p["system"]:
                    msgs.append({"role": "system", "content": p["system"]})
                msgs.append({"role": "user", "content": p["user"]})
                ids = tok.encode(bc.chat_text(msgs))
                t0 = time.time()
                gen, stats, per_forward = engine.generate(ids)
                runs.append({"id": p["id"], "max_new": max_new, "prompt_ids": ids, "gen_ids": gen,
                             "answer": tok.decode([t for t in gen if t not in bc.EOS]),
                             "wall_s": round(time.time() - t0, 3),
                             "stats": stats, "per_forward": per_forward})
                print(f"{p['id']:28s} {len(ids):4d} prompt ids, {len(gen):4d} generated | {stats}",
                      flush=True)
        finally:
            engine.close()

    json.dump({"head": a.head, "exe": exe, "banner": banner, "runs": runs},
              open(a.out, "w", encoding="utf-8"), indent=1)
    print(f"wrote {a.out}: {len(runs)} prompts, {sum(len(r['gen_ids']) for r in runs)} generated ids")
    return 0


if __name__ == "__main__":
    sys.exit(main())
