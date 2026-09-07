"""Run the quality battery on the KV260 through bitnet_kria and keep every answer verbatim.

usage (on the board, as the normal user, after 'echo oracle-kria | sudo -S -v'):
  python3 quality_board.py quality_prompts.json board-answers.json --context 1024

bitnet_kria takes --max-new once, at startup, so the prompts are grouped by their max_new and one
long-lived bitnet_kria is opened for each group: every prompt then gets exactly the max_new the
battery asks for, the same number the HF run and bitnet.cpp are given. Greedy throughout (no --temp,
so the runtime's argmax path is used). Nothing is judged here; the answer text, the ids and the
runtime's own numbers are written out as they come.
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
    ap.add_argument("prompts")
    ap.add_argument("out")
    ap.add_argument("--dir", default="/home/ubuntu/bitnet-kria")
    ap.add_argument("--context", type=int, default=1024)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--cache-dtype", default="f32")
    ap.add_argument("--no-sudo", action="store_true")
    ap.add_argument("--only", default=None, help="comma-separated prompt ids")
    a = ap.parse_args()

    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    spec = json.load(open(a.prompts, encoding="utf-8"))
    prompts, system = spec["prompts"], spec.get("system")
    if a.only:
        want = set(a.only.split(","))
        prompts = [p for p in prompts if p["id"] in want]
    tok = bc.Tok(os.path.join(a.dir, "tokenizer.json"))

    groups = {}
    for p in prompts:
        groups.setdefault(int(p["max_new"]), []).append(p)

    runs, banner = [], None
    for max_new in sorted(groups):
        cmd = [os.path.join(a.dir, "bitnet_kria"), "--dir", a.dir, "--max-new", str(max_new),
               "--context", str(a.context), "--threads", str(a.threads),
               "--cache-dtype", a.cache_dtype, "--timing"]
        if os.geteuid() != 0 and not a.no_sudo:
            cmd = (["sudo", "-A"] if os.environ.get("SUDO_ASKPASS") else ["sudo", "-n"]) + cmd
        t_open = time.time()
        engine = bc.Engine(cmd, quiet=(banner is not None))
        if banner is None:
            banner = list(engine.banner)
        print(f"--- engine for max_new={max_new} open in {time.time() - t_open:.2f} s "
              f"({len(groups[max_new])} prompts)", flush=True)
        try:
            for p in groups[max_new]:
                for rep in range(int(p.get("repeats", 1))):
                    msgs = []
                    if system:
                        msgs.append({"role": "system", "content": system})
                    msgs.append({"role": "user", "content": p["prompt"]})
                    text = bc.chat_text(msgs)
                    ids = tok.encode(text)
                    t0 = time.time()
                    gen, stats, per_forward = engine.generate(ids)
                    wall = time.time() - t0
                    answer = tok.decode([t for t in gen if t not in bc.EOS])
                    r = {"id": p["id"], "category": p["category"], "rep": rep,
                         "max_new": max_new, "prompt": p["prompt"], "system": system,
                         "prompt_ids": ids, "gen_ids": gen, "answer": answer,
                         "wall_s": round(wall, 3), "stats": stats, "per_forward": per_forward,
                         "hit_cap": len(gen) >= max_new}
                    r.update(bc.parse_stats(stats) or {})
                    runs.append(r)
                    print(f"=== {p['id']} rep {rep} [{p['category']}] max_new {max_new}")
                    print(f"> {p['prompt']}")
                    print(answer)
                    print(f"{stats}   wall {wall:.2f} s, {len(ids)} prompt ids, "
                          f"{len(gen)} generated ids, hit_cap {r['hit_cap']}")
                    sys.stdout.flush()
        finally:
            engine.close()

    json.dump({"way": "board", "banner": banner, "context": a.context, "threads": a.threads,
               "cache_dtype": a.cache_dtype, "system": system, "runs": runs},
              open(a.out, "w", encoding="utf-8"), indent=1, ensure_ascii=False)

    det = [r for r in runs if r["id"] == "q24_determinism"]
    if len(det) >= 2:
        same_ids = det[0]["gen_ids"] == det[1]["gen_ids"]
        same_txt = det[0]["answer"] == det[1]["answer"]
        print(f"\ndeterminism q24: identical gen_ids {same_ids}, identical text {same_txt}, "
              f"{len(det[0]['gen_ids'])} vs {len(det[1]['gen_ids'])} ids")
    ptok = sum(r.get("prompt_tokens", 0) for r in runs)
    pms = sum(r.get("prompt_ms", 0.0) for r in runs)
    gtok = sum(r.get("gen_tokens", 0) for r in runs)
    gms = sum(r.get("gen_ms", 0.0) for r in runs)
    print(f"{len(runs)} runs; aggregate {ptok} prompt tokens in {pms / 1e3:.3f} s "
          f"= {ptok / (pms / 1e3):.2f} tok/s; {gtok} generated tokens in {gms / 1e3:.3f} s "
          f"= {gtok / (gms / 1e3):.2f} tok/s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
