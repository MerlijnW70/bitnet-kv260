"""Merge the three quality runs into one quality-answers.json.

The copy this repository's numbers were read out of is results/quality-answers.json; this script
is what made it.

usage (on the PC):
  python quality_merge.py quality_prompts.json quality-answers.json \
      --board board-answers.json --hf hf-answers.json --cpp bitnetcpp-answers.json

Nothing is scored. Every answer goes in verbatim, together with the token counts and the seconds
each way took, whatever the answer looks like. Two mechanical measurements are added because the
battery asks for them, and both are measurements, not judgements:
  - determinism: whether the two runs of the repeated prompt gave byte-identical text and ids;
  - repetition: the longest block of words that repeats back to back inside an answer, quoted, plus
    how many of the answer's sentences are distinct.
Prompts or ways that produced nothing are recorded as null with the reason.
"""
import argparse
import json
import re
import sys
from collections import Counter


def load(path):
    if not path:
        return None
    try:
        return json.load(open(path, encoding="utf-8"))
    except FileNotFoundError:
        return None


def repetition(text):
    """The longest word block that repeats immediately, how many times, and how many of the
    answer's sentences are distinct. A back-to-back repeat of a long block is what 'fell into
    repetition' means for a greedy decoder that cannot escape a loop."""
    words = text.split()
    best = {"block_words": 0, "repeats": 1, "text": ""}
    n_max = max(1, len(words) // 2)
    for n in range(1, min(n_max, 40) + 1):
        for i in range(0, len(words) - 2 * n + 1):
            block = words[i:i + n]
            k = 1
            while words[i + k * n:i + (k + 1) * n] == block:
                k += 1
            if k >= 2 and (n * k, n) > (best["block_words"] * best["repeats"], best["block_words"]):
                best = {"block_words": n, "repeats": k, "text": " ".join(block)}
    sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+", text) if s.strip()]
    counts = Counter(sents)
    return {"longest_repeated_block_words": best["block_words"],
            "repeated_back_to_back_times": best["repeats"],
            "repeated_text": best["text"] if best["repeats"] >= 2 else "",
            "words": len(words), "sentences": len(sents), "distinct_sentences": len(set(sents)),
            "duplicate_sentences": len(sents) - len(set(sents)),
            "duplicate_sentence_texts": [{"text": s, "times": c}
                                         for s, c in counts.items() if c > 1]}


def board_rec(r):
    return {"answer": r["answer"], "prompt_tokens": r.get("prompt_tokens"),
            "gen_tokens": r.get("gen_tokens"), "hit_cap": r.get("hit_cap"),
            "prompt_s": round(r["prompt_ms"] / 1e3, 3) if r.get("prompt_ms") else None,
            "gen_s": round(r["gen_ms"] / 1e3, 3) if r.get("gen_ms") else None,
            "prompt_tps": r.get("prompt_tps"), "gen_tps": r.get("gen_tps"),
            "wall_s": r.get("wall_s"), "gen_ids": r.get("gen_ids")}


def hf_rec(r):
    return {"answer": r["answer"], "prompt_tokens": r.get("prompt_tokens"),
            "gen_tokens": r.get("gen_tokens"), "hit_cap": r.get("hit_cap"),
            "gen_s": r.get("wall_s"), "gen_tps": r.get("gen_tps"),
            "wall_s": r.get("wall_s"), "gen_ids": r.get("gen_ids")}


def cpp_rec(r):
    return {"answer": r["answer"], "prompt_tokens": r.get("prompt_tokens"),
            "gen_tokens": r.get("gen_tokens"), "hit_cap": r.get("hit_cap"),
            "prompt_s": round(r["prompt_ms"] / 1e3, 3) if r.get("prompt_ms") else None,
            "gen_s": round(r["gen_ms"] / 1e3, 3) if r.get("gen_ms") else None,
            "prompt_tps": r.get("prompt_tps"), "gen_tps": r.get("gen_tps"),
            "load_s": round(r["load_ms"] / 1e3, 3) if r.get("load_ms") else None,
            "wall_s": r.get("wall_s"), "returncode": r.get("returncode"),
            "gen_ids": r.get("gen_ids")}


def compare_ids(board_runs, hf_runs):
    """Where the board's greedy ids first leave the bf16 checkpoint's. The two are different
    arithmetic (int8 activations and ternary integer matvecs against bf16), so they are expected to
    agree until a near-tie in the logits flips; this records the position, it does not judge it."""
    if not board_runs or not hf_runs:
        return None
    b, h = board_runs[0].get("gen_ids") or [], hf_runs[0].get("gen_ids") or []
    first = None
    for i in range(min(len(b), len(h))):
        if b[i] != h[i]:
            first = i
            break
    if first is None and len(b) != len(h):
        first = min(len(b), len(h))
    return {"board_gen_tokens": len(b), "hf_gen_tokens": len(h),
            "identical_gen_ids": b == h,
            "identical_text": board_runs[0]["answer"] == hf_runs[0]["answer"],
            "first_divergent_position": first,
            "prompt_ids_identical": board_runs[0].get("prompt_ids") == hf_runs[0].get("prompt_ids")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("prompts")
    ap.add_argument("out")
    ap.add_argument("--board")
    ap.add_argument("--hf")
    ap.add_argument("--cpp")
    ap.add_argument("--measured", default="2026-09-06")
    a = ap.parse_args()

    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    spec = json.load(open(a.prompts, encoding="utf-8"))
    ways = {"board": (load(a.board), board_rec), "hf_bf16_rtx4080": (load(a.hf), hf_rec),
            "bitnet.cpp": (load(a.cpp), cpp_rec)}
    index = {}
    for name, (doc, _) in ways.items():
        index[name] = {}
        if doc:
            for r in doc["runs"]:
                index[name].setdefault(r["id"], []).append(r)

    failures = []
    out_prompts = []
    for p in spec["prompts"]:
        reps = int(p.get("repeats", 1))
        entry = {"id": p["id"], "category": p["category"], "prompt": p["prompt"],
                 "max_new": p["max_new"], "must_contain": p["must_contain"],
                 "repeats": reps}
        for name, (doc, mk) in ways.items():
            rs = index[name].get(p["id"], [])
            if not rs:
                entry[name] = None
                failures.append(f"{p['id']}: no run recorded for {name}"
                                f"{' (its answer file is missing)' if doc is None else ''}")
                continue
            recs = [mk(r) for r in rs]
            for rec in recs:
                rec["repetition"] = repetition(rec["answer"])
            entry[name] = recs[0] if reps == 1 else {"runs": recs}
            if not recs[0]["answer"].strip():
                failures.append(f"{p['id']}: {name} produced an empty answer")
            if rs[0].get("returncode") not in (None, 0):
                failures.append(f"{p['id']}: {name} exited {rs[0]['returncode']}")
        entry["board_vs_hf"] = compare_ids(index["board"].get(p["id"]),
                                           index["hf_bf16_rtx4080"].get(p["id"]))
        out_prompts.append(entry)

    det = {}
    for name, (doc, mk) in ways.items():
        rs = index[name].get("q24_determinism", [])
        if len(rs) >= 2:
            det[name] = {"runs": len(rs),
                         "identical_text": rs[0]["answer"] == rs[1]["answer"],
                         "identical_gen_ids": rs[0].get("gen_ids") == rs[1].get("gen_ids"),
                         "gen_tokens": [r.get("gen_tokens") for r in rs],
                         "answers": [r["answer"] for r in rs]}
        else:
            det[name] = {"runs": len(rs), "identical_text": None, "identical_gen_ids": None}

    long_gen = {}
    for name, (doc, mk) in ways.items():
        rs = index[name].get("q23_long", [])
        if rs:
            rep = repetition(rs[0]["answer"])
            long_gen[name] = {"gen_tokens": rs[0].get("gen_tokens"),
                              "hit_cap": rs[0].get("hit_cap"), "repetition": rep,
                              "criterion": "fell_into_repetition is true when some block of words "
                                           "repeats back to back three times or more, or when two or "
                                           "more whole sentences appear twice; the repeated text is "
                                           "quoted in the repetition record",
                              "fell_into_repetition": rep["repeated_back_to_back_times"] >= 3
                              or rep["duplicate_sentences"] >= 2}

    cmp = [e["board_vs_hf"] for e in out_prompts if e.get("board_vs_hf")]
    agree = {"prompts_compared": len(cmp),
             "prompt_ids_identical": sum(1 for c in cmp if c["prompt_ids_identical"]),
             "generation_identical": sum(1 for c in cmp if c["identical_gen_ids"]),
             "text_identical": sum(1 for c in cmp if c["identical_text"]),
             "first_divergent_positions": sorted(c["first_divergent_position"] for c in cmp
                                                 if c["first_divergent_position"] is not None)}

    doc = {"measured": a.measured,
           "what": "BitNet b1.58 2B4T answering the quality battery three ways: on the KV260's grown "
                   "fabric (runtime/bitnet_kria.c), on the HF checkpoint in bf16 on an RTX 4080, and on "
                   "bitnet.cpp's own ARM kernels on the same KV260. Greedy everywhere, the same chat "
                   "template, the same system message and the same max_new. Collected, not judged.",
           "battery": "tools/quality_prompts.json",
           "system": spec.get("system"),
           "ways": {name: ({k: v for k, v in (doc or {}).items() if k != "runs"} or None)
                    for name, (doc, _) in ways.items()},
           "determinism_q24": det, "long_generation_q23": long_gen,
           "board_vs_hf_agreement": agree, "failures": failures, "prompts": out_prompts}
    json.dump(doc, open(a.out, "w", encoding="utf-8"), indent=1, ensure_ascii=False)

    n = {name: sum(1 for e in out_prompts if e[name]) for name in ways}
    print(f"{len(out_prompts)} prompts; answers kept: " +
          ", ".join(f"{k} {v}" for k, v in n.items()))
    for name, d in det.items():
        print(f"determinism q24 {name}: {d}")
    for name, d in long_gen.items():
        r = d["repetition"]
        print(f"long q23 {name}: {d['gen_tokens']} tokens, hit_cap {d['hit_cap']}, "
              f"fell_into_repetition {d['fell_into_repetition']}, longest back-to-back block "
              f"{r['longest_repeated_block_words']} words x{r['repeated_back_to_back_times']}, "
              f"{r['duplicate_sentences']} duplicate sentences of {r['sentences']}")
    print(f"board against HF: prompt ids identical on {agree['prompt_ids_identical']} of "
          f"{agree['prompts_compared']}, whole generation identical on {agree['generation_identical']}, "
          f"answer text identical on {agree['text_identical']}; first divergent positions "
          f"{agree['first_divergent_positions']}")
    for f in failures:
        print("FAILURE:", f)
    print(f"written to {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
