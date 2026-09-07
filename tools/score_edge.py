"""Score the edge triage job the same way for all three ways of running it.

  board      edge-run.json   from runtime/edge_monitor.py: bitnet_kria, every matrix product in the
                             KV260's fabric
  cpp        edge-cpp-answers.json   from quality_bitnetcpp.py: bitnet.cpp on the same board's four
                             A53 cores (the speed baseline)
  gpu        edge-hf-answers.json    from quality_hf.py: the HF checkpoint in bf16 on an RTX 4080
                             (the quality baseline, to separate "the model is small" from "the board
                             is wrong")

One rule for all three: the answer, stripped of surrounding whitespace, must itself parse as a JSON
object. A lenient first-{ to last-} parse is computed too but only ever reported as a diagnostic.
Accuracy is over the thirty lines edge_alarms.json calls clear; the ten it marks ambiguous are
scored and printed separately and are never folded into the headline.

usage:
  python score_edge.py --alarms edge_alarms.json --board edge-run.json \
      --cpp edge-cpp-answers.json --gpu edge-hf-answers.json --out edge-scores.json
"""
import argparse
import json
import sys

CATEGORIES = ("mechanical", "electrical", "thermal", "sensor", "network", "software", "none")
SEVERITIES = ("critical", "warning", "info")


def strict_json(text):
    try:
        v = json.loads((text or "").strip())
    except (ValueError, TypeError):
        return None
    return v if isinstance(v, dict) else None


def loose_json(text):
    t = text or ""
    i, j = t.find("{"), t.rfind("}")
    if i < 0 or j <= i:
        return None
    try:
        v = json.loads(t[i:j + 1])
    except ValueError:
        return None
    return v if isinstance(v, dict) else None


def field(obj, key):
    v = obj.get(key) if isinstance(obj, dict) else None
    return v.strip().lower() if isinstance(v, str) else None


def score_one(ans, want_cat, want_sev):
    s = strict_json(ans)
    lo = s if s is not None else loose_json(ans)
    cat, sev = field(lo or {}, "category"), field(lo or {}, "severity")
    act = (lo or {}).get("action")
    return {"json_strict_ok": s is not None, "json_recovered_ok": lo is not None,
            "keys_exact": isinstance(lo, dict) and set(lo) == {"category", "severity", "action"},
            "got_category": cat, "got_severity": sev,
            "category_in_vocab": cat in CATEGORIES, "severity_in_vocab": sev in SEVERITIES,
            "category_ok": cat == want_cat, "severity_ok": sev == want_sev,
            "action_ok": isinstance(act, str) and bool(act.strip()), "got_action": act}


def frac(rows, key):
    return round(sum(1 for r in rows if r.get(key)) / len(rows), 4) if rows else None


def stat(vals):
    if not vals:
        return None
    v = sorted(vals)
    return {"mean": round(sum(v) / len(v), 1), "min": round(v[0], 1), "max": round(v[-1], 1),
            "median": round(v[len(v) // 2], 1)}


def collect_board(doc):
    """Only the main pass: the --saving pass re-asks the first five alarms and tags them."""
    out = {}
    for r in doc["runs"]:
        if r.get("job") != "triage" or r.get("mode"):
            continue
        out[r["id"]] = {"answer": r["answer"], "ms": r["answer_ms"],
                        "prompt_tokens": r.get("prompt_tokens", r.get("prompt_ids")),
                        "gen_tokens": r.get("gen_tokens", r.get("gen_ids")),
                        "prompt_ids_n": r.get("prompt_ids"),
                        "gen_ids": [t for t in (r.get("gen_token_ids") or [])
                                    if t not in (128001, 128009)],
                        "joules": (r.get("power") or {}).get("joules"),
                        "watts": (r.get("power") or {}).get("mean_w"),
                        "net_delta": r.get("net_delta")}
    return out


def collect_runs(doc, ms_key):
    out = {}
    for r in doc["runs"]:
        ms = r[ms_key] * 1e3 if ms_key == "wall_s" else r[ms_key]
        out[r["id"]] = {"answer": r["answer"], "ms": ms,
                        "prompt_tokens": r.get("prompt_tokens"),
                        "gen_tokens": r.get("gen_tokens") or len(r.get("gen_ids") or []),
                        "prompt_ids_n": r.get("prompt_tokens"),
                        "gen_ids": [t for t in (r.get("gen_ids") or [])
                                    if t not in (128001, 128009)],
                        "joules": (r.get("power") or {}).get("joules"),
                        "watts": (r.get("power") or {}).get("mean_w")}
    return out


def agreement(a, b, na, nb):
    """How far two ways agree: same prompt length, same generated token ids, same text.

    The token comparison is the sharp one. Two ways that emit the identical id sequence on a greedy
    decode are running the same arithmetic to the same answer; anything softer (same category, same
    gist) could hide a real difference in the model."""
    ids = sorted(set(a) & set(b))
    same_len = sum(1 for i in ids if a[i]["prompt_ids_n"] == b[i]["prompt_ids_n"])
    have = [i for i in ids if a[i]["gen_ids"] and b[i]["gen_ids"]]
    same_ids = [i for i in have if a[i]["gen_ids"] == b[i]["gen_ids"]]
    same_text = [i for i in ids if a[i]["answer"].strip() == b[i]["answer"].strip()]
    return {"pair": f"{na} vs {nb}", "n": len(ids),
            "same_prompt_length": f"{same_len} of {len(ids)}",
            "same_generated_ids": f"{len(same_ids)} of {len(have)}",
            "same_answer_text": f"{len(same_text)} of {len(ids)}",
            "differing_ids": [i for i in have if i not in same_ids]}


def summarise(name, got, alarms):
    rows = []
    for a in alarms:
        g = got.get(a["id"])
        if g is None:
            continue
        r = {"id": a["id"], "ambiguous": bool(a.get("ambiguous")),
             "want_category": a["category"], "want_severity": a["severity"],
             "line": a["line"], "answer": g["answer"], "ms": g["ms"],
             "prompt_tokens": g["prompt_tokens"], "gen_tokens": g["gen_tokens"],
             "joules": g["joules"], "watts": g["watts"]}
        r.update(score_one(g["answer"], a["category"], a["severity"]))
        rows.append(r)
    clear = [r for r in rows if not r["ambiguous"]]
    amb = [r for r in rows if r["ambiguous"]]
    js = [r["joules"] for r in rows if r["joules"] is not None]
    out = {
        "way": name, "answers": len(rows), "clear": len(clear), "ambiguous": len(amb),
        "json_strict_frac": frac(rows, "json_strict_ok"),
        "json_recovered_frac": frac(rows, "json_recovered_ok"),
        "keys_exact_frac": frac(rows, "keys_exact"),
        "category_in_vocab_frac": frac(rows, "category_in_vocab"),
        "severity_in_vocab_frac": frac(rows, "severity_in_vocab"),
        "action_nonempty_frac": frac(rows, "action_ok"),
        "category_acc_clear": frac(clear, "category_ok"),
        "severity_acc_clear": frac(clear, "severity_ok"),
        "both_acc_clear": round(sum(1 for r in clear if r["category_ok"] and r["severity_ok"])
                                / len(clear), 4) if clear else None,
        "category_acc_ambiguous": frac(amb, "category_ok"),
        "severity_acc_ambiguous": frac(amb, "severity_ok"),
        "category_acc_all40": frac(rows, "category_ok"),
        "severity_acc_all40": frac(rows, "severity_ok"),
        "ms": stat([r["ms"] for r in rows]),
        "prompt_tokens_mean": round(sum(r["prompt_tokens"] or 0 for r in rows) / len(rows), 1),
        "gen_tokens_mean": round(sum(r["gen_tokens"] or 0 for r in rows) / len(rows), 1),
    }
    if js:
        out["joules"] = {"mean": round(sum(js) / len(js), 3), "min": round(min(js), 3),
                         "max": round(max(js), 3), "total": round(sum(js), 1)}
    return out, rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--alarms", default="edge_alarms.json")
    ap.add_argument("--board", default="edge-run.json")
    ap.add_argument("--cpp", default=None)
    ap.add_argument("--gpu", default=None)
    ap.add_argument("--out", default="edge-scores.json")
    a = ap.parse_args()

    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    alarms = json.load(open(a.alarms, encoding="utf-8"))["alarms"]
    ways, rowsets, got = {}, {}, {}
    if a.board:
        doc = json.load(open(a.board, encoding="utf-8"))
        got["board"] = collect_board(doc)
        ways["board"], rowsets["board"] = summarise("board (KV260 fabric)", got["board"], alarms)
    if a.cpp:
        doc = json.load(open(a.cpp, encoding="utf-8"))
        got["cpp"] = collect_runs(doc, "wall_s")
        ways["cpp"], rowsets["cpp"] = summarise("bitnet.cpp (KV260 A53 x4)", got["cpp"], alarms)
    if a.gpu:
        doc = json.load(open(a.gpu, encoding="utf-8"))
        got["gpu"] = collect_runs(doc, "wall_s")
        ways["gpu"], rowsets["gpu"] = summarise("hf bf16 (RTX 4080)", got["gpu"], alarms)

    keys = ["answers", "json_strict_frac", "keys_exact_frac", "category_acc_clear",
            "severity_acc_clear", "both_acc_clear", "category_acc_ambiguous",
            "severity_acc_ambiguous", "prompt_tokens_mean", "gen_tokens_mean"]
    names = list(ways)
    print(f"{'':26s}" + "".join(f"{n:>22s}" for n in names))
    for k in keys:
        print(f"{k:26s}" + "".join(f"{str(ways[n].get(k)):>22s}" for n in names))
    for k in ("mean", "min", "max"):
        print(f"{'ms_' + k:26s}"
              + "".join(f"{str((ways[n].get('ms') or {}).get(k)):>22s}" for n in names))
    for k in ("mean", "min", "max"):
        print(f"{'joules_' + k:26s}"
              + "".join(f"{str((ways[n].get('joules') or {}).get(k)):>22s}" for n in names))

    disagree = []
    ids = [x["id"] for x in alarms]
    by = {n: {r["id"]: r for r in rowsets[n]} for n in names}
    print("\n--- per alarm: truth | " + " | ".join(names))
    for i in ids:
        a0 = next(x for x in alarms if x["id"] == i)
        cells = []
        for n in names:
            r = by[n].get(i)
            cells.append(f"{r['got_category']}/{r['got_severity']}" if r else "-")
        truth = f"{a0['category']}/{a0['severity']}"
        flag = "AMBIG" if a0.get("ambiguous") else "     "
        mark = "" if len(set(cells)) == 1 else "  <-- ways differ"
        print(f"  {i} {flag} {truth:22s} " + " ".join(f"{c:22s}" for c in cells) + mark)
        if len(set(cells)) > 1:
            disagree.append({"id": i, "truth": truth, "ambiguous": bool(a0.get("ambiguous")),
                             **{n: cells[k] for k, n in enumerate(names)}})

    if "board" in by and "gpu" in by:
        print("\n--- the board's mistakes on the thirty clear lines, against the RTX 4080")
        for i in ids:
            a0 = next(x for x in alarms if x["id"] == i)
            if a0.get("ambiguous"):
                continue
            b, g = by["board"].get(i), by["gpu"].get(i)
            if not b or (b["category_ok"] and b["severity_ok"]):
                continue
            gsame = g and g["got_category"] == b["got_category"] and \
                g["got_severity"] == b["got_severity"]
            gok = g and g["category_ok"] and g["severity_ok"]
            print(f"  {i} want {a0['category']}/{a0['severity']:8s} "
                  f"board {b['got_category']}/{b['got_severity']:8s} "
                  f"gpu {g['got_category']}/{g['got_severity']:8s} "
                  f"{'(gpu agrees with the board)' if gsame else ''}"
                  f"{'(gpu is right)' if gok else ''}")

    agree = []
    print("\n--- how far the ways agree, answer by answer")
    for i, n1 in enumerate(names):
        for n2 in names[i + 1:]:
            ag = agreement(got[n1], got[n2], n1, n2)
            agree.append(ag)
            print(f"  {ag['pair']:16s} prompt length {ag['same_prompt_length']}, "
                  f"generated ids {ag['same_generated_ids']}, text {ag['same_answer_text']}"
                  + (f", differ on {','.join(ag['differing_ids'])}" if ag["differing_ids"] else ""))

    json.dump({"ways": ways, "rows": rowsets, "disagreements": disagree, "agreement": agree},
              open(a.out, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
