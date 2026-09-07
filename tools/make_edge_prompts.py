"""Turn edge_alarms.json into the prompt file the two baseline harnesses already read.

The point is that all three ways are given the SAME bytes. The system message is lifted out of
runtime/edge_monitor.py by parsing its source, so it cannot drift from what the board actually sent,
and the user turn is built the way do_triage builds it ("Alarm: " + line). max_new is
edge_monitor.py's own default.

usage:
  python make_edge_prompts.py runtime/edge_monitor.py edge_alarms.json edge_prompts.json
"""
import argparse
import ast
import json
import sys


def constant(path, name):
    """The value of a module-level string assignment, without importing the module."""
    tree = ast.parse(open(path, encoding="utf-8").read())
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id == name for t in node.targets):
            return ast.literal_eval(node.value)
    raise KeyError(name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("monitor")
    ap.add_argument("alarms", help="edge_alarms.json, or an edge_monitor run's JSON with --selftest")
    ap.add_argument("out")
    ap.add_argument("--max-new", type=int, default=96)
    ap.add_argument("--selftest", action="store_true",
                    help="instead of the alarms, take the self-report turns verbatim out of a run's "
                         "JSON, so the baseline is asked about the same real sensor readings the "
                         "board was asked about")
    a = ap.parse_args()

    if a.selftest:
        system = constant(a.monitor, "SYSTEM_SELFTEST")
        runs = json.load(open(a.alarms, encoding="utf-8"))["runs"]
        prompts = [{"id": r["id"], "category": "selftest", "prompt": r["prompt"],
                    "readings": r["readings"], "max_new": a.max_new, "repeats": 1,
                    "board_answer": r["answer"], "board_sentences": r["sentences"],
                    "board_words": r["words"], "board_verdict": r["verdict"]}
                   for r in runs if r.get("job") == "selftest"]
        what = "the board's own self-report turns, as the GPU takes them"
    else:
        system = constant(a.monitor, "SYSTEM_TRIAGE")
        spec = json.load(open(a.alarms, encoding="utf-8"))
        prompts = [{"id": x["id"], "category": x["category"], "severity": x["severity"],
                    "ambiguous": bool(x.get("ambiguous")), "line": x["line"],
                    "prompt": "Alarm: " + x["line"], "max_new": a.max_new, "repeats": 1}
                   for x in spec["alarms"]]
        what = "the edge triage job, as the two baselines take it"
    json.dump({"what": what, "from": [a.monitor, a.alarms], "system": system, "prompts": prompts},
              open(a.out, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    print(f"{len(prompts)} prompts, system {len(system)} chars, written to {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
