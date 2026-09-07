"""Run the quality battery through bitnet.cpp's own kernels on the same KV260, greedy.

usage (on the board):
  python3 quality_bitnetcpp.py quality_prompts.json bitnetcpp-answers.json

One llama-cli a prompt, with exactly the flags kria.txt measured (--temp 0, 4 threads, the same
chat-template file, the same system message), and -n set to the battery's max_new for that prompt so
all three ways are given the same budget. The model stays in the page cache between prompts, so only
the first invocation pays a cold load.

llama-cli prints its answer into a conversation transcript on stdout; the answer is taken as
everything between the echoed question and the '[ Prompt: ... ]' status line, with the leading
spinner run removed. The raw stdout and stderr of every invocation are kept in the output file, so
nothing depends on the parser being right.
"""
import argparse
import json
import re
import subprocess
import sys
import time

ANSI = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")
STATUS = re.compile(r"\[\s*Prompt:.*?\]", re.S)
DECODED = re.compile(r"slot decode token, id=(\d+)")


def apply_backspaces(s):
    """llama-cli draws its waiting spinner as character-then-backspace pairs, so the transcript only
    reads as text once the backspaces are applied. Anything else would either leave '|-\\|/' glued to
    the front of every answer or strip characters the model actually produced."""
    out = []
    for ch in s:
        if ch == "\b":
            if out:
                out.pop()
        elif ch == "\r":
            while out and out[-1] != "\n":
                out.pop()
        else:
            out.append(ch)
    return "".join(out)


TRUNCATED = re.compile(r"\.\.\.\s*\(truncated\)")


def extract(stdout, question):
    """The assistant's turn out of llama-cli's conversation transcript.

    llama-cli echoes the question after '> ' and then answers, but it CUTS a long echo short and
    writes '... (truncated)' in its place, so looking for the question itself finds nothing on a long
    prompt and the whole banner would be taken for the answer. The anchors are tried in order: the
    question verbatim, then the truncation marker, then the last '> ' line."""
    s = apply_backspaces(ANSI.sub("", stdout))
    i = s.rfind(question)
    if i >= 0:
        body = s[i + len(question):]
    else:
        m = list(TRUNCATED.finditer(s))
        if m:
            body = s[m[-1].end():]
        else:
            j = s.rfind("\n> ")
            k = s.find("\n", j + 3) if j >= 0 else -1
            body = s[k:] if k >= 0 else s
    m = STATUS.search(body)
    if m:
        body = body[:m.start()]
    return body.strip()


def timings(text):
    """llama's own load/prompt-eval/eval numbers; this build prefixes every line with a timestamp,
    so the lines are found by substring, not by anchoring at the start."""
    out = {}
    num = r"=\s*([0-9.]+)\s*ms\s*/\s*(\d+)\s*(?:tokens|runs)"
    for ln in text.split("\n"):
        if "prompt eval time" in ln:
            m = re.search(num, ln)
            if m:
                out["prompt_ms"], out["prompt_tokens"] = float(m.group(1)), int(m.group(2))
        elif "eval time" in ln:
            m = re.search(num, ln)
            if m:
                out["gen_ms"], out["gen_tokens"] = float(m.group(1)), int(m.group(2))
        elif "load time" in ln:
            m = re.search(r"=\s*([0-9.]+)\s*ms", ln)
            if m:
                out["load_ms"] = float(m.group(1))
    m = re.search(r"Prompt:\s*([0-9.]+)\s*t/s\s*\|\s*Generation:\s*([0-9.]+)\s*t/s", text)
    if m:
        out["status_prompt_tps"], out["status_gen_tps"] = float(m.group(1)), float(m.group(2))
    for k, tk in (("prompt", "prompt"), ("gen", "gen")):
        if out.get(tk + "_ms") and out.get(tk + "_tokens"):
            out[k + "_tps"] = round(out[tk + "_tokens"] / (out[tk + "_ms"] / 1e3), 3)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("prompts")
    ap.add_argument("out")
    ap.add_argument("--bin", default="/home/ubuntu/bitnet/BitNet/build/bin/llama-cli")
    ap.add_argument("--model", default="/home/ubuntu/bitnet/ggml-model-i2_s.gguf")
    ap.add_argument("--template", default="/home/ubuntu/bitnet/bitnet_chat.jinja")
    ap.add_argument("--cwd", default="/home/ubuntu/bitnet/BitNet")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--context", type=int, default=2048)
    ap.add_argument("--only", default=None, help="comma-separated prompt ids")
    ap.add_argument("--reextract", action="store_true",
                    help="do not run llama-cli: re-parse the answers already in <out>, which keeps "
                         "every invocation's raw stdout, and write them back")
    a = ap.parse_args()

    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    spec = json.load(open(a.prompts, encoding="utf-8"))
    if a.reextract:
        doc = json.load(open(a.out, encoding="utf-8"))
        changed = 0
        for r in doc["runs"]:
            new = extract(r["stdout"], r["prompt"])
            if new != r["answer"]:
                changed += 1
                print(f"=== {r['id']} rep {r['rep']} re-extracted "
                      f"({len(r['answer'])} -> {len(new)} chars)")
                print(new)
                r["answer"] = new
        json.dump(doc, open(a.out, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
        print(f"{changed} of {len(doc['runs'])} answers changed in {a.out}")
        return 0
    system = spec.get("system")
    prompts = spec["prompts"]
    if a.only:
        want = set(a.only.split(","))
        prompts = [p for p in prompts if p["id"] in want]

    runs = []
    for p in prompts:
        for rep in range(int(p.get("repeats", 1))):
            cmd = [a.bin, "-m", a.model, "-t", str(a.threads), "-n", str(int(p["max_new"])),
                   "-c", str(a.context), "--temp", "0", "-ngl", "0",
                   "--override-kv", "tokenizer.ggml.pre=str:llama-bpe",
                   "--jinja", "--chat-template-file", a.template, "-cnv", "-st", "-v"]
            if system:
                cmd += ["-sys", system]
            cmd += ["-p", p["prompt"]]
            t0 = time.time()
            r = subprocess.run(cmd, cwd=a.cwd, capture_output=True, timeout=1800)
            wall = time.time() - t0
            so = r.stdout.decode("utf-8", "replace")
            se = r.stderr.decode("utf-8", "replace")
            ans = extract(so, p["prompt"])
            gen_ids = [int(x) for x in DECODED.findall(se)]
            keep = "\n".join(ln for ln in se.split("\n") if "time =" in ln or "release:" in ln)
            rec = {"id": p["id"], "category": p["category"], "rep": rep,
                   "max_new": int(p["max_new"]), "prompt": p["prompt"], "system": system,
                   "answer": ans, "gen_ids": gen_ids, "wall_s": round(wall, 3),
                   "t0": t0, "t1": t0 + wall,
                   "returncode": r.returncode, "cmd": cmd, "stdout": so,
                   "stderr_timing": keep, "stderr_tail": se[-2000:]}
            rec.update(timings(se + "\n" + so))
            rec["hit_cap"] = rec.get("gen_tokens", 0) >= int(p["max_new"])
            runs.append(rec)
            print(f"=== {p['id']} rep {rep} [{p['category']}] -n {p['max_new']} rc {r.returncode}")
            print(f"> {p['prompt']}")
            print(ans)
            print(f"wall {wall:.2f} s, prompt {rec.get('prompt_tokens')} tokens "
                  f"{rec.get('prompt_ms')} ms, generated {rec.get('gen_tokens')} "
                  f"{rec.get('gen_ms')} ms, hit_cap {rec['hit_cap']}")
            sys.stdout.flush()

    json.dump({"way": "bitnet.cpp", "system": system, "threads": a.threads,
               "context": a.context, "runs": runs},
              open(a.out, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    det = [x for x in runs if x["id"] == "q24_determinism"]
    if len(det) >= 2:
        print(f"\ndeterminism q24: identical text {det[0]['answer'] == det[1]['answer']}")
    print(f"{len(runs)} runs written to {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
