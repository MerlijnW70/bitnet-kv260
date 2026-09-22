"""Plain completions from a base BitNet checkpoint on the KV260: no chat template, no roles.

bitnet_chat.py is for 2B4T, which is instruction tuned and has a chat template. A base checkpoint
such as 1bitLLM/bitnet_b1_58-large has neither, so the honest way to drive it is to give it text and
let it continue. This encodes the prompt with the checkpoint's own tokenizer.json, feeds the ids to
bitnet_kria on stdin, and prints what comes back.

usage (on the board, as root because bitnet_kria opens /dev/mem, and with the tokenizer on the path
because it is installed for the ordinary user rather than for root):

  SITE=$(python3 -c 'import site; print(site.getusersitepackages())')
  sudo -A env PYTHONPATH="$SITE" ATTN_SHAPE=16x96 python3 run073.py --dir ~/bitnet073 \\
       --cache-dtype fab --attn-ports 4 "The capital of France is"

ATTN_SHAPE=16x96 and --cache-dtype fab ask for the fabric attention block, which is worth having
from a few hundred positions of context upward; below that --cache-dtype i8 is faster. Leave both
off and attention runs on the A53s.
"""
import argparse
import os
import subprocess
import sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("prompt", nargs="+")
    ap.add_argument("--dir", default="/home/ubuntu/bitnet073")
    ap.add_argument("--exe", default="./bitnet_kria")
    ap.add_argument("--max-new", type=int, default=64)
    ap.add_argument("--context", type=int, default=1024)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--head", default="fabric")
    ap.add_argument("--cache-dtype", default="i8")
    ap.add_argument("--attn-ports", type=int, default=0)
    ap.add_argument("--temp", type=float, default=0.0)
    a = ap.parse_args()

    try:
        from tokenizers import Tokenizer
    except ImportError:
        sys.exit("no tokenizers module: PYTHONPATH must carry the ordinary user's site-packages")
    path = os.path.join(a.dir, "tokenizer.json")
    if not os.path.exists(path):
        sys.exit("no %s" % path)
    tok = Tokenizer.from_file(path)

    given = [[1] + tok.encode(p, add_special_tokens=False).ids for p in a.prompt]
    lines = [" ".join(str(i) for i in ids) for ids in given]
    cmd = [a.exe, "--dir", a.dir, "--context", str(a.context), "--threads", str(a.threads),
           "--head", a.head, "--quiet", "--cache-dtype", a.cache_dtype,
           "--max-new", str(a.max_new), "--temp", str(a.temp)]
    if a.attn_ports:
        cmd += ["--attn-ports", str(a.attn_ports)]
    out = subprocess.run(cmd, input="\n".join(lines) + "\n", capture_output=True, text=True)

    made, cur = [], []
    for line in out.stdout.splitlines():
        s = line.strip()
        if s.lstrip("-").isdigit():
            cur.append(int(s))
        elif s.startswith("done") and cur:
            made.append(cur)
            cur = []
    if cur:
        made.append(cur)
    if not made:
        sys.stderr.write(out.stdout[-800:] + out.stderr[-800:])
        sys.exit("bitnet_kria returned no ids")
    for ids, got in zip(given, made):
        print(tok.decode(ids[1:] + got))
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
