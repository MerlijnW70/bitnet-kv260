"""Type a sentence, read the answer: the chat front end for bitnet_kria on the KV260.

It holds one bitnet_kria process open (so the 521 MB of weights are read into udmabuf0 once, not once
a turn), builds the checkpoint's own chat prompt, feeds the token ids in on stdin, and prints the text
as the ids come back one a line. The conversation is resent whole every turn, because the runtime
starts a fresh KV cache for each line of ids it reads.

usage (on the board, as root because bitnet_kria opens /dev/mem):
  sudo python3 bitnet_chat.py                          interactive
  sudo python3 bitnet_chat.py --prompt "..."           one shot
  sudo python3 bitnet_chat.py --check tokencheck.json  the tokenizer against ids from the PC

options: --dir DIR (default /home/ubuntu/bitnet-kria), --exe PATH, --max-new N, --context N,
         --threads N, --cache-dtype f32|bf16|i8|fx|fab, --head auto|fabric|arm, --temp T, --top-p P, --seed N,
         --timing, --system TEXT, --no-stream, --quiet
in the interactive loop: /reset starts a new conversation, /quit leaves.

--head picks which output head bitnet_kria runs.  The default, auto, asks for the fabric head when
head3_t.bin, head_t_scale.bin and /dev/udmabuf2 are all present and falls back to the ARM head when
they are not; --head fabric or --head arm force it.  This matters: on the reference board the fabric
head costs 5.40 ms a call against the ARM head's ~52 ms, which is the difference between 16.8 and
about 8 generated tokens a second.  (bitnet_kria's own default is arm; this front end chooses.)

The chat format is the checkpoint's own template (bitnet_chat.jinja, beside this file):
each message becomes "Role: content<|eot_id|>" and the turn ends with "Assistant: ", with no BOS.
"""
import argparse
import json
import os
import subprocess
import sys
import time

EOS = (128001, 128009)
DEFAULT_DIR = "/home/ubuntu/bitnet-kria"


def pick_head(dirname):
    """fabric when the ternary head is packed and its buffer exists, else arm."""
    need = ("head3_t.bin", "head_t_scale.bin")
    have = all(os.path.exists(os.path.join(dirname, f)) for f in need)
    if have and os.path.exists("/dev/udmabuf2"):
        return "fabric"
    why = "no /dev/udmabuf2" if have else "no " + ", ".join(
        f for f in need if not os.path.exists(os.path.join(dirname, f)))
    print(f"[--head auto: {why}, so the ARM head is used; a generated token costs about 52 ms more]",
          file=sys.stderr)
    return "arm"


def chat_text(messages, add_generation_prompt=True):
    out = []
    for m in messages:
        role, content = m["role"], m["content"]
        if role == "system":
            out.append(f"System: {content}<|eot_id|>")
        elif role == "user":
            out.append(f"User: {content}<|eot_id|>")
        elif role == "assistant":
            out.append(f"Assistant: {content}<|eot_id|>")
    text = "".join(out)
    if add_generation_prompt:
        text += "Assistant: "
    return text


class Tok:
    def __init__(self, path):
        from tokenizers import Tokenizer
        self.tk = Tokenizer.from_file(path)

    def encode(self, text):
        return self.tk.encode(text, add_special_tokens=False).ids

    def decode(self, ids):
        return self.tk.decode([int(v) for v in ids], skip_special_tokens=True)


class Engine:
    """One long-lived bitnet_kria. Setup (weights into udmabuf0) is paid once."""

    def __init__(self, argv, quiet=False):
        self.timing = "--timing" in argv
        self.p = subprocess.Popen(argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                  stderr=None, text=True, bufsize=1)
        self.banner = []
        while True:
            line = self.p.stdout.readline()
            if not line:
                self.p.wait()
                raise SystemExit(f"bitnet_kria stopped before it was ready (exit {self.p.returncode}); "
                                 f"it needs /dev/mem, so it must run as root")
            self.banner.append(line.rstrip("\n"))
            if not quiet:
                print(line.rstrip("\n"))
            if line.startswith("  setup "):
                break

    def generate(self, ids, on_token=None):
        """Send the prompt ids, return (generated ids, the runtime's own stats line)."""
        self.p.stdin.write(" ".join(str(int(v)) for v in ids) + "\n")
        self.p.stdin.flush()
        gen, stats, per_forward = [], "", ""
        done = False
        while True:
            line = self.p.stdout.readline()
            if not line:
                break
            s = line.rstrip("\n")
            if s.startswith("done "):
                done = True
                continue
            if s.startswith("prompt "):
                stats = s
                if not self.timing:
                    break
                continue
            if s.startswith("per forward"):
                per_forward = s
                break
            if done:
                continue
            try:
                t = int(s)
            except ValueError:
                print(s, file=sys.stderr)
                continue
            gen.append(t)
            if on_token and t not in EOS:
                on_token(t)
        return gen, stats, per_forward

    def close(self):
        try:
            self.p.stdin.close()
            self.p.wait(timeout=10)
        except Exception:
            self.p.kill()


def parse_stats(line):
    """'prompt 25 tokens 1668.2 ms (14.99 tok/s), generated 49 5969.3 ms (8.21 tok/s)' -> numbers."""
    try:
        a, b = line.split(", generated ")
        pa = a.split()
        pb = b.split()
        return {"prompt_tokens": int(pa[1]), "prompt_ms": float(pa[3]), "prompt_tps": float(pa[5].strip("(")),
                "gen_tokens": int(pb[0]), "gen_ms": float(pb[1]), "gen_tps": float(pb[3].strip("("))}
    except Exception:
        return None


def turn(engine, tok, messages, stream=True, quiet=False, context=0, max_new=0):
    text = chat_text(messages)
    ids = tok.encode(text)
    if context and len(ids) + max_new > context:
        print(f"[the conversation is {len(ids)} tokens and up to {max_new} more are asked for, past the "
              f"{context}-token context: the runtime will stop at {context}. /reset, or start again with "
              f"a larger --context]", file=sys.stderr)
    shown, acc = [""], []

    def on_token(t):
        """Decode the whole answer so far and print only what is new: a token can be half of a
        character, and the decoder only settles it when the rest arrives."""
        acc.append(t)
        out = tok.decode(acc)
        if out.startswith(shown[0]):
            sys.stdout.write(out[len(shown[0]):])
        else:
            sys.stdout.write("\n" + out)
        sys.stdout.flush()
        shown[0] = out

    t0 = time.time()
    gen, stats, per_forward = engine.generate(ids, on_token if stream else None)
    wall = time.time() - t0
    answer = tok.decode([t for t in gen if t not in EOS])
    if not stream:
        sys.stdout.write(answer)
    sys.stdout.write("\n")
    sys.stdout.flush()
    if not quiet:
        st = parse_stats(stats)
        if st:
            print(f"[{st['prompt_tokens']} prompt tokens {st['prompt_ms'] / 1e3:.2f} s "
                  f"({st['prompt_tps']:.2f} tok/s), {st['gen_tokens']} generated "
                  f"{st['gen_ms'] / 1e3:.2f} s ({st['gen_tps']:.2f} tok/s), {wall:.2f} s of wall clock]")
        else:
            print(f"[{stats}] {wall:.2f} s")
        if per_forward:
            print(per_forward)
    return answer


def check_tokenizer(tok, path):
    """Every string in the file must encode to the ids the PC's tokenizer produced for it."""
    cases = json.load(open(path, encoding="utf-8"))
    bad = 0
    for c in cases:
        got = tok.encode(c["text"])
        if got != c["ids"]:
            bad += 1
            print(f"  MISMATCH {c['text']!r}\n    here {got}\n    PC   {c['ids']}")
    print(f"tokenizer: {len(cases) - bad} of {len(cases)} strings encode exactly as the PC's tokenizer does")
    return 1 if bad else 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", default=DEFAULT_DIR)
    ap.add_argument("--exe", default=None)
    ap.add_argument("--prompt", default=None)
    ap.add_argument("--system", default=None)
    ap.add_argument("--max-new", type=int, default=256)
    ap.add_argument("--context", type=int, default=2048)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--cache-dtype", default="f32", choices=("f32", "bf16", "i8", "fx", "fab"))
    ap.add_argument("--attn-ports", type=int, default=0)
    ap.add_argument("--temp", type=float, default=0.0)
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--timing", action="store_true")
    ap.add_argument("--no-stream", action="store_true")
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--check", default=None)
    ap.add_argument("--head", default="auto", choices=("auto", "fabric", "arm"),
                    help="which output head bitnet_kria uses; auto picks fabric when the ternary head "
                         "file and /dev/udmabuf2 are both there, else arm")
    ap.add_argument("--no-sudo", action="store_true", help="run bitnet_kria directly, without sudo")
    a = ap.parse_args(argv)

    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    try:
        tok = Tok(os.path.join(a.dir, "tokenizer.json"))
    except ImportError:
        raise SystemExit("no tokenizers module: pip3 install --user tokenizers, and run this script as the\n"
                         "normal user (it starts bitnet_kria under sudo itself)")
    if a.check:
        return check_tokenizer(tok, a.check)

    exe = a.exe or os.path.join(a.dir, "bitnet_kria")
    head = a.head if a.head != "auto" else pick_head(a.dir)
    cmd = [exe, "--dir", a.dir, "--max-new", str(a.max_new), "--context", str(a.context),
           "--threads", str(a.threads), "--cache-dtype", a.cache_dtype, "--head", head]
    if a.attn_ports:
        cmd += ["--attn-ports", str(a.attn_ports)]
    if a.temp > 0:
        cmd += ["--temp", str(a.temp), "--top-p", str(a.top_p), "--seed", str(a.seed)]
    if a.timing:
        cmd += ["--timing"]
    if os.geteuid() != 0 and not a.no_sudo:
        cmd = ["sudo"] + cmd
        print("not root: running bitnet_kria under sudo (it needs /dev/mem). If sudo has no terminal to "
              "ask on, run 'sudo -v' first.", file=sys.stderr)

    engine = Engine(cmd, quiet=a.quiet)
    messages = []
    if a.system:
        messages.append({"role": "system", "content": a.system})
    try:
        if a.prompt is not None:
            messages.append({"role": "user", "content": a.prompt})
            print(f"> {a.prompt}")
            turn(engine, tok, messages, stream=not a.no_stream, quiet=a.quiet,
                 context=a.context, max_new=a.max_new)
            return 0
        print("type a sentence; /reset starts a new conversation, /quit leaves")
        while True:
            try:
                line = input("> ")
            except EOFError:
                break
            s = line.strip()
            if not s:
                continue
            if s in ("/quit", "/exit", "/q"):
                break
            if s in ("/reset", "/new"):
                messages = [m for m in messages if m["role"] == "system"]
                print("(a new conversation)")
                continue
            messages.append({"role": "user", "content": s})
            answer = turn(engine, tok, messages, stream=not a.no_stream, quiet=a.quiet,
                          context=a.context, max_new=a.max_new)
            messages.append({"role": "assistant", "content": answer.strip()})
    finally:
        engine.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
