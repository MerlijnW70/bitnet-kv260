"""An OpenAI-compatible HTTP endpoint for bitnet_kria on the KV260.

It holds one bitnet_kria open so the 521 MB of weights are read into udmabuf0 once, and it groups
requests that arrive close together into a single --gen-batch run, so the whole model's weight
stream is read once for the group instead of once a request. That is where the throughput comes
from: one sequence generates at 17.2 tok/s, two at 22.7 and four at 24.0 aggregate, and a token
falls from 0.394 J to 0.286 (two) or 0.248 (four). Three sequences is always worse than two, so
--gen-batch 3 is refused; docs/measurements.md has the table and the reason.

usage (on the board, as the normal user: the server starts bitnet_kria under sudo itself, because
bitnet_kria needs /dev/mem while the tokenizers module is installed for the user):
  python3 bitnet_serve.py
  python3 bitnet_serve.py --port 8080 --gen-batch 2 --context 1024

  curl http://localhost:8080/v1/chat/completions -H 'content-type: application/json' \
    -d '{"model":"bitnet-2b4t","messages":[{"role":"user","content":"144 / 12 + 7?"}]}'

options: --dir DIR, --exe PATH, --host, --port, --gen-batch 1|2|4, --context N, --max-new N,
         --attn-ports N, --head auto|fabric|arm, --threads N, --group-ms MS, --quiet

routes: POST /v1/chat/completions (stream true or false), POST /v1/completions, GET /v1/models,
        GET /healthz.

A sequence's key/value cache is layers x context x 1312 bytes and udmabuf0 has about 107 MB free
once the weights are in it, so the context a sequence may have falls as the batch grows: 2718
positions at one sequence, 1359 at two, 679 at four. The defaults here are two sequences of 1024,
which leaves room for a conversation of a few turns; bitnet_kria refuses rather than overrun.

The board samples greedily, so a request's temperature and top_p are accepted and ignored; the
runtime's own --temp is what decides. max_tokens is honoured: the group's line opens with !N, the
tokens the run may generate, so a request that wants sixty-four does not pay for the two hundred
the answer would have run to. A request shorter than the group's longest still rides to the end of
the group, so a group is as long as its longest member asks for.
"""
import argparse
import json
import os
import queue
import subprocess
import sys
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

EOS = (128001, 128009)
DEFAULT_DIR = "/home/ubuntu/bitnet-kria"
MODEL = "bitnet-2b4t"


class Tok:
    def __init__(self, path):
        from tokenizers import Tokenizer
        self.tk = Tokenizer.from_file(path)

    def encode(self, text):
        return self.tk.encode(text, add_special_tokens=False).ids

    def decode(self, ids):
        return self.tk.decode([int(v) for v in ids], skip_special_tokens=True)


def chat_text(messages):
    """The checkpoint's own template: Role: content<|eot_id|>, then Assistant: to answer."""
    out = []
    for m in messages:
        role, content = m.get("role", "user"), m.get("content", "")
        if isinstance(content, list):
            content = "".join(p.get("text", "") for p in content if isinstance(p, dict))
        if role == "system":
            out.append(f"System: {content}<|eot_id|>")
        elif role == "assistant":
            out.append(f"Assistant: {content}<|eot_id|>")
        else:
            out.append(f"User: {content}<|eot_id|>")
    return "".join(out) + "Assistant: "


class Req:
    def __init__(self, ids, max_new):
        self.ids = ids
        self.max_new = max_new
        self.out = queue.Queue()
        self.made = 0
        self.done = threading.Event()

    def push(self, tok):
        if self.done.is_set():
            return
        if tok in EOS or self.made >= self.max_new:
            self.finish()
            return
        self.made += 1
        self.out.put(tok)
        if self.made >= self.max_new:
            self.finish()

    def finish(self):
        if not self.done.is_set():
            self.done.set()
            self.out.put(None)


class Engine:
    """One bitnet_kria, fed a group at a time; the reader hands each slot's ids to its request."""

    def __init__(self, argv, nseq, quiet):
        self.nseq = nseq
        self.lock = threading.Lock()
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
                print(line.rstrip("\n"), file=sys.stderr)
            if line.startswith("  setup "):
                break

    def alive(self):
        return self.p.poll() is None

    def run(self, group):
        """Send one group of requests and dispatch every token it produces back to its own.

        One line carries one prompt a request, semicolons between, so the run holds exactly as many
        sequences as there are requests: a lone caller is never slowed by a copy of itself riding
        the other slot."""
        with self.lock:
            want = max(r.max_new for r in group)
            line = " ; ".join(" ".join(str(int(v)) for v in r.ids) for r in group)
            self.p.stdin.write(f"!{want} " + line + "\n")
            self.p.stdin.flush()
            stats = ""
            while True:
                raw = self.p.stdout.readline()
                if not raw:
                    break
                s = raw.rstrip("\n")
                if s.startswith("prompt "):
                    stats = s
                    break
                if s.startswith("done ") or s.startswith("sequences "):
                    continue
                slot, tok = 0, None
                if ":" in s:
                    a, b = s.split(":", 1)
                    try:
                        slot, tok = int(a), int(b)
                    except ValueError:
                        tok = None
                else:
                    try:
                        tok = int(s)
                    except ValueError:
                        tok = None
                if tok is None:
                    print(s, file=sys.stderr)
                    continue
                if 0 <= slot < len(group):
                    group[slot].push(tok)
            for r in group:
                r.finish()
            return stats

    def close(self):
        try:
            self.p.stdin.close()
            self.p.wait(timeout=10)
        except Exception:
            self.p.kill()


class Scheduler(threading.Thread):
    """Take a request, wait group_ms for company, run up to nseq of them as one batch."""

    def __init__(self, engine, group_ms, quiet):
        super().__init__(daemon=True)
        self.engine = engine
        self.group_ms = group_ms / 1e3
        self.quiet = quiet
        self.pending = queue.Queue()
        self.stats = ""

    def submit(self, req):
        self.pending.put(req)

    def run(self):
        while True:
            first = self.pending.get()
            if first is None:
                return
            group = [first]
            if self.engine.nseq > 1:
                deadline = time.time() + self.group_ms
                while len(group) < self.engine.nseq:
                    left = deadline - time.time()
                    if left <= 0:
                        break
                    try:
                        group.append(self.pending.get(timeout=left))
                    except queue.Empty:
                        break
            try:
                began = time.time()
                self.stats = self.engine.run(group)
                if not self.quiet:
                    made = sum(r.made for r in group)
                    took = time.time() - began
                    print(f"bitnet_serve: {len(group)} in the group, {made} tokens in {took:.2f} s "
                          f"({made / took:.2f} tok/s aggregate) | {self.stats}", file=sys.stderr)
            except Exception as exc:
                for r in group:
                    r.finish()
                print(f"bitnet_serve: the group failed: {exc}", file=sys.stderr)
                if not self.engine.alive():
                    print("bitnet_serve: bitnet_kria has gone; every request from here is a 503",
                          file=sys.stderr)


class Handler(BaseHTTPRequestHandler):
    server_version = "bitnet-kv260"
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):
        if not self.server.quiet:
            sys.stderr.write("%s %s\n" % (self.address_string(), fmt % args))

    def _json(self, code, body):
        raw = json.dumps(body).encode("utf-8")
        self.send_response(code)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def _error(self, code, message):
        self._json(code, {"error": {"message": message, "type": "invalid_request_error"}})

    def do_GET(self):
        if self.path.rstrip("/") == "/healthz":
            self._json(200, {"status": "ok", "model": MODEL, "sequences": self.server.nseq,
                             "context": self.server.context})
        elif self.path.rstrip("/") == "/v1/models":
            self._json(200, {"object": "list",
                             "data": [{"id": MODEL, "object": "model", "owned_by": "kv260"}]})
        else:
            self._error(404, f"no route {self.path}")

    def do_POST(self):
        route = self.path.rstrip("/")
        if route not in ("/v1/chat/completions", "/v1/completions"):
            self._error(404, f"no route {self.path}")
            return
        try:
            n = int(self.headers.get("content-length") or 0)
            body = json.loads(self.rfile.read(n) or b"{}")
        except (ValueError, json.JSONDecodeError) as exc:
            self._error(400, f"the body is not json: {exc}")
            return

        srv = self.server
        if route == "/v1/chat/completions":
            messages = body.get("messages")
            if not isinstance(messages, list) or not messages:
                self._error(400, "messages must be a non-empty list")
                return
            text = chat_text(messages)
        else:
            prompt = body.get("prompt")
            if not isinstance(prompt, str) or not prompt:
                self._error(400, "prompt must be a non-empty string")
                return
            text = prompt

        want = body.get("max_tokens") or body.get("max_completion_tokens") or srv.max_new
        try:
            want = max(1, min(int(want), srv.max_new))
        except (TypeError, ValueError):
            self._error(400, "max_tokens must be a number")
            return

        ids = srv.tok.encode(text)
        if len(ids) + want > srv.context:
            self._error(400, f"{len(ids)} prompt tokens and {want} to generate pass the "
                             f"{srv.context}-token context this server was started with")
            return

        if not srv.sched.engine.alive():
            self._error(503, "bitnet_kria is not running any more; the server must be restarted")
            return

        req = Req(ids, want)
        srv.sched.submit(req)
        if body.get("stream"):
            self._stream(req, ids, route)
        else:
            self._whole(req, ids, route)

    def _finish_reason(self, req):
        return "length" if req.made >= req.max_new else "stop"

    def _whole(self, req, ids, route):
        got = []
        while True:
            tok = req.out.get()
            if tok is None:
                break
            got.append(tok)
        text = self.server.tok.decode(got)
        made = str(uuid.uuid4())
        created = int(time.time())
        if route == "/v1/chat/completions":
            choice = {"index": 0, "message": {"role": "assistant", "content": text},
                      "finish_reason": self._finish_reason(req)}
            obj = "chat.completion"
        else:
            choice = {"index": 0, "text": text, "finish_reason": self._finish_reason(req)}
            obj = "text_completion"
        self._json(200, {"id": f"cmpl-{made}", "object": obj, "created": created, "model": MODEL,
                         "choices": [choice],
                         "usage": {"prompt_tokens": len(ids), "completion_tokens": len(got),
                                   "total_tokens": len(ids) + len(got)}})

    def _stream(self, req, ids, route):
        self.send_response(200)
        self.send_header("content-type", "text/event-stream")
        self.send_header("cache-control", "no-cache")
        self.send_header("connection", "close")
        self.end_headers()
        made = f"cmpl-{uuid.uuid4()}"
        created = int(time.time())
        obj = "chat.completion.chunk" if route == "/v1/chat/completions" else "text_completion"

        def send(choice):
            frame = {"id": made, "object": obj, "created": created, "model": MODEL,
                     "choices": [choice]}
            self.wfile.write(b"data: " + json.dumps(frame).encode("utf-8") + b"\n\n")
            self.wfile.flush()

        if route == "/v1/chat/completions":
            send({"index": 0, "delta": {"role": "assistant"}, "finish_reason": None})
        acc, shown = [], ""
        try:
            while True:
                tok = req.out.get()
                if tok is None:
                    break
                acc.append(tok)
                whole = self.server.tok.decode(acc)
                if not whole.startswith(shown):
                    shown = ""
                piece, shown = whole[len(shown):], whole
                if not piece:
                    continue
                if route == "/v1/chat/completions":
                    send({"index": 0, "delta": {"content": piece}, "finish_reason": None})
                else:
                    send({"index": 0, "text": piece, "finish_reason": None})
            if route == "/v1/chat/completions":
                send({"index": 0, "delta": {}, "finish_reason": self._finish_reason(req)})
            else:
                send({"index": 0, "text": "", "finish_reason": self._finish_reason(req)})
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            req.finish()


def pick_head(dirname):
    need = ("head3_t.bin", "head_t_scale.bin")
    have = all(os.path.exists(os.path.join(dirname, f)) for f in need)
    return "fabric" if have and os.path.exists("/dev/udmabuf2") else "arm"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", default=DEFAULT_DIR)
    ap.add_argument("--exe", default=None)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--gen-batch", type=int, default=2, choices=(1, 2, 4))
    ap.add_argument("--context", type=int, default=1024)
    ap.add_argument("--max-new", type=int, default=512)
    ap.add_argument("--attn-ports", type=int, default=2)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--head", default="auto", choices=("auto", "fabric", "arm"))
    ap.add_argument("--group-ms", type=float, default=25.0)
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--no-sudo", action="store_true")
    a = ap.parse_args(argv)

    room = 107_000_000 // (30 * 1312)
    if a.gen_batch * a.context > room:
        raise SystemExit(f"{a.gen_batch} sequences of {a.context} positions do not fit udmabuf0: "
                         f"the room is about {room} positions in all, so {room // a.gen_batch} "
                         f"a sequence at this batch")

    try:
        tok = Tok(os.path.join(a.dir, "tokenizer.json"))
    except ImportError:
        raise SystemExit("no tokenizers module: pip3 install --user tokenizers")

    exe = a.exe or os.path.join(a.dir, "bitnet_kria")
    head = a.head if a.head != "auto" else pick_head(a.dir)
    cmd = [exe, "--dir", a.dir, "--max-new", str(a.max_new), "--context", str(a.context),
           "--threads", str(a.threads), "--cache-dtype", "fab", "--head", head,
           "--attn-ports", str(a.attn_ports), "--gen-batch", str(a.gen_batch)]
    if os.geteuid() != 0 and not a.no_sudo:
        cmd = (["sudo", "-A"] if os.environ.get("SUDO_ASKPASS") else ["sudo"]) + cmd
        print("not root: running bitnet_kria under sudo (it needs /dev/mem). Run this as the normal "
              "user, which is where the tokenizers module is installed; with no terminal to ask on, "
              "either run 'sudo -v' first or set SUDO_ASKPASS.", file=sys.stderr)

    engine = Engine(cmd, a.gen_batch, a.quiet)
    sched = Scheduler(engine, a.group_ms, a.quiet)
    sched.start()

    httpd = ThreadingHTTPServer((a.host, a.port), Handler)
    httpd.daemon_threads = True
    httpd.tok = tok
    httpd.sched = sched
    httpd.nseq = a.gen_batch
    httpd.context = a.context
    httpd.max_new = a.max_new
    httpd.quiet = a.quiet
    print(f"bitnet_serve: http://{a.host}:{a.port}/v1/chat/completions, {a.gen_batch} sequence"
          f"{'' if a.gen_batch == 1 else 's'} of {a.context} positions, head {head}", flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        engine.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
