"""Exercise a running bitnet_serve endpoint: routes, shape, streaming, limits, errors, concurrency.

usage: python3 tools/api_check.py [http://host:8080]   (default http://kria:8080)

It needs the endpoint up (runtime/bitnet_serve.py on the board) and nothing else; every check is
plain urllib. exit 0 all passed, 1 something failed. The last three time one request alone against
two at once, which is what the grouping is for: on the reference board two at once run about 1.4
times the aggregate of one, measured from the client, where the server's own figure is 1.27 because
a client also pays its connection and transfer once a request.
"""
import json
import sys
import threading
import time
import urllib.error
import urllib.request

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://kria:8080"
PASS, FAIL = [], []


def check(name, ok, detail=""):
    (PASS if ok else FAIL).append(name)
    print(("  ok   " if ok else "  FAIL ") + name + (("  " + detail) if detail else ""))


def call(path, body=None, stream=False, timeout=180):
    url = BASE + path
    if body is None:
        req = urllib.request.Request(url)
    else:
        req = urllib.request.Request(url, data=json.dumps(body).encode(),
                                     headers={"content-type": "application/json"})
    try:
        r = urllib.request.urlopen(req, timeout=timeout)
        raw = r.read().decode()
        return r.status, (raw if stream else json.loads(raw))
    except urllib.error.HTTPError as e:
        raw = e.read().decode()
        try:
            return e.code, json.loads(raw)
        except json.JSONDecodeError:
            return e.code, raw


print("== routes")
s, b = call("/healthz")
check("GET /healthz", s == 200 and b.get("status") == "ok", str(b))
s, b = call("/v1/models")
check("GET /v1/models", s == 200 and b["data"][0]["id"] == "bitnet-2b4t")
s, b = call("/nope")
check("unknown route is 404", s == 404)

print("== chat completions")
t0 = time.time()
s, b = call("/v1/chat/completions",
            {"messages": [{"role": "user", "content": "What is 144 divided by 12, plus 7?"}],
             "max_tokens": 48})
dt = time.time() - t0
ok = s == 200 and b.get("object") == "chat.completion"
text = b["choices"][0]["message"]["content"] if ok else ""
check("shape", ok and b["choices"][0]["message"]["role"] == "assistant")
check("answers 19", "19" in text, repr(text[:70]))
u = b.get("usage", {})
check("usage adds up", u.get("prompt_tokens", 0) + u.get("completion_tokens", 0) == u.get("total_tokens"),
      str(u))
check("finish_reason is stop", b["choices"][0].get("finish_reason") == "stop",
      "%.1f s" % dt)

print("== max_tokens")
s, b = call("/v1/chat/completions",
            {"messages": [{"role": "user", "content": "Count upward from one, forever."}],
             "max_tokens": 8})
n = b["usage"]["completion_tokens"]
check("asked 8, got at most 8", n <= 8, "got %d" % n)
check("finish_reason is length", b["choices"][0].get("finish_reason") == "length")

print("== system message and multi-turn")
s, b = call("/v1/chat/completions",
            {"messages": [{"role": "system", "content": "Answer with a single word."},
                          {"role": "user", "content": "What colour is grass?"}],
             "max_tokens": 16})
check("system + user accepted", s == 200 and len(b["choices"][0]["message"]["content"]) > 0,
      repr(b["choices"][0]["message"]["content"][:50]))

print("== text completions")
s, b = call("/v1/completions", {"prompt": "The capital of France is", "max_tokens": 12})
check("POST /v1/completions", s == 200 and b.get("object") == "text_completion",
      repr(b["choices"][0]["text"][:50]) if s == 200 else str(b))

print("== streaming")
s, raw = call("/v1/chat/completions",
              {"messages": [{"role": "user", "content": "Name one prime number above 50."}],
               "max_tokens": 24, "stream": True}, stream=True)
frames = [l[6:] for l in raw.splitlines() if l.startswith("data: ")]
check("stream ends with [DONE]", frames and frames[-1] == "[DONE]")
objs = [json.loads(f) for f in frames if f != "[DONE]"]
check("first frame carries the role", objs and objs[0]["choices"][0]["delta"].get("role") == "assistant")
joined = "".join(o["choices"][0]["delta"].get("content", "") for o in objs)
check("stream has text", len(joined) > 0, repr(joined[:60]))
check("last frame has a finish_reason", objs[-1]["choices"][0].get("finish_reason") is not None)

print("== errors")
req = urllib.request.Request(BASE + "/v1/chat/completions", data=b"not json",
                             headers={"content-type": "application/json"})
try:
    urllib.request.urlopen(req, timeout=30)
    check("bad json is 400", False)
except urllib.error.HTTPError as e:
    check("bad json is 400", e.code == 400)
s, b = call("/v1/chat/completions", {"messages": []})
check("empty messages is 400", s == 400)
s, b = call("/v1/chat/completions",
            {"messages": [{"role": "user", "content": "word " * 2000}], "max_tokens": 16})
check("over-long prompt is 400", s == 400, str(b.get("error", {}).get("message", ""))[:80])

print("== one alone against two at once")
PARA = {"messages": [{"role": "user", "content":
                      "Write a detailed paragraph about field programmable gate arrays."}],
        "max_tokens": 64}


def rate(n):
    out = {}

    def one(i):
        st, bb = call("/v1/chat/completions", PARA)
        out[i] = (st, bb["usage"]["completion_tokens"] if st == 200 else 0)

    ts = [threading.Thread(target=one, args=(i,)) for i in range(n)]
    t0 = time.time()
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    dt = time.time() - t0
    tot = sum(v[1] for v in out.values())
    return all(v[0] == 200 for v in out.values()), tot, dt


ok1, tot1, dt1 = rate(1)
ok2, tot2, dt2 = rate(2)
check("one alone answers", ok1, "%d tokens in %.2f s = %.2f tok/s" % (tot1, dt1, tot1 / dt1))
check("two at once both answer", ok2, "%d tokens in %.2f s = %.2f tok/s" % (tot2, dt2, tot2 / dt2))
check("two at once beat one alone", (tot2 / dt2) > (tot1 / dt1) * 1.15,
      "%.2fx" % ((tot2 / dt2) / (tot1 / dt1)))

print("\n%d passed, %d failed" % (len(PASS), len(FAIL)))
if FAIL:
    print("failed: " + ", ".join(FAIL))
sys.exit(1 if FAIL else 0)
