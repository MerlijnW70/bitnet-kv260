"""An edge device that answers from its own data, with no network in the loop at all.

Two jobs, both of which a cloud service would need a connection for and both of which this KV260
does in its own silicon at a few watts:

  triage    a line of machine telemetry arrives; the board answers with ONE line of strict JSON,
            {"category": ..., "severity": ..., "action": ...}, and nothing else, so a real gateway
            could act on it without a human reading it.
  selftest  the board reads its own live sensors (INA260 power, the AMS die temperatures, the four
            A53 clocks, load, free memory, the fabric's state) and reports on itself in two
            sentences a maintenance engineer would want.

Everything the model computes runs on this board: bitnet_kria holds every matrix product of
microsoft/bitnet-b1.58-2B-4T in the KV260's fabric. ONE bitnet_kria is held open for the whole run,
so the 521 MB of weights are read into udmabuf0 once instead of once an answer (--saving N measures
what that is worth). Decoding is greedy, so every answer here is reproducible.

usage (on the board; bitnet_kria needs /dev/mem, the tokenizer is the ubuntu user's --user install):
  sudo -S env PYTHONPATH=/home/ubuntu/.local/lib/python3.10/site-packages \
    python3 edge_monitor.py --job both --no-sudo --json-out edge.json

  --job triage|selftest|both   which jobs to run (default both)
  --alarms FILE                the alarm set (default edge_alarms.json beside this script)
  --only ID[,ID...]            just these alarm ids
  --limit N                    just the first N alarms
  --repeat N                   run the whole alarm set N times (default 1)
  --selftests N                how many self-reports to take (default 3)
  --selftest-gap S             seconds between self-reports (default 2)
  --saving N                   after the run, redo N alarms with a fresh process each, and report
                               what holding one process open saves
  --live                       loop for ever: one self-report, then the alarms, then round again
  --json-out FILE              every answer, every timing, every power window, written out
  --no-power                   do not sample the INA260
"""
import argparse
import json
import os
import subprocess
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "/home/ubuntu/bitnet-kria")
import bitnet_chat as bc

HERE = os.path.dirname(os.path.abspath(__file__))
INA = "/sys/class/hwmon/hwmon2"
AMS = "/sys/class/hwmon/hwmon0"
FPGA = "/sys/class/fpga_manager/fpga0/state"

CATEGORIES = ("mechanical", "electrical", "thermal", "sensor", "network", "software", "none")
SEVERITIES = ("critical", "warning", "info")

SYSTEM_TRIAGE = (
    "You are an alarm triage unit inside an industrial gateway. You are given one line of equipment "
    "telemetry and you answer with one line of JSON and nothing else.\n"
    "Answer in exactly this form, on a single line:\n"
    '{"category": "<c>", "severity": "<s>", "action": "<a>"}\n'
    "<c> is exactly one of: mechanical, electrical, thermal, sensor, network, software, none\n"
    "<s> is exactly one of: critical, warning, info\n"
    "<a> is one short imperative sentence telling the technician what to do.\n"
    "Write the JSON object and stop. No greeting, no explanation, no code fence, no second line.\n"
    "Alarm: BRG-01 bearing housing 96 C, alarm 90 C, rising 2 C per hour\n"
    '{"category": "thermal", "severity": "critical", "action": "Stop the machine and inspect the '
    'bearing before it seizes."}\n'
    "Alarm: Shift report printed, 1842 units, no faults\n"
    '{"category": "none", "severity": "info", "action": "File the report, no action needed."}'
)

SYSTEM_SELFTEST = (
    "You are the health monitor of an edge computing board. You are given the board's own live "
    "sensor readings. Answer with exactly two sentences of plain English for a maintenance "
    "engineer and then stop.\n"
    "Sentence 1 says what state the board is in. Every reading you quote goes in this one "
    "sentence, joined with commas; do not start a new sentence for more readings.\n"
    "Sentence 2 is the verdict and must begin with either 'Nothing needs attention' or "
    "'Attention needed:'.\n"
    "Stop after sentence 2. No third sentence, no lists, no JSON, no headings, no preamble. The "
    "whole reply is under 45 words."
)

VERDICTS = ("nothing needs attention", "attention needed")


def read_first(path, default=None):
    try:
        with open(path, "rb") as f:
            return f.read().strip().decode()
    except OSError:
        return default


class Power:
    """The INA260 sampled in one thread with the three sysfs files held open.

    A shell loop forking date+cat every period is itself about 0.15 W on these four A53s, which
    would be charged to the thing being measured; this reads three already-open descriptors.
    """

    def __init__(self, period=0.05):
        self.period = period
        self.samples = []
        self.on = False
        self.files = None
        try:
            self.files = [open(f"{INA}/{n}", "rb") for n in
                          ("power1_input", "curr1_input", "in1_input")]
        except OSError:
            return
        self.on = True
        self.stop = threading.Event()
        self.t = threading.Thread(target=self._loop, daemon=True)
        self.t.start()

    def _loop(self):
        while not self.stop.is_set():
            vals = []
            for f in self.files:
                try:
                    f.seek(0)
                    vals.append(int(f.read().strip()))
                except (OSError, ValueError):
                    vals.append(0)
            self.samples.append((time.time(), vals[0] / 1e6, vals[1], vals[2]))
            self.stop.wait(self.period)

    def close(self):
        if self.on:
            self.stop.set()
            self.t.join(timeout=2)

    def window(self, t0, t1):
        """Trapezoid-integrate power over [t0, t1]: joules, mean/min/max watts, sample count."""
        if not self.on or t1 <= t0:
            return None
        s = [(t, w) for (t, w, _c, _v) in self.samples if t0 - self.period <= t <= t1 + self.period]
        s.sort()
        inner = [(t, w) for (t, w) in s if t0 <= t <= t1]
        if len(s) < 2:
            return None
        j = 0.0
        for (ta, wa), (tb, wb) in zip(s, s[1:]):
            lo, hi = max(ta, t0), min(tb, t1)
            if hi <= lo or tb <= ta:
                continue
            wlo = wa + (wb - wa) * (lo - ta) / (tb - ta)
            whi = wa + (wb - wa) * (hi - ta) / (tb - ta)
            j += 0.5 * (wlo + whi) * (hi - lo)
        ws = [w for (_t, w) in inner] or [w for (_t, w) in s]
        return {"joules": round(j, 4), "mean_w": round(j / (t1 - t0), 4),
                "min_w": round(min(ws), 4), "max_w": round(max(ws), 4), "samples": len(inner)}

    def idle(self, seconds):
        """Sit still and report the board's idle draw over that stretch."""
        t0 = time.time()
        time.sleep(seconds)
        return self.window(t0, time.time())


def net_counters():
    """/proc/net/dev, whole: bytes and packets in and out of every interface."""
    out = {}
    try:
        lines = open("/proc/net/dev", encoding="utf-8").read().splitlines()
    except OSError:
        return out
    for line in lines[2:]:
        if ":" not in line:
            continue
        name, rest = line.split(":", 1)
        f = rest.split()
        if len(f) < 16:
            continue
        out[name.strip()] = {"rx_bytes": int(f[0]), "rx_packets": int(f[1]),
                             "tx_bytes": int(f[8]), "tx_packets": int(f[9])}
    return out


def sensors():
    """Everything the board knows about itself, from its own sysfs."""
    s = {}
    p = read_first(f"{INA}/power1_input")
    c = read_first(f"{INA}/curr1_input")
    v = read_first(f"{INA}/in1_input")
    s["power_w"] = round(int(p) / 1e6, 3) if p else None
    s["current_ma"] = int(c) if c else None
    s["voltage_v"] = round(int(v) / 1e3, 3) if v else None
    for i, key in ((1, "temp_ps_lpd_c"), (2, "temp_ps_fpd_c"), (3, "temp_pl_c")):
        t = read_first(f"{AMS}/temp{i}_input")
        s[key] = round(int(t) / 1e3, 1) if t else None
    clocks = []
    for i in range(64):
        f = read_first(f"/sys/devices/system/cpu/cpu{i}/cpufreq/scaling_cur_freq")
        if f is None:
            break
        clocks.append(int(f) // 1000)
    s["cpu_mhz"] = clocks
    la = read_first("/proc/loadavg", "")
    s["loadavg"] = [float(x) for x in la.split()[:3]] if la else None
    mem = {}
    try:
        for line in open("/proc/meminfo", encoding="utf-8"):
            k, _, rest = line.partition(":")
            mem[k] = int(rest.split()[0])
    except OSError:
        pass
    s["mem_total_mb"] = mem.get("MemTotal", 0) // 1024
    s["mem_available_mb"] = mem.get("MemAvailable", 0) // 1024
    s["fpga_state"] = read_first(FPGA)
    up = read_first("/proc/uptime", "")
    s["uptime_s"] = float(up.split()[0]) if up else None
    return s


def sensor_block(s):
    """The readings as the model sees them: short lines, units spelled out, no jargon."""
    lines = [
        f"Board power {s['power_w']:.2f} W ({s['current_ma']} mA at {s['voltage_v']:.2f} V)",
        f"Processor die temperature {s['temp_ps_lpd_c']:.1f} C, "
        f"second processor sensor {s['temp_ps_fpd_c']:.1f} C",
        f"FPGA fabric die temperature {s['temp_pl_c']:.1f} C",
        f"CPU core clocks {' '.join(str(m) for m in s['cpu_mhz'])} MHz on {len(s['cpu_mhz'])} cores",
        f"Load average {s['loadavg'][0]:.2f} {s['loadavg'][1]:.2f} {s['loadavg'][2]:.2f}",
        f"Memory {s['mem_available_mb']} MB free of {s['mem_total_mb']} MB",
        f"FPGA configuration state: {s['fpga_state']}",
        f"Uptime {s['uptime_s'] / 86400:.1f} days",
    ]
    return "\n".join(lines)


def strict_json(text):
    """A parse with no rescuing: the answer, stripped of surrounding whitespace, must itself be the
    JSON object. Anything the model puts around it counts as a failure of the prompt, not of this."""
    try:
        v = json.loads(text.strip())
    except (ValueError, TypeError):
        return None
    return v if isinstance(v, dict) else None


def loose_json(text):
    """Diagnosis only, never the headline number: the first {...} anywhere in the answer."""
    i, j = text.find("{"), text.rfind("}")
    if i < 0 or j <= i:
        return None
    try:
        v = json.loads(text[i:j + 1])
    except ValueError:
        return None
    return v if isinstance(v, dict) else None


def count_sentences(text):
    """Sentence ends, not full stops: '5.87 W' must not count as one."""
    t = text.strip()
    n = sum(t.count(p + " ") + t.count(p + "\n") for p in ".!?")
    if t and t[-1] in ".!?":
        n += 1
    return n


def judge(obj, want_cat, want_sev):
    """What the answer got right, and whether it stayed inside the two vocabularies."""
    cat = obj.get("category") if isinstance(obj, dict) else None
    sev = obj.get("severity") if isinstance(obj, dict) else None
    act = obj.get("action") if isinstance(obj, dict) else None
    cat = cat.strip().lower() if isinstance(cat, str) else None
    sev = sev.strip().lower() if isinstance(sev, str) else None
    return {"got_category": cat, "got_severity": sev, "got_action": act,
            "category_in_vocab": cat in CATEGORIES, "severity_in_vocab": sev in SEVERITIES,
            "category_ok": cat == want_cat, "severity_ok": sev == want_sev,
            "action_ok": isinstance(act, str) and bool(act.strip()),
            "keys_exact": isinstance(obj, dict) and set(obj) == {"category", "severity", "action"}}


def open_engine(a, quiet):
    head = a.head if a.head != "auto" else bc.pick_head(a.dir)
    cmd = [a.exe, "--dir", a.dir, "--max-new", str(a.max_new), "--context", str(a.context),
           "--threads", str(a.threads), "--cache-dtype", a.cache_dtype, "--head", head]
    if os.geteuid() != 0 and not a.no_sudo:
        cmd = (["sudo", "-A"] if os.environ.get("SUDO_ASKPASS") else ["sudo", "-n"]) + cmd
    t0 = time.time()
    eng = bc.Engine(cmd, quiet=quiet)
    return eng, time.time() - t0, cmd


def ask(engine, tok, power, system, user, max_new_note=None):
    """One answer, timed end to end, with the power drawn over exactly that window."""
    ids = tok.encode(bc.chat_text([{"role": "system", "content": system},
                                   {"role": "user", "content": user}]))
    net0 = net_counters()
    t0 = time.time()
    gen, stats, _pf = engine.generate(ids)
    t1 = time.time()
    net1 = net_counters()
    answer = tok.decode([t for t in gen if t not in bc.EOS])
    r = {"prompt": user, "answer": answer, "answer_ms": round((t1 - t0) * 1e3, 1),
         "prompt_ids": len(ids), "gen_ids": len(gen), "gen_token_ids": gen,
         "stats": stats, "t0": t0, "t1": t1,
         "power": power.window(t0, t1),
         "net_delta": {k: {m: net1[k][m] - net0[k][m] for m in net0[k]}
                       for k in net0 if k in net1}}
    r.update(bc.parse_stats(stats) or {})
    if max_new_note:
        r["hit_cap"] = len(gen) >= max_new_note
    return r


def do_triage(engine, tok, power, alarm, rep, quiet=False):
    user = "Alarm: " + alarm["line"]
    r = ask(engine, tok, power, SYSTEM_TRIAGE, user)
    obj = strict_json(r["answer"])
    r["json_strict_ok"] = obj is not None
    loose = obj if obj is not None else loose_json(r["answer"])
    r["json_recovered_ok"] = loose is not None
    r["parsed"] = loose
    r.update({"id": alarm["id"], "rep": rep, "job": "triage",
              "ambiguous": bool(alarm.get("ambiguous")),
              "want_category": alarm["category"], "want_severity": alarm["severity"]})
    r.update(judge(loose or {}, alarm["category"], alarm["severity"]))
    if not quiet:
        flag = "AMBIG" if r["ambiguous"] else "clear"
        print(f"=== {alarm['id']} [{flag}] want {alarm['category']}/{alarm['severity']}")
        print(f"> {user}")
        print(r["answer"])
        pw = r["power"]
        print(f"    json_strict {r['json_strict_ok']}  category {r['got_category']} "
              f"{'OK' if r['category_ok'] else 'no'}  severity {r['got_severity']} "
              f"{'OK' if r['severity_ok'] else 'no'}")
        print(f"    {r['answer_ms']:.0f} ms, {r['prompt_ids']} prompt ids, {r['gen_ids']} gen ids"
              + (f", {pw['joules']:.3f} J at {pw['mean_w']:.2f} W mean" if pw else ""))
        sys.stdout.flush()
    return r


def do_selftest(engine, tok, power, index, quiet=False):
    s = sensors()
    block = sensor_block(s)
    user = "Board sensor readings:\n" + block + "\nReport on the board."
    r = ask(engine, tok, power, SYSTEM_SELFTEST, user)
    text = r["answer"].strip()
    low = text.lower()
    verdict = next((v for v in VERDICTS if v in low), None)
    r.update({"id": f"self{index:02d}", "job": "selftest", "readings": s,
              "sentences": count_sentences(text), "verdict": verdict,
              "verdict_ok": verdict is not None, "words": len(text.split())})
    if not quiet:
        print(f"=== self-report {index}")
        print(block)
        print("---")
        print(text)
        pw = r["power"]
        print(f"    {r['answer_ms']:.0f} ms, {r['prompt_ids']} prompt ids, {r['gen_ids']} gen ids, "
              f"{r['sentences']} sentence ends, {r['words']} words, verdict {r['verdict']!r}"
              + (f", {pw['joules']:.3f} J at {pw['mean_w']:.2f} W mean" if pw else ""))
        sys.stdout.flush()
    return r


def summarise(runs):
    tri = [r for r in runs if r["job"] == "triage"]
    if not tri:
        return {}
    clear = [r for r in tri if not r["ambiguous"]]
    amb = [r for r in tri if r["ambiguous"]]

    def frac(rows, key):
        return round(sum(1 for r in rows if r.get(key)) / len(rows), 4) if rows else None

    ms = sorted(r["answer_ms"] for r in tri)
    js = [r["power"]["joules"] for r in tri if r.get("power")]
    out = {
        "answers": len(tri), "clear": len(clear), "ambiguous": len(amb),
        "json_strict_frac": frac(tri, "json_strict_ok"),
        "json_recovered_frac": frac(tri, "json_recovered_ok"),
        "keys_exact_frac": frac(tri, "keys_exact"),
        "category_in_vocab_frac": frac(tri, "category_in_vocab"),
        "severity_in_vocab_frac": frac(tri, "severity_in_vocab"),
        "category_acc_clear": frac(clear, "category_ok"),
        "severity_acc_clear": frac(clear, "severity_ok"),
        "both_acc_clear": round(sum(1 for r in clear if r["category_ok"] and r["severity_ok"])
                                / len(clear), 4) if clear else None,
        "category_acc_ambiguous": frac(amb, "category_ok"),
        "severity_acc_ambiguous": frac(amb, "severity_ok"),
        "ms_mean": round(sum(ms) / len(ms), 1), "ms_min": ms[0], "ms_max": ms[-1],
        "ms_median": ms[len(ms) // 2],
        "prompt_tokens_mean": round(sum(r.get("prompt_tokens", r["prompt_ids"]) for r in tri)
                                    / len(tri), 1),
        "gen_tokens_mean": round(sum(r.get("gen_tokens", r["gen_ids"]) for r in tri) / len(tri), 1),
    }
    if js:
        out.update({"joules_mean": round(sum(js) / len(js), 4),
                    "joules_min": round(min(js), 4), "joules_max": round(max(js), 4)})
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", default="/home/ubuntu/bitnet-kria")
    ap.add_argument("--exe", default=None)
    ap.add_argument("--alarms", default=os.path.join(HERE, "edge_alarms.json"))
    ap.add_argument("--job", default="both", choices=("triage", "selftest", "both"))
    ap.add_argument("--only", default=None)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--repeat", type=int, default=1)
    ap.add_argument("--selftests", type=int, default=3)
    ap.add_argument("--selftest-gap", type=float, default=2.0)
    ap.add_argument("--saving", type=int, default=0)
    ap.add_argument("--live", action="store_true")
    ap.add_argument("--json-out", default=None)
    ap.add_argument("--max-new", type=int, default=96)
    ap.add_argument("--context", type=int, default=512)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--cache-dtype", default="f32", choices=("f32", "bf16", "i8"))
    ap.add_argument("--head", default="auto", choices=("auto", "fabric", "arm"),
                    help="which output head bitnet_kria uses; auto picks fabric when it is packed")
    ap.add_argument("--idle", type=float, default=0.0, help="seconds of idle power before the run")
    ap.add_argument("--no-power", action="store_true")
    ap.add_argument("--no-sudo", action="store_true")
    ap.add_argument("--quiet", action="store_true", help="do not print the runtime's banner")
    ap.add_argument("--no-transcript", action="store_true", help="do not print each answer")
    a = ap.parse_args()
    a.exe = a.exe or os.path.join(a.dir, "bitnet_kria")

    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    spec = json.load(open(a.alarms, encoding="utf-8"))
    alarms = spec["alarms"]
    if a.only:
        want = set(a.only.split(","))
        alarms = [x for x in alarms if x["id"] in want]
    if a.limit:
        alarms = alarms[:a.limit]

    power = Power() if not a.no_power else type("Off", (), {"on": False, "window": lambda *_: None,
                                                            "close": lambda *_: None,
                                                            "idle": lambda *_: None})()
    print(f"edge_monitor: {len(alarms)} alarms "
          f"({sum(1 for x in alarms if x.get('ambiguous'))} marked ambiguous), job {a.job}, "
          f"power sampling {'on' if power.on else 'off'}", flush=True)
    idle_before = power.idle(a.idle) if a.idle else None
    if idle_before:
        print(f"idle before the run: {idle_before['mean_w']:.3f} W mean over {a.idle:.0f} s",
              flush=True)

    tok = bc.Tok(os.path.join(a.dir, "tokenizer.json"))
    net_run0 = net_counters()
    t_run0 = time.time()
    engine, setup_s, cmd = open_engine(a, quiet=a.quiet)
    print(f"--- one bitnet_kria open in {setup_s:.2f} s: {' '.join(cmd)}", flush=True)

    runs, saving = [], None
    try:
        rounds = 0
        while True:
            rounds += 1
            if a.job in ("selftest", "both"):
                n = 1 if a.live else a.selftests
                for i in range(n):
                    runs.append(do_selftest(engine, tok, power, len(
                        [r for r in runs if r["job"] == "selftest"]) + 1,
                        quiet=a.no_transcript))
                    if i + 1 < n and a.selftest_gap:
                        time.sleep(a.selftest_gap)
            if a.job in ("triage", "both"):
                for rep in range(a.repeat):
                    for x in alarms:
                        runs.append(do_triage(engine, tok, power, x, rep,
                                              quiet=a.no_transcript))
            if not a.live:
                break
            print(f"--- round {rounds} done, going round again (ctrl-C to stop)", flush=True)

        if a.saving:
            saving = measure_saving(a, tok, power, alarms, runs, engine)
    except KeyboardInterrupt:
        print("\n(stopped)", flush=True)
    finally:
        engine.close()
    t_run1 = time.time()
    net_run1 = net_counters()

    summ = summarise(runs)
    print("\n--- numbers")
    for k, v in summ.items():
        print(f"  {k:26s} {v}")
    if saving:
        print("\n--- what one long-lived process saves")
        for k, v in saving.items():
            print(f"  {k:26s} {v}")
    print("\n--- /proc/net/dev across the whole run "
          f"({t_run1 - t_run0:.1f} s, {len(runs)} answers)")
    for k in sorted(net_run0):
        if k in net_run1:
            d = {m: net_run1[k][m] - net_run0[k][m] for m in net_run0[k]}
            print(f"  {k:8s} rx {d['rx_bytes']:>12d} B / {d['rx_packets']:>7d} pkt   "
                  f"tx {d['tx_bytes']:>12d} B / {d['tx_packets']:>7d} pkt")
    print("  (this process printed its transcript down the ssh control connection, which is itself "
          "eth0 traffic; the inference read nothing from any interface)")

    if a.json_out:
        json.dump({"what": "edge_monitor on the KV260", "alarms_file": os.path.abspath(a.alarms),
                   "alarms": len(alarms),
                   "ambiguous": sum(1 for x in alarms if x.get("ambiguous")),
                   "cmd": cmd, "setup_s": round(setup_s, 3),
                   "system_triage": SYSTEM_TRIAGE, "system_selftest": SYSTEM_SELFTEST,
                   "max_new": a.max_new, "context": a.context, "threads": a.threads,
                   "cache_dtype": a.cache_dtype, "greedy": True,
                   "idle_before": idle_before, "summary": summ, "saving": saving,
                   "net_before": net_run0, "net_after": net_run1,
                   "run_s": round(t_run1 - t_run0, 3), "runs": runs},
                  open(a.json_out, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
        print(f"\nwrote {a.json_out}")
    power.close()
    return 0


def measure_saving(a, tok, power, alarms, runs, engine):
    """The same alarms twice: once more on the process that is already open, then once each on a
    process started for that one answer. The difference is what holding the weights costs."""
    n = min(a.saving, len(alarms))
    picked = alarms[:n]
    print(f"\n--- what one long-lived process saves: {n} alarms on the open process, "
          f"then {n} on fresh ones", flush=True)
    shared = []
    for x in picked:
        r = do_triage(engine, tok, power, x, 0, quiet=True)
        r["mode"] = "shared"
        shared.append(r)
        runs.append(r)
        print(f"    shared {x['id']}: {r['answer_ms']:.0f} ms", flush=True)
    fresh = []
    for x in picked:
        t0 = time.time()
        eng2, setup_s, _cmd = open_engine(a, quiet=True)
        try:
            r = do_triage(eng2, tok, power, x, 0, quiet=True)
        finally:
            eng2.close()
        r["mode"] = "fresh"
        r["setup_s"] = round(setup_s, 3)
        r["end_to_end_ms"] = round((time.time() - t0) * 1e3, 1)
        fresh.append(r)
        runs.append(r)
        print(f"    fresh  {x['id']}: {r['end_to_end_ms']:.0f} ms end to end "
              f"({setup_s * 1e3:.0f} ms of it is setup)", flush=True)
    sm = sum(r["answer_ms"] for r in shared) / len(shared)
    fm = sum(r["end_to_end_ms"] for r in fresh) / len(fresh)
    fa = sum(r["answer_ms"] for r in fresh) / len(fresh)
    su = sum(r["setup_s"] for r in fresh) / len(fresh) * 1e3
    same = sum(1 for x, y in zip(shared, fresh) if x["answer"] == y["answer"])
    return {"n": n, "shared_ms_mean": round(sm, 1), "fresh_end_to_end_ms_mean": round(fm, 1),
            "fresh_answer_ms_mean": round(fa, 1), "fresh_setup_ms_mean": round(su, 1),
            "saving_ms_per_answer": round(fm - sm, 1),
            "saving_frac": round((fm - sm) / fm, 4),
            "identical_answers": f"{same} of {len(shared)}"}


if __name__ == "__main__":
    sys.exit(main())
