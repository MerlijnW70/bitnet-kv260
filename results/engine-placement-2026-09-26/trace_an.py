import re, sys, collections

path = sys.argv[1] if len(sys.argv) > 1 else "/home/ubuntu/trace/trace.txt"
line_re = re.compile(r"^\s*(.+?)-(\d+)\s+\[(\d+)\]\s+\S+\s+([\d.]+): (\w+): (.*)$")
kv_re = re.compile(r"(\w+)=(\S+)")

cur = {}            # cpu -> (comm, pid) running now
pending = {}        # pool pid -> (wake time, target cpu, blocker comm on that cpu at wake)
preempt = {}        # pool pid -> (time switched out while runnable, comm that took the cpu)
irq_start = {}      # cpu -> (t, name)
soft_start = {}     # cpu -> (t, vec)
wake_wait = collections.Counter()
wake_wait_n = collections.Counter()
pre_wait = collections.Counter()
pre_n = collections.Counter()
irq_time = collections.Counter()
irq_cpu = collections.Counter()
soft_time = collections.Counter()
soft_cpu = collections.Counter()
migr = collections.Counter()
bit = set()
t_first = t_last = None
waits_all = []

for line in open(path, errors="replace"):
    m = line_re.match(line)
    if not m:
        continue
    comm, pid, cpu, t, ev, rest = m.group(1).strip(), int(m.group(2)), int(m.group(3)), float(m.group(4)), m.group(5), m.group(6)
    if t_first is None:
        t_first = t
    t_last = t
    if ev == "sched_switch":
        a = rest.split(" ==> ")
        p = dict(kv_re.findall(a[0]))
        n = dict(kv_re.findall(a[1]))
        pc, pp, st = p.get("prev_comm", ""), int(p.get("prev_pid", 0)), p.get("prev_state", "")
        nc, np_ = n.get("next_comm", ""), int(n.get("next_pid", 0))
        if pc == "bitnet_kria":
            bit.add(pp)
            if st.startswith("R"):
                preempt[pp] = (t, nc if nc != "swapper/%d" % cpu else nc)
        if nc == "bitnet_kria":
            bit.add(np_)
            if np_ in pending:
                tw, tc, blk = pending.pop(np_)
                w = (t - tw) * 1e3
                wake_wait[blk] += w
                wake_wait_n[blk] += 1
                waits_all.append(w)
            if np_ in preempt:
                tp, who = preempt.pop(np_)
                pre_wait[who] += (t - tp) * 1e3
                pre_n[who] += 1
        cur[cpu] = (nc, np_)
    elif ev == "sched_wakeup":
        d = dict(kv_re.findall(rest))
        if d.get("comm") == "bitnet_kria":
            wp = int(d["pid"]); tc = int(d.get("target_cpu", "0"))
            blk = cur.get(tc, ("?", 0))
            name = blk[0]
            if name == "bitnet_kria":
                name = "bitnet_kria (own thread)"
            pending[wp] = (t, tc, name)
    elif ev == "sched_migrate_task":
        d = dict(kv_re.findall(rest))
        if d.get("comm") == "bitnet_kria":
            migr[(d.get("orig_cpu"), d.get("dest_cpu"))] += 1
    elif ev == "irq_handler_entry":
        d = dict(kv_re.findall(rest))
        irq_start[cpu] = (t, d.get("name", d.get("irq")))
    elif ev == "irq_handler_exit":
        if cpu in irq_start:
            ts, name = irq_start.pop(cpu)
            irq_time[name] += (t - ts) * 1e3
            irq_cpu[cpu] += (t - ts) * 1e3
    elif ev == "softirq_entry":
        vm = re.search(r"action=(\w+)", rest)
        soft_start[cpu] = (t, vm.group(1) if vm else rest)
    elif ev == "softirq_exit":
        if cpu in soft_start:
            ts, name = soft_start.pop(cpu)
            soft_time[name] += (t - ts) * 1e3
            soft_cpu[cpu] += (t - ts) * 1e3

span = (t_last - t_first) if t_first else 0
print("trace span %.2f s, bitnet_kria threads seen %d" % (span, len(bit)))
print("\nwoken bitnet_kria thread had to wait; what the target cpu was running when it woke (ms total, count):")
for k, v in wake_wait.most_common(12):
    print("  %-32s %9.1f ms  %7d" % (k, v, wake_wait_n[k]))
if waits_all:
    ws = sorted(waits_all)
    q = lambda f: ws[min(len(ws) - 1, int(f * len(ws)))]
    print("  wake->run: n %d  median %.3f ms  p90 %.3f  p99 %.3f  max %.3f  total %.1f ms" %
          (len(ws), q(0.5), q(0.9), q(0.99), ws[-1], sum(ws)))
print("\nbitnet_kria thread preempted while runnable; who took the cpu (ms until it ran again, count):")
for k, v in pre_wait.most_common(12):
    print("  %-32s %9.1f ms  %7d" % (k, v, pre_n[k]))
print("\nhard irq time (ms): by cpu", {c: round(v, 1) for c, v in sorted(irq_cpu.items())})
for k, v in irq_time.most_common(8):
    print("  %-32s %9.1f ms" % (k, v))
print("softirq time (ms): by cpu", {c: round(v, 1) for c, v in sorted(soft_cpu.items())})
for k, v in soft_time.most_common(8):
    print("  %-32s %9.1f ms" % (k, v))
print("\nbitnet_kria migrations (orig->dest: count):", dict(migr.most_common(12)))
