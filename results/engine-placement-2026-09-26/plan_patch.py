import re

T = {}
for l in open("sweep.txt"):
    m = re.match(r"sweep grid\s+mod64K\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+) K\s+GB/s ([\d. ]+?)\s+all", l)
    if m:
        o = [int(m.group(i)) for i in range(1, 5)]
        T[(o[1] // 8, o[2] // 8, o[3] // 8)] = [float(x) for x in m.group(5).split()]
assert len(T) == 512

rows = []
for a in range(8):
    for b in range(8):
        for c in range(8):
            rows.append("{%s}" % ", ".join("%.3f" % x for x in T[(a, b, c)]))
table = "static const float BANK_RATE[512][4] = {\n    " + ",\n    ".join(rows) + "\n};\n"

s = open("bitnet_kria.c").read()

plan = table + r'''
/* engine rates measured on this board with each engine's start placed at 8 KiB steps modulo 64 KiB
   relative to engine 0 (index m1*64 + m2*8 + m3); a split is scored by its slowest engine */
static double plan_score(const unsigned *n, unsigned rowbytes)
{
    uint64_t st[4] = { 0, n[0], n[0] + n[1], n[0] + n[1] + n[2] };
    unsigned m[4];
    for (unsigned i = 0; i < E; i++) m[i] = (unsigned)(((st[i] * rowbytes % 65536u) + 4096u) / 8192u) % 8u;
    const float *r = BANK_RATE[m[1] * 64u + m[2] * 8u + m[3]];
    double worst = 0;
    for (unsigned i = 0; i < E; i++) {
        double t = (double)n[i] * rowbytes / r[i];
        if (t > worst) worst = t;
    }
    return worst;
}

#define PLAN_N 8
static struct { char what[16]; unsigned total, rowbytes, n[4]; } PLANS[PLAN_N];
static int PLAN_ON = -1;

static int plan_for(const char *what, unsigned total, unsigned *n, unsigned rowbytes)
{
    if (PLAN_ON < 0) { const char *pe = getenv("ENGINE_PLAN"); PLAN_ON = pe && pe[0] == '1'; }
    if (!PLAN_ON || total % 64u) return 0;
    int k;
    for (k = 0; k < PLAN_N && PLANS[k].what[0]; k++)
        if (!strcmp(PLANS[k].what, what) && PLANS[k].total == total && PLANS[k].rowbytes == rowbytes) {
            memcpy(n, PLANS[k].n, sizeof PLANS[k].n);
            return 1;
        }
    const int base = (int)(total / 64u), units = (int)(total / 16u), span = 6;
    double best = -1;
    unsigned bn[4] = { 0, 0, 0, 0 };
    for (int a = -span; a <= span; a++)
        for (int b = -span; b <= span; b++)
            for (int c = -span; c <= span; c++) {
                int d = units - 3 * base - a - b - c;
                if (base + a < 1 || base + b < 1 || base + c < 1 || d < 1) continue;
                unsigned t[4] = { 16u * (unsigned)(base + a), 16u * (unsigned)(base + b), 16u * (unsigned)(base + c), 16u * (unsigned)d };
                if (t[0] > 0xFFFFu || t[1] > 0xFFFFu || t[2] > 0xFFFFu || t[3] > 0xFFFFu) continue;
                double sc = plan_score(t, rowbytes);
                if (best < 0 || sc < best) { best = sc; memcpy(bn, t, sizeof bn); }
            }
    if (best < 0) return 0;
    if (bn[0] + bn[1] + bn[2] + bn[3] != total) { fprintf(stderr, "engine plan: %s rows do not add up\n", what); exit(2); }
    unsigned even[4] = { total / 4u, total / 4u, total / 4u, total / 4u };
    fprintf(stderr, "engine plan %s: rows %u %u %u %u, predicted %.1f us against %.1f even\n", what,
            bn[0], bn[1], bn[2], bn[3], best / 1e3, plan_score(even, rowbytes) / 1e3);
    if (k < PLAN_N) {
        snprintf(PLANS[k].what, sizeof PLANS[k].what, "%s", what);
        PLANS[k].total = total; PLANS[k].rowbytes = rowbytes;
        memcpy(PLANS[k].n, bn, sizeof bn);
    }
    memcpy(n, bn, sizeof bn);
    return 1;
}

static int engine_phase('''

old = "static int engine_phase("
assert s.count(old) == 1
s = s.replace(old, plan)

old = "        if (even_layout && shares_for(what, npe * E, n, B * BEAT_BYTES)) {"
assert s.count(old) == 1
s = s.replace(old, "        if (even_layout && (plan_for(what, npe * E, n, B * BEAT_BYTES) || shares_for(what, npe * E, n, B * BEAT_BYTES))) {")
open("bitnet_kria.c", "w").write(s)
print("plan patched")
