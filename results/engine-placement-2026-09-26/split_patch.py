s = open("bitnet_kria.c").read()

a = s.index("static int engine_phase(")
b = s.index("static int glue_pass(")

new = r'''#define PROBE_NPH 16
struct probe_phase {
    const char *what;
    unsigned batch;
    unsigned long clean, missed;
    double fin[4], bytes[4], lead_done[4], all_fin;
};
static struct probe_phase PROBE[PROBE_NPH];
static int PROBE_ON = -1;

static struct probe_phase *probe_slot(const char *what, unsigned batch)
{
    for (int k = 0; k < PROBE_NPH; k++) {
        if (PROBE[k].what && !strcmp(PROBE[k].what, what) && PROBE[k].batch == batch) return &PROBE[k];
        if (!PROBE[k].what) { PROBE[k].what = what; PROBE[k].batch = batch; return &PROBE[k]; }
    }
    return 0;
}

static void probe_chunk(const char *what, unsigned batch, double tarm, const unsigned *cr, const size_t *cwb)
{
    struct probe_phase *p = probe_slot(what, batch);
    double fin[4] = { -1, -1, -1, -1 }, lead[4] = { 0, 0, 0, 0 };
    unsigned left = E, i, first = 1, have_lead = 0;
    double tw = now_ms();
    while (left) {
        uint32_t d[4];
        double t = now_ms();
        for (i = 0; i < E; i++) d[i] = STATUS_DONE(rd(gpio[i], GPIO2_DATA));
        for (i = 0; i < E; i++) {
            if (fin[i] < 0 && d[i] >= cr[i]) {
                if (first) { if (p) p->missed++; return; }
                fin[i] = t - tarm;
                left--;
                if (!have_lead) {
                    for (unsigned j = 0; j < E; j++) lead[j] = (double)d[j] / cr[j];
                    have_lead = 1;
                }
            }
        }
        first = 0;
        if (t - tw > TIMEOUT_MS) return;
    }
    if (!p) return;
    p->clean++;
    double mx = 0;
    for (i = 0; i < E; i++) {
        p->fin[i] += fin[i];
        p->bytes[i] += (double)cwb[i];
        p->lead_done[i] += lead[i];
        if (fin[i] > mx) mx = fin[i];
    }
    p->all_fin += mx;
}

static void probe_report(void)
{
    if (PROBE_ON != 1) return;
    printf("engine probe (prewake off; a burst is 'missed' when an engine had finished before the first poll)\n");
    for (int k = 0; k < PROBE_NPH && PROBE[k].what; k++) {
        struct probe_phase *p = &PROBE[k];
        if (!p->clean) { printf("  %-8s batch %u clean 0 missed %lu\n", p->what, p->batch, p->missed); continue; }
        double n = (double)p->clean, all = p->all_fin / n, tot = 0;
        for (unsigned i = 0; i < E; i++) tot += p->bytes[i] / n;
        printf("  %-8s batch %u clean %lu missed %lu, all four done at %.1f us = %.2f GB/s together\n",
               p->what, p->batch, p->clean, p->missed, all * 1000.0, tot / (all * 1e-3) / 1e9);
        for (unsigned i = 0; i < E; i++) {
            double f = p->fin[i] / n, bpe = p->bytes[i] / n;
            printf("    engine %u  %8.0f B  done at %8.1f us  %6.2f GB/s  %5.1f%% done when the first engine finished\n",
                   i, bpe, f * 1000.0, bpe / (f * 1e-3) / 1e9, 100.0 * p->lead_done[i] / n);
        }
    }
}

#define SHARE_NPH 8
static struct { char what[16]; double w[4]; } SHARES[SHARE_NPH];
static int SHARES_N = -1, BANKFIX = 0;

static void shares_parse(void)
{
    SHARES_N = 0;
    { const char *bf = getenv("ENGINE_BANKFIX"); BANKFIX = bf && bf[0] == '1'; }
    const char *env = getenv("ENGINE_SHARES");
    if (!env) return;
    char buf[512];
    snprintf(buf, sizeof buf, "%s", env);
    for (char *save = 0, *tok = strtok_r(buf, ";", &save); tok && SHARES_N < SHARE_NPH; tok = strtok_r(0, ";", &save)) {
        char *colon = strchr(tok, ':');
        double w[4];
        if (!colon || sscanf(colon + 1, "%lf,%lf,%lf,%lf", &w[0], &w[1], &w[2], &w[3]) != 4) {
            fprintf(stderr, "ENGINE_SHARES: cannot read '%s' (want name:w0,w1,w2,w3)\n", tok);
            exit(2);
        }
        for (int i = 0; i < 4; i++)
            if (!(w[i] > 0.1 && w[i] < 10.0)) { fprintf(stderr, "ENGINE_SHARES: weight out of range in '%s'\n", tok); exit(2); }
        *colon = 0;
        snprintf(SHARES[SHARES_N].what, sizeof SHARES[SHARES_N].what, "%s", tok);
        memcpy(SHARES[SHARES_N].w, w, sizeof w);
        fprintf(stderr, "engine shares %s: %.3f %.3f %.3f %.3f\n", tok, w[0], w[1], w[2], w[3]);
        SHARES_N++;
    }
}

/* rows an engine, in units of 16 rows, proportional to the phase's weights; 0 = keep the even split.
   Engines 1 and 2 share one DDR controller port; engine 1's rows are then moved a few units (engine 2
   taking the difference) so that the gap between their start addresses sits as near 32 KiB modulo
   64 KiB as it can, the farthest two streams can be from the same bank with bank bits 14-15 */
static int shares_for(const char *what, unsigned total, unsigned *n, unsigned rowbytes)
{
    if (SHARES_N < 0) shares_parse();
    const double *w = 0;
    for (int k = 0; k < SHARES_N; k++) if (!strcmp(SHARES[k].what, what)) w = SHARES[k].w;
    if (!w || total % 16u) return 0;
    const unsigned units = total / 16u;
    double sum = w[0] + w[1] + w[2] + w[3];
    unsigned got = 0;
    for (unsigned i = 0; i < E; i++) { n[i] = (unsigned)(units * w[i] / sum + 0.5); got += n[i]; }
    while (got > units) { unsigned m = 0; for (unsigned i = 1; i < E; i++) if (n[i] > n[m]) m = i; n[m]--; got--; }
    while (got < units) { unsigned m = 0; for (unsigned i = 1; i < E; i++) if (n[i] < n[m]) m = i; n[m]++; got++; }
    if (BANKFIX) {
        int best = 0;
        long bestd = -1;
        for (int dlt = -4; dlt <= 4; dlt++) {
            long n1 = (long)n[1] + dlt, n2 = (long)n[2] - dlt;
            if (n1 < 1 || n2 < 1) continue;
            long gap = (long)((uint64_t)n1 * 16u * rowbytes % 65536u);
            long dist = gap > 32768 ? gap - 32768 : 32768 - gap;
            if (bestd < 0 || dist < bestd || (dist == bestd && abs(dlt) < abs(best))) { bestd = dist; best = dlt; }
        }
        n[1] += best; n[2] -= best;
    }
    for (unsigned i = 0; i < E; i++) { if (!n[i] || n[i] * 16u > 0xFFFFu) return 0; n[i] *= 16u; }
    static char told[SHARE_NPH][16];
    for (int k = 0; k < SHARE_NPH; k++) {
        if (!strcmp(told[k], what)) break;
        if (!told[k][0]) {
            snprintf(told[k], sizeof told[k], "%s", what);
            fprintf(stderr, "engine rows %s: %u %u %u %u, engine 1-2 gap %u mod 64 KiB\n", what, n[0], n[1], n[2], n[3],
                    (unsigned)((uint64_t)n[1] * rowbytes % 65536u));
            break;
        }
    }
    return 1;
}

static int engine_phase(const char *what, unsigned B, unsigned batch, unsigned npe, uint64_t act_phys,
                        size_t act_bytes, const uint64_t *w_phys, const uint64_t *r_phys,
                        double *ms_act, double *ms_w,
                        int prewake, unsigned chunks, void (*on_chunk)(void *, unsigned, unsigned), void *carg)
{
    const unsigned CH = chunks < 1 || npe % chunks ? 1u : chunks;
    const unsigned act_beats = (unsigned)(batch * act_bytes / BEAT_BYTES);
    const uint32_t bits = CTRL_BEATS(B) | CTRL_BATCH(batch);
    unsigned i, c;
    unsigned cr[4];
    size_t cwbytes[4], crbytes[4];
    uint64_t wp[4], rp[4];
    double ta = now_ms();
    CHUNK_WORK_MS = 0;
    if (PROBE_ON < 0) { const char *pe = getenv("ENGINE_PROBE"); PROBE_ON = pe && pe[0] == '1'; }
    if (PROBE_ON) prewake = 0;
    double tarm;

    for (i = 0; i < E; i++) {
        cr[i] = npe / CH;
        wp[i] = w_phys[i];
        rp[i] = r_phys[i];
    }
    if (batch == 1 && CH == 1) {
        unsigned n[4];
        int even_layout = 1;
        for (i = 0; i < E; i++)
            if (w_phys[i] != w_phys[0] + (uint64_t)i * npe * B * BEAT_BYTES ||
                r_phys[i] != r_phys[0] + (uint64_t)i * npe * 4u) even_layout = 0;
        if (even_layout && shares_for(what, npe * E, n, B * BEAT_BYTES)) {
            unsigned start = 0;
            for (i = 0; i < E; i++) {
                cr[i] = n[i];
                wp[i] = w_phys[0] + (uint64_t)start * B * BEAT_BYTES;
                rp[i] = r_phys[0] + (uint64_t)start * 4u;
                start += n[i];
            }
            if (start != npe * E) { fprintf(stderr, "engine shares: %u rows of %u\n", start, npe * E); exit(2); }
        }
    }
    for (i = 0; i < E; i++) {
        cwbytes[i] = (size_t)cr[i] * B * BEAT_BYTES;
        crbytes[i] = (size_t)cr[i] * batch * 4u;
    }

    for (i = 0; i < E; i++) s2mm_start(i, rp[i], crbytes[i]);
    for (i = 0; i < E; i++) {
        wr(gpio[i], GPIO_DATA, CTRL_NEURONS(cr[i] * CH) | bits | CTRL_RUN);
        mm2s_start(i, act_phys, batch * act_bytes);
    }
    for (i = 0; i < E; i++)
        if (wait_ioc(i, MM2S_DMASR, what)) { print_dma_status(); return 3; }
    for (i = 0; i < E; i++) {
        uint32_t s;
        if (wait_loaded(i, act_beats, &s)) {
            fprintf(stderr, "  %s: engine %u status 0x%08x (state %u, %u beats) never reached %u | idle\n",
                    what, i, s, STATUS_STATE(s), STATUS_ACT(s), act_beats);
            print_dma_status();
            return 3;
        }
    }
    for (i = 0; i < E; i++) wr(gpio[i], GPIO_DATA, 0);

    double t0 = now_ms();
    for (i = 0; i < E; i++) wr(gpio[i], GPIO_DATA, CTRL_NEURONS(cr[i]) | bits | CTRL_MODE_WEIGHTS | CTRL_RUN);
    for (i = 0; i < E; i++) mm2s_start(i, wp[i], cwbytes[i]);
    tarm = now_ms();
    for (c = 0; c < CH; c++) {
        if (PROBE_ON) probe_chunk(what, batch, tarm, cr, cwbytes);
        if (prewake) {
            unsigned cpn = B * ((batch + 1u) / 2u);
            unsigned left = 50000u / (cpn ? cpn : 1u), thresh = cr[0] > left ? cr[0] - left : 0;
            double tw = now_ms();
            while (thresh && STATUS_DONE(rd(gpio[0], GPIO2_DATA)) < thresh && now_ms() - tw < TIMEOUT_MS)
                ;
            pool_prewake();
        }
        for (i = 0; i < E; i++)
            if (wait_ioc(i, MM2S_DMASR, what)) { print_dma_status(); return 3; }
        for (i = 0; i < E; i++)
            if (wait_ioc(i, S2MM_DMASR, what)) { print_dma_status(); return 3; }
        for (i = 0; i < E; i++) {
            uint32_t s;
            if (wait_done(i, cr[i], &s)) {
                fprintf(stderr, "  %s: engine %u status 0x%08x (done %u) never reached %u | idle\n", what, i, s, STATUS_DONE(s), cr[i]);
                print_dma_status();
                return 3;
            }
        }
        for (i = 0; i < E; i++) wr(gpio[i], GPIO_DATA, 0);
        if (c + 1 < CH) {
            for (i = 0; i < E; i++) s2mm_start(i, rp[i] + (uint64_t)(c + 1) * crbytes[i], crbytes[i]);
            for (i = 0; i < E; i++) wr(gpio[i], GPIO_DATA, CTRL_NEURONS(cr[i]) | bits | CTRL_MODE_WEIGHTS | CTRL_RUN);
            for (i = 0; i < E; i++) mm2s_start(i, wp[i] + (uint64_t)(c + 1) * cwbytes[i], cwbytes[i]);
            tarm = now_ms();
        }
        if (on_chunk) on_chunk(carg, c, CH);
    }
    double t1 = now_ms();
    *ms_act += t0 - ta;
    *ms_w += t1 - t0 - CHUNK_WORK_MS;
    return 0;
}

'''
s = s[:a] + new + s[b:]

old = "static void print_timing(void)\n{\n"
assert s.count(old) == 1
s = s.replace(old, old + "    probe_report();\n")
open("bitnet_kria.c", "w").write(s)
print("patched")
