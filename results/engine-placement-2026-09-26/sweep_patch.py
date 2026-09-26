s = open("bitnet_kria.c").read()

sweep = r'''static void sweep_point(const char *label, const uint64_t *off, unsigned reps, unsigned npe, unsigned B,
                        uint64_t act_phys, size_t act_bytes)
{
    uint64_t w[4], r[4];
    const size_t len = (size_t)npe * B * BEAT_BYTES;
    for (unsigned i = 0; i < E; i++) {
        if (off[i] + len > B0.size) { fprintf(stderr, "sweep: engine %u would read past udmabuf0\n", i); exit(2); }
        w[i] = B0.phys + off[i];
        r[i] = B1.phys + OFF_OI + (uint64_t)i * npe * 4u;
    }
    struct probe_phase *p = probe_slot("sweep", 1u);
    if (!p) { fprintf(stderr, "sweep: no probe slot\n"); exit(2); }
    double a = 0, ww = 0;
    if (engine_phase("sweep", B, 1u, npe, act_phys, act_bytes, w, r, &a, &ww, 0, 1, 0, 0)) exit(3);
    p->clean = p->missed = 0; p->all_fin = 0;
    for (unsigned i = 0; i < E; i++) p->fin[i] = p->bytes[i] = p->lead_done[i] = 0;
    for (unsigned k = 0; k < reps; k++)
        if (engine_phase("sweep", B, 1u, npe, act_phys, act_bytes, w, r, &a, &ww, 0, 1, 0, 0)) exit(3);
    double n = (double)(p->clean ? p->clean : 1);
    printf("sweep %-6s mod64K %2llu %2llu %2llu %2llu K  GB/s", label,
           (unsigned long long)(w[0] % 65536u / 1024u), (unsigned long long)(w[1] % 65536u / 1024u),
           (unsigned long long)(w[2] % 65536u / 1024u), (unsigned long long)(w[3] % 65536u / 1024u));
    for (unsigned i = 0; i < E; i++) printf(" %.3f", p->bytes[i] / n / (p->fin[i] / n * 1e-3) / 1e9);
    printf("  all %.1f us  clean %lu missed %lu\n", p->all_fin / n * 1000.0, p->clean, p->missed);
    fflush(stdout);
}

static void engine_sweep(void)
{
    const unsigned B = M.mat[0][PROJ_O].beats, npe = M.mat[0][PROJ_O].n / E, reps = 40;
    const uint64_t act_phys = B1.phys + OFF_ACT;
    const size_t act_bytes = (size_t)B * M.wpb;
    const uint64_t X = 64ull << 20, REG = 2ull << 20, K8 = 8192;
    const uint64_t real = M.mat[0][PROJ_O].offset;
    uint64_t off[4];
    PROBE_ON = 1;
    printf("sweep: o_proj shape, %u rows of %u beats an engine, %u reps a point, udmabuf0 phys 0x%llx\n",
           npe, B, reps, (unsigned long long)B0.phys);

    for (unsigned i = 0; i < E; i++) off[i] = real + (uint64_t)i * npe * B * BEAT_BYTES;
    sweep_point("real", off, reps, npe, B, act_phys, act_bytes);

    for (unsigned m1 = 0; m1 < 8; m1++)
        for (unsigned m2 = 0; m2 < 8; m2++)
            for (unsigned m3 = 0; m3 < 8; m3++) {
                off[0] = X;
                off[1] = X + REG + m1 * K8;
                off[2] = X + 2 * REG + m2 * K8;
                off[3] = X + 3 * REG + m3 * K8;
                sweep_point("grid", off, reps, npe, B, act_phys, act_bytes);
            }

    const unsigned pick[4][3] = { { 0, 0, 0 }, { 2, 4, 6 }, { 4, 0, 4 }, { 1, 5, 3 } };
    for (unsigned c = 0; c < 4; c++)
        for (unsigned sh = 0; sh < 8; sh++) {
            off[0] = X + sh * K8;
            off[1] = X + REG + (pick[c][0] + sh) % 8 * K8;
            off[2] = X + 2 * REG + (pick[c][1] + sh) % 8 * K8;
            off[3] = X + 3 * REG + (pick[c][2] + sh) % 8 * K8;
            sweep_point("shift", off, reps, npe, B, act_phys, act_bytes);
        }

    for (unsigned i = 0; i < E; i++) off[i] = real + (uint64_t)i * npe * B * BEAT_BYTES;
    sweep_point("real", off, reps, npe, B, act_phys, act_bytes);
}

static void print_timing(void)
'''
old = "static void print_timing(void)\n"
assert s.count(old) == 1
s = s.replace(old, sweep)

old = "    int rc = 0;\n\n    if (stage_file) {"
assert s.count(old) == 1
s = s.replace(old, "    if (getenv(\"ENGINE_SWEEP\")) { engine_sweep(); return 0; }\n\n" + old)
open("bitnet_kria.c", "w").write(s)
print("sweep patched")
