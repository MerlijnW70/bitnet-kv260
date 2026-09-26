s = open("bitnet_kria.c").read()

def once(old, new):
    global s
    assert s.count(old) == 1, old[:70]
    s = s.replace(old, new)

once("static struct pool POOL;\n", r'''static struct pool POOL;
static int PIN_ON = -1;

/* ENGINE_PIN=1: thread i of the pool on cpu i (the main thread is thread 0), the side thread on the
   cpus the pool does not use, or on every cpu but 0 when the pool uses them all */
static void pin_self(int cpu_lo, int cpu_hi)
{
    if (PIN_ON < 0) { const char *pe = getenv("ENGINE_PIN"); PIN_ON = pe && pe[0] == '1'; }
    if (!PIN_ON) return;
    long ncpu = sysconf(_SC_NPROCESSORS_ONLN);
    if (cpu_hi >= ncpu) cpu_hi = (int)ncpu - 1;
    if (cpu_lo > cpu_hi) return;
    cpu_set_t set;
    CPU_ZERO(&set);
    for (int c = cpu_lo; c <= cpu_hi; c++) CPU_SET(c, &set);
    if (sched_setaffinity(0, sizeof set, &set)) perror("sched_setaffinity");
}
''')

once("""    long id = (long)v;
    unsigned seen = 0, hot = 0;
""", """    long id = (long)v;
    unsigned seen = 0, hot = 0;
    pin_self((int)id, (int)id);
""")

once("""    POOL.n = n;
""", """    POOL.n = n;
    pin_self(0, 0);
""")

once("""    unsigned seen = 0;
    (void)v;
""", """    unsigned seen = 0;
    (void)v;
    pin_self(POOL.n < 4 ? POOL.n : 1, 3);
""")

open("bitnet_kria.c", "w").write(s)
print("pin patched")
