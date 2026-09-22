#define _GNU_SOURCE
#include <errno.h>
#include <fcntl.h>
#include <inttypes.h>
#include <math.h>
#include <pthread.h>
#include <sched.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>
#ifdef __aarch64__
#include <arm_neon.h>
#endif

#include "refstages.h"

#define E 4u
#define GDMA E
#define PAGE 4096u
#define MAXT 8
#define PROJ_Q 0
#define PROJ_K 1
#define PROJ_V 2
#define PROJ_O 3
#define PROJ_GATE 4
#define PROJ_UP 5
#define PROJ_DOWN 6
#define NPROJ 7

#define OFF_ACT 0x000000u
#define OFF_QKV 0x008000u
#define OFF_OI 0x018000u
#define HEAD_ROW_PAD (E * 8u)
#define OFF_GU 0x023000u
#define OFF_H 0x059000u
#define OFF_Q8 0x074000u
#define OFF_D 0x07B000u
#define OFF_BEATS 0x085000u
#define OFF_HEADSUM 0x0F1000u
#define NEED1 (OFF_HEADSUM)
#define HEAD_KMAX 4096

#define GPIO_DATA 0x00
#define GPIO2_DATA 0x08
#define CTRL_RUN (1u << 0)
#define CTRL_MODE_WEIGHTS (1u << 1)
#define CTRL_BEATS(b) (((uint32_t)(b) & 0x7Fu) << 2)
#define CTRL_BATCH(v) ((((uint32_t)(v) - 1u) & 3u) << 9)
#define CTRL_NEURONS(n) ((uint32_t)(n) << 16)
#define STATUS_DONE(s) ((s) & 0xFFFFu)
#define STATUS_ACT(s) (((s) >> 16) & 0x7FFu)
#define STATUS_STATE(s) (((s) >> 27) & 0x7u)
#define STATUS_IDLE (1u << 31)
#define BMAX 4u
#define BEAT_BYTES 16u
#define GCTRL_RUN (1u << 0)
#define GCTRL_MODE_B (1u << 1)
#define GCTRL_N(n) ((uint32_t)(n) << 16)
#define GSTATUS_HMAX(s) ((s) & 0x7FFFu)
#define GSTATUS_DONE(s) (((s) >> 16) & 0x7FFFu)
#define GSTATUS_IDLE (1u << 31)
#define PARAMS(sg, su, sS, sT, sV) ((uint32_t)(sg) | (uint32_t)(su) << 5 | (uint32_t)(sS) << 10 | (uint32_t)(sT) << 15 | (uint32_t)(sV) << 20)

#define MM2S_DMACR 0x00
#define MM2S_DMASR 0x04
#define MM2S_SA 0x18
#define MM2S_SA_MSB 0x1C
#define MM2S_LENGTH 0x28
#define S2MM_DMACR 0x30
#define S2MM_DMASR 0x34
#define S2MM_DA 0x48
#define S2MM_DA_MSB 0x4C
#define S2MM_LENGTH 0x58
#define DMACR_RS (1u << 0)
#define DMACR_RESET (1u << 2)
#define DMACR_IOC_IRQEN (1u << 12)
#define DMACR_ERR_IRQEN (1u << 14)
#define DMASR_IOC (1u << 12)
#define DMASR_ERR_IRQ (1u << 14)
#define DMASR_DMAINTERR (1u << 4)
#define DMASR_DMASLVERR (1u << 5)
#define DMASR_DMADECERR (1u << 6)
#define DMASR_ANY_ERROR (DMASR_DMAINTERR | DMASR_DMASLVERR | DMASR_DMADECERR | DMASR_ERR_IRQ)

enum { JNULL, JBOOL, JNUM, JSTR, JARR, JOBJ };

struct jnode {
    int type;
    double num;
    char *str;
    char *key;
    struct jnode *first;
    struct jnode *next;
    int count;
};

struct jparse {
    const char *p, *end;
    int bad;
};

static void *xmalloc(size_t n)
{
    void *p = malloc(n ? n : 1);
    if (!p) {
        fprintf(stderr, "bitnet_kria: out of memory (%zu bytes)\n", n);
        exit(2);
    }
    return p;
}

static void jskip(struct jparse *j)
{
    while (j->p < j->end && (*j->p == ' ' || *j->p == '\t' || *j->p == '\n' || *j->p == '\r'))
        j->p++;
}

static char *jstring(struct jparse *j)
{
    if (j->p >= j->end || *j->p != '"') { j->bad = 1; return 0; }
    j->p++;
    const char *s = j->p;
    size_t n = 0;
    while (j->p < j->end && *j->p != '"') {
        if (*j->p == '\\') j->p++;
        j->p++;
        n++;
    }
    if (j->p >= j->end) { j->bad = 1; return 0; }
    char *out = xmalloc(n + 1);
    size_t k = 0;
    for (const char *q = s; q < j->p; q++) {
        if (*q == '\\' && q + 1 < j->p) {
            q++;
            switch (*q) {
            case 'n': out[k++] = '\n'; break;
            case 't': out[k++] = '\t'; break;
            case 'r': out[k++] = '\r'; break;
            case 'b': out[k++] = '\b'; break;
            case 'f': out[k++] = '\f'; break;
            case 'u': out[k++] = '?'; q += 4; break;
            default: out[k++] = *q; break;
            }
        } else {
            out[k++] = *q;
        }
    }
    out[k] = 0;
    j->p++;
    return out;
}

static struct jnode *jvalue(struct jparse *j);

static struct jnode *jnew(int type)
{
    struct jnode *n = xmalloc(sizeof *n);
    memset(n, 0, sizeof *n);
    n->type = type;
    return n;
}

static struct jnode *jvalue(struct jparse *j)
{
    jskip(j);
    if (j->p >= j->end) { j->bad = 1; return 0; }
    char c = *j->p;
    if (c == '{' || c == '[') {
        int obj = c == '{';
        struct jnode *n = jnew(obj ? JOBJ : JARR), *tail = 0;
        j->p++;
        jskip(j);
        if (j->p < j->end && *j->p == (obj ? '}' : ']')) { j->p++; return n; }
        for (;;) {
            jskip(j);
            char *key = 0;
            if (obj) {
                key = jstring(j);
                if (j->bad) return n;
                jskip(j);
                if (j->p >= j->end || *j->p != ':') { j->bad = 1; return n; }
                j->p++;
            }
            struct jnode *v = jvalue(j);
            if (j->bad) return n;
            v->key = key;
            if (tail) tail->next = v; else n->first = v;
            tail = v;
            n->count++;
            jskip(j);
            if (j->p < j->end && *j->p == ',') { j->p++; continue; }
            if (j->p < j->end && *j->p == (obj ? '}' : ']')) { j->p++; return n; }
            j->bad = 1;
            return n;
        }
    }
    if (c == '"') {
        struct jnode *n = jnew(JSTR);
        n->str = jstring(j);
        return n;
    }
    if (!strncmp(j->p, "true", 4)) { j->p += 4; struct jnode *n = jnew(JBOOL); n->num = 1; return n; }
    if (!strncmp(j->p, "false", 5)) { j->p += 5; struct jnode *n = jnew(JBOOL); n->num = 0; return n; }
    if (!strncmp(j->p, "null", 4)) { j->p += 4; return jnew(JNULL); }
    char *endp = 0;
    double v = strtod(j->p, &endp);
    if (endp == j->p) { j->bad = 1; return 0; }
    j->p = endp;
    struct jnode *n = jnew(JNUM);
    n->num = v;
    return n;
}

static struct jnode *jget(const struct jnode *o, const char *key)
{
    if (!o || o->type != JOBJ) return 0;
    for (struct jnode *c = o->first; c; c = c->next)
        if (c->key && !strcmp(c->key, key)) return c;
    return 0;
}

static double jnum(const struct jnode *o, const char *key, double dflt)
{
    struct jnode *c = jget(o, key);
    return c && (c->type == JNUM || c->type == JBOOL) ? c->num : dflt;
}

static const char *jstr(const struct jnode *o, const char *key)
{
    struct jnode *c = jget(o, key);
    return c && c->type == JSTR ? c->str : 0;
}

static struct jnode *jparse_file(const char *path)
{
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "bitnet_kria: %s: %s\n", path, strerror(errno)); exit(2); }
    struct stat st;
    if (fstat(fileno(f), &st)) { perror(path); exit(2); }
    char *text = xmalloc((size_t)st.st_size + 1);
    if (fread(text, 1, (size_t)st.st_size, f) != (size_t)st.st_size) { perror(path); exit(2); }
    text[st.st_size] = 0;
    fclose(f);
    struct jparse j = { text, text + st.st_size, 0 };
    struct jnode *n = jvalue(&j);
    if (j.bad || !n) { fprintf(stderr, "bitnet_kria: %s is not valid JSON\n", path); exit(2); }
    return n;
}

static double now_ms(void)
{
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec * 1e3 + t.tv_nsec / 1e6;
}

static void die(const char *what)
{
    fprintf(stderr, "bitnet_kria: %s: %s\n", what, strerror(errno));
    exit(2);
}

#ifdef POOL_TRACE
#include <sys/syscall.h>
#define PFN 8
struct pfsite {
    const void *fn;
    unsigned calls, yields, slow;
    double self, join, maxjoin, wake, maxwake, work;
};
static struct pfsite PFS[PFN];
static double PF_WAKE[MAXT], PF_FIN[MAXT];
static int PF_TID[MAXT], PF_SIDE_TID;
static long PF_S0[MAXT + 1][3];
static double SCAN_FWD[8192];
static int SCAN_N;
static double SCAN_LAST;

static double PF_BAR[MAXT], PF_LOOP[MAXT], PF_MARK[MAXT], PF_MAXDISP;
static const void *PF_MAXFN;

struct pfslow {
    int idx;
    double scan, wall, maxdisp, when;
    const void *fn;
    long minflt, majflt;
    double run[MAXT + 1], wait[MAXT + 1];
    double busy[8], idle[8], syst[8];
    double bar[MAXT], loop[MAXT];
};
static struct pfslow PF_SLOW[24];
static int PF_NSLOW, PF_FWD;

static void pf_snap(struct pfslow *s);

static struct pfsite *pf_site(const void *fn)
{
    for (int i = 0; i < PFN; i++) {
        if (PFS[i].fn == fn) return &PFS[i];
        if (!PFS[i].fn) { PFS[i].fn = fn; return &PFS[i]; }
    }
    return &PFS[PFN - 1];
}

static void pf_sched(int tid, long *out)
{
    char p[64], buf[128];
    out[0] = out[1] = out[2] = 0;
    if (!tid) return;
    snprintf(p, sizeof p, "/proc/self/task/%d/schedstat", tid);
    int fd = open(p, O_RDONLY);
    if (fd < 0) return;
    int n = (int)read(fd, buf, sizeof buf - 1);
    close(fd);
    if (n <= 0) return;
    buf[n] = 0;
    sscanf(buf, "%ld %ld %ld", &out[0], &out[1], &out[2]);
}

static long PF_C0[8][8];
static int PF_NCPU;

static void pf_cpustat(long out[8][8])
{
    FILE *f = fopen("/proc/stat", "r");
    char line[256];
    if (!f) return;
    while (fgets(line, sizeof line, f)) {
        int c;
        long v[8];
        if (sscanf(line, "cpu%d %ld %ld %ld %ld %ld %ld %ld %ld",
                   &c, &v[0], &v[1], &v[2], &v[3], &v[4], &v[5], &v[6], &v[7]) == 9 && c >= 0 && c < 8) {
            for (int i = 0; i < 8; i++) out[c][i] = v[i];
            if (c + 1 > PF_NCPU) PF_NCPU = c + 1;
        }
    }
    fclose(f);
}

static void pf_flt(long *mn, long *mj)
{
    char buf[512];
    *mn = *mj = 0;
    int fd = open("/proc/self/stat", O_RDONLY);
    if (fd < 0) return;
    int n = (int)read(fd, buf, sizeof buf - 1);
    close(fd);
    if (n <= 0) return;
    buf[n] = 0;
    char *p = strrchr(buf, ')');
    if (!p) return;
    long v[12];
    if (sscanf(p + 2, "%*c %ld %ld %ld %ld %ld %ld %ld %ld %ld %ld %ld %ld",
               &v[0], &v[1], &v[2], &v[3], &v[4], &v[5], &v[6], &v[7], &v[8], &v[9], &v[10], &v[11]) >= 10) {
        *mn = v[6];
        *mj = v[8];
    }
}

static double pf_temp(const char *name)
{
    char p[160], buf[64];
    snprintf(p, sizeof p, "/sys/bus/iio/devices/iio:device0/in_temp%s_raw", name);
    int fd = open(p, O_RDONLY);
    if (fd < 0) return 0;
    int n = (int)read(fd, buf, sizeof buf - 1);
    close(fd);
    if (n <= 0) return 0;
    buf[n] = 0;
    return (atof(buf) - 36058.0) * 7.771514892 / 1000.0;
}
#endif

static inline float bf16_dec(uint16_t v)
{
    union { uint32_t u; float f; } x;
    x.u = (uint32_t)v << 16;
    return x.f;
}

static inline uint16_t bf16_enc(float f)
{
    union { float f; uint32_t u; } x;
    x.f = f;
    uint32_t u = x.u;
    return (uint16_t)((u + 0x7FFFu + ((u >> 16) & 1u)) >> 16);
}

static void barrier(void)
{
#ifdef __aarch64__
    __asm__ __volatile__("dsb sy" ::: "memory");
#else
    __sync_synchronize();
#endif
}

static uint32_t rd(volatile uint32_t *base, unsigned off) { barrier(); uint32_t v = base[off / 4]; barrier(); return v; }
static void wr(volatile uint32_t *base, unsigned off, uint32_t v) { barrier(); base[off / 4] = v; }

static unsigned CACHE_LINE = 64;

static void cache_line_size(void)
{
#ifdef __aarch64__
    uint64_t ctr;
    __asm__ __volatile__("mrs %0, ctr_el0" : "=r"(ctr));
    CACHE_LINE = 4u << (unsigned)((ctr >> 16) & 0xFu);
    if (CACHE_LINE < 16 || CACHE_LINE > 2048) CACHE_LINE = 64;
#endif
}

static void inval_range(const void *p, size_t n)
{
#ifdef __aarch64__
    uintptr_t a = (uintptr_t)p & ~(uintptr_t)(CACHE_LINE - 1);
    uintptr_t end = (uintptr_t)p + n;
    for (; a < end; a += CACHE_LINE)
        __asm__ __volatile__("dc civac, %0" :: "r"(a) : "memory");
    __asm__ __volatile__("dsb sy" ::: "memory");
#else
    (void)p; (void)n;
#endif
}

#define POOL_SPIN 50000
#define POOL_SPIN_SHORT 5000

static inline void cpu_relax(void)
{
#ifdef __aarch64__
    __asm__ __volatile__("yield" ::: "memory");
#else
    __sync_synchronize();
#endif
}

struct pool {
    int n;
    pthread_t th[MAXT];
    pthread_mutex_t mu;
    pthread_cond_t work;
    void (*fn)(void *, int, int);
    void *arg;
    unsigned gen, hot;
    int pending, stop;
};

static struct pool POOL;

#ifdef POOL_TRACE
static void pf_snap(struct pfslow *s)
{
    long a[3], c[8][8];
    pf_flt(&s->minflt, &s->majflt);
    for (int i = 0; i <= POOL.n; i++) {
        pf_sched(i < POOL.n ? PF_TID[i] : PF_SIDE_TID, a);
        s->run[i] = a[0] / 1e6;
        s->wait[i] = a[1] / 1e6;
    }
    memset(c, 0, sizeof c);
    pf_cpustat(c);
    for (int i = 0; i < 8; i++) {
        s->busy[i] = (double)(c[i][0] + c[i][1] + c[i][2] + c[i][5] + c[i][6] + c[i][7]) * 10.0;
        s->idle[i] = (double)(c[i][3] + c[i][4]) * 10.0;
        s->syst[i] = (double)(c[i][2] + c[i][5] + c[i][6]) * 10.0;
    }
    for (int i = 0; i < MAXT; i++) { s->bar[i] = PF_BAR[i]; s->loop[i] = PF_LOOP[i]; }
}
#endif

static void *pool_worker(void *v)
{
    long id = (long)v;
    unsigned seen = 0, hot = 0;
#ifdef POOL_TRACE
    PF_TID[id] = (int)syscall(SYS_gettid);
#endif
    for (;;) {
        int spins = POOL_SPIN_SHORT;
        for (;;) {
            if (__atomic_load_n(&POOL.gen, __ATOMIC_ACQUIRE) != seen || POOL.stop) break;
            unsigned h = __atomic_load_n(&POOL.hot, __ATOMIC_ACQUIRE);
            if (h != hot) { hot = h; spins = POOL_SPIN; }
            if (--spins <= 0) {
                pthread_mutex_lock(&POOL.mu);
                while (POOL.gen == seen && POOL.hot == hot && !POOL.stop)
                    pthread_cond_wait(&POOL.work, &POOL.mu);
                hot = POOL.hot;
                pthread_mutex_unlock(&POOL.mu);
                spins = POOL_SPIN;
                continue;
            }
            cpu_relax();
        }
        if (POOL.stop) return 0;
        seen = __atomic_load_n(&POOL.gen, __ATOMIC_ACQUIRE);
#ifdef POOL_TRACE
        PF_WAKE[id] = now_ms();
#endif
        POOL.fn(POOL.arg, (int)id, POOL.n);
#ifdef POOL_TRACE
        PF_FIN[id] = now_ms();
#endif
        __atomic_sub_fetch(&POOL.pending, 1, __ATOMIC_RELEASE);
    }
}

static void pool_prewake(void)
{
    if (POOL.n <= 1) return;
    pthread_mutex_lock(&POOL.mu);
    POOL.hot++;
    pthread_cond_broadcast(&POOL.work);
    pthread_mutex_unlock(&POOL.mu);
}

static void pool_start(int n)
{
    if (n < 1) n = 1;
    if (n > MAXT) n = MAXT;
    POOL.n = n;
#ifdef POOL_TRACE
    PF_TID[0] = (int)syscall(SYS_gettid);
#endif
    pthread_mutex_init(&POOL.mu, 0);
    pthread_cond_init(&POOL.work, 0);
    for (long i = 1; i < n; i++)
        if (pthread_create(&POOL.th[i], 0, pool_worker, (void *)i)) die("pthread_create");
}

static void parallel_for(void (*fn)(void *, int, int), void *arg)
{
    if (POOL.n <= 1) { fn(arg, 0, 1); return; }
#ifdef POOL_TRACE
    double tA = now_ms();
    unsigned yields = 0;
    for (int i = 1; i < POOL.n; i++) { PF_WAKE[i] = 0; PF_FIN[i] = 0; }
#endif
    POOL.fn = fn;
    POOL.arg = arg;
    __atomic_store_n(&POOL.pending, POOL.n - 1, __ATOMIC_RELAXED);
    pthread_mutex_lock(&POOL.mu);
    __atomic_add_fetch(&POOL.gen, 1, __ATOMIC_RELEASE);
    pthread_cond_broadcast(&POOL.work);
    pthread_mutex_unlock(&POOL.mu);
    fn(arg, 0, POOL.n);
#ifdef POOL_TRACE
    double tB = now_ms();
#endif
    int spins = 8 * POOL_SPIN;
    while (__atomic_load_n(&POOL.pending, __ATOMIC_ACQUIRE) > 0) {
        if (spins-- > 0) cpu_relax();
#ifdef POOL_TRACE
        else { sched_yield(); yields++; }
#else
        else sched_yield();
#endif
    }
#ifdef POOL_TRACE
    double tC = now_ms();
    struct pfsite *s = pf_site((const void *)fn);
    double wk = 0, wo = 0;
    for (int i = 1; i < POOL.n; i++) {
        double w = PF_WAKE[i] - tA, d = PF_FIN[i] - PF_WAKE[i];
        if (w > wk) wk = w;
        if (d > wo) wo = d;
    }
    if (tC - tA > PF_MAXDISP) { PF_MAXDISP = tC - tA; PF_MAXFN = (const void *)fn; }
    s->calls++;
    s->yields += yields;
    s->self += tB - tA;
    s->join += tC - tB;
    s->wake += wk;
    s->work += wo;
    if (tC - tB > s->maxjoin) s->maxjoin = tC - tB;
    if (wk > s->maxwake) s->maxwake = wk;
    if (tC - tA > 0.5) s->slow++;
#endif
}

static void pool_stop(void)
{
    if (POOL.n <= 1) return;
    pthread_mutex_lock(&POOL.mu);
    POOL.stop = 1;
    __atomic_add_fetch(&POOL.gen, 1, __ATOMIC_RELEASE);
    pthread_cond_broadcast(&POOL.work);
    pthread_mutex_unlock(&POOL.mu);
    for (int i = 1; i < POOL.n; i++) pthread_join(POOL.th[i], 0);
}

struct side {
    pthread_t th;
    pthread_mutex_t mu;
    pthread_cond_t post, prog;
    void (*fn)(void);
    unsigned gen;
    int stage, stop, on;
};

static struct side SIDE;

static void side_mark(int s)
{
    if (!SIDE.on) return;
    pthread_mutex_lock(&SIDE.mu);
    SIDE.stage = s;
    pthread_cond_broadcast(&SIDE.prog);
    pthread_mutex_unlock(&SIDE.mu);
}

static void *side_worker(void *v)
{
    unsigned seen = 0;
    (void)v;
#ifdef POOL_TRACE
    PF_SIDE_TID = (int)syscall(SYS_gettid);
#endif
    for (;;) {
        pthread_mutex_lock(&SIDE.mu);
        while (SIDE.gen == seen && !SIDE.stop) pthread_cond_wait(&SIDE.post, &SIDE.mu);
        if (SIDE.stop) { pthread_mutex_unlock(&SIDE.mu); return 0; }
        seen = SIDE.gen;
        void (*fn)(void) = SIDE.fn;
        pthread_mutex_unlock(&SIDE.mu);
        fn();
    }
}

static void side_start(int on)
{
    SIDE.on = on;
    if (!on) return;
    pthread_mutex_init(&SIDE.mu, 0);
    pthread_cond_init(&SIDE.post, 0);
    pthread_cond_init(&SIDE.prog, 0);
    if (pthread_create(&SIDE.th, 0, side_worker, 0)) die("pthread_create (side)");
}

static void side_post(void (*fn)(void))
{
    if (!SIDE.on) { fn(); return; }
    pthread_mutex_lock(&SIDE.mu);
    SIDE.fn = fn;
    SIDE.stage = 0;
    SIDE.gen++;
    pthread_cond_signal(&SIDE.post);
    pthread_mutex_unlock(&SIDE.mu);
}

static void side_wait(int s, double *ms)
{
    if (!SIDE.on) return;
    double t0 = now_ms();
    pthread_mutex_lock(&SIDE.mu);
    while (SIDE.stage < s) pthread_cond_wait(&SIDE.prog, &SIDE.mu);
    pthread_mutex_unlock(&SIDE.mu);
    if (ms) *ms += now_ms() - t0;
}

static void side_stop(void)
{
    if (!SIDE.on) return;
    pthread_mutex_lock(&SIDE.mu);
    SIDE.stop = 1;
    pthread_cond_broadcast(&SIDE.post);
    pthread_mutex_unlock(&SIDE.mu);
    pthread_join(SIDE.th, 0);
}

struct udmabuf {
    const char *dev;
    int fd, cfd, sync_fd;
    uint64_t phys;
    size_t size;
    volatile uint8_t *va;
    const uint8_t *cva;
};

static struct udmabuf B0, B1, B2;
static volatile uint32_t *gpio[E], *dmar[E + 1], *ggpio_a, *ggpio_b;
static uint64_t gpio_addr[E], dma_addr[E + 1], ggpio_a_addr, ggpio_b_addr;
static const char *dma_name[E + 1] = { "dma0", "dma1", "dma2", "dma3", "glue dma" };
static double TIMEOUT_MS = 4000.0;

static uint64_t sysfs_u64(const char *dev, const char *attr)
{
    char path[256], text[64];
    const char *name = strrchr(dev, '/');
    name = name ? name + 1 : dev;
    snprintf(path, sizeof path, "/sys/class/u-dma-buf/%s/%s", name, attr);
    FILE *f = fopen(path, "r");
    if (!f) die(path);
    if (!fgets(text, sizeof text, f)) die(path);
    fclose(f);
    return strtoull(text, 0, 0);
}

static void sysfs_write(const char *dev, const char *attr, const char *value)
{
    char path[256];
    const char *name = strrchr(dev, '/');
    name = name ? name + 1 : dev;
    snprintf(path, sizeof path, "/sys/class/u-dma-buf/%s/%s", name, attr);
    FILE *f = fopen(path, "w");
    if (!f) die(path);
    fputs(value, f);
    fclose(f);
}

static void open_udmabuf(struct udmabuf *b, const char *dev)
{
    b->dev = dev;
    b->phys = sysfs_u64(dev, "phys_addr");
    b->size = (size_t)sysfs_u64(dev, "size");
    sysfs_write(dev, "sync_mode", "2");
    b->fd = open(dev, O_RDWR | O_SYNC);
    if (b->fd < 0) die(dev);
    void *m = mmap(0, b->size, PROT_READ | PROT_WRITE, MAP_SHARED, b->fd, 0);
    if (m == MAP_FAILED) die(dev);
    b->va = (volatile uint8_t *)m;
    b->cfd = open(dev, O_RDONLY);
    if (b->cfd < 0) die(dev);
    m = mmap(0, b->size, PROT_READ, MAP_SHARED, b->cfd, 0);
    if (m == MAP_FAILED) die(dev);
    b->cva = (const uint8_t *)m;
    char path[256];
    const char *name = strrchr(dev, '/');
    name = name ? name + 1 : dev;
    snprintf(path, sizeof path, "/sys/class/u-dma-buf/%s/sync_for_cpu", name);
    b->sync_fd = open(path, O_WRONLY);
    if (b->sync_fd < 0) die(path);
}

static void sums_in(const struct udmabuf *b, size_t off, size_t n)
{
    inval_range(b->cva + off, n);
}

static void sync_for_cpu(const struct udmabuf *b, size_t off, size_t n)
{
    char text[32];
    uint64_t cmd = (uint64_t)off << 32 | (uint64_t)((n + 15u) & 0xFFFFFFF0u) | 2u << 2 | 1u;
    int len = snprintf(text, sizeof text, "%" PRIu64, cmd);
    if (pwrite(b->sync_fd, text, (size_t)len, 0) != len) die("sync_for_cpu");
}

static void copy_out(void *dst, const struct udmabuf *b, size_t off, size_t n)
{
    sync_for_cpu(b, off, n);
    memcpy(dst, b->cva + off, n);
}

static void copy_in(volatile uint8_t *dst, const void *src, size_t n)
{
    volatile uint64_t *d = (volatile uint64_t *)dst;
    const uint64_t *s = (const uint64_t *)src;
    size_t i, w = n / 8;
    for (i = 0; i < w; i++) d[i] = s[i];
    for (i = w * 8; i < n; i++) ((volatile uint8_t *)dst)[i] = ((const uint8_t *)src)[i];
    barrier();
}

static void *map_phys(int memfd, uint64_t base, size_t len)
{
    void *m = mmap(0, len, PROT_READ | PROT_WRITE, MAP_SHARED, memfd, (off_t)base);
    if (m == MAP_FAILED) die("mmap /dev/mem");
    return m;
}

static void print_dma_status(void)
{
    for (unsigned i = 0; i <= E; i++) {
        uint32_t m = rd(dmar[i], MM2S_DMASR), s = rd(dmar[i], S2MM_DMASR);
        fprintf(stderr, "    %-8s: MM2S_DMASR 0x%08x  S2MM_DMASR 0x%08x", dma_name[i], m, s);
        if (i < E) {
            uint32_t st = rd(gpio[i], GPIO2_DATA);
            fprintf(stderr, "   status 0x%08x (done %u act %u state %u idle %u)\n",
                    st, STATUS_DONE(st), STATUS_ACT(st), STATUS_STATE(st), !!(st & STATUS_IDLE));
        } else {
            uint32_t st = rd(ggpio_a, GPIO2_DATA);
            fprintf(stderr, "   glue status 0x%08x (hmax %u done %u idle %u)\n",
                    st, GSTATUS_HMAX(st), GSTATUS_DONE(st), !!(st & GSTATUS_IDLE));
        }
    }
}

static double CHUNK_WORK_MS;

static int wait_ioc(unsigned i, unsigned sr_off, const char *what)
{
    double t0 = now_ms();
    for (;;) {
        uint32_t sr = rd(dmar[i], sr_off);
        if (sr & DMASR_ANY_ERROR) { fprintf(stderr, "  %s %s: DMA error, DMASR 0x%08x\n", dma_name[i], what, sr); return -1; }
        if (sr & DMASR_IOC) return 0;
        if (now_ms() - t0 > TIMEOUT_MS) { fprintf(stderr, "  %s %s: timeout, DMASR 0x%08x\n", dma_name[i], what, sr); return -2; }
    }
}

static uint32_t mm2s_msb[E + 1], s2mm_msb[E + 1];

static int dma_reset(unsigned i)
{
    wr(dmar[i], MM2S_DMACR, DMACR_RESET);
    double t0 = now_ms();
    while (rd(dmar[i], MM2S_DMACR) & DMACR_RESET)
        if (now_ms() - t0 > TIMEOUT_MS) { fprintf(stderr, "  %s: reset did not complete\n", dma_name[i]); return -2; }
    wr(dmar[i], MM2S_DMASR, DMASR_IOC | DMASR_ERR_IRQ);
    wr(dmar[i], S2MM_DMASR, DMASR_IOC | DMASR_ERR_IRQ);
    wr(dmar[i], S2MM_DMACR, DMACR_RS | DMACR_IOC_IRQEN | DMACR_ERR_IRQEN);
    wr(dmar[i], MM2S_DMACR, DMACR_RS | DMACR_IOC_IRQEN | DMACR_ERR_IRQEN);
    mm2s_msb[i] = s2mm_msb[i] = 0;
    return 0;
}

static void mm2s_start(unsigned i, uint64_t phys, size_t len)
{
    uint32_t hi = (uint32_t)(phys >> 32);
    wr(dmar[i], MM2S_DMASR, DMASR_IOC);
    wr(dmar[i], MM2S_SA, (uint32_t)phys);
    if (hi != mm2s_msb[i]) { wr(dmar[i], MM2S_SA_MSB, hi); mm2s_msb[i] = hi; }
    wr(dmar[i], MM2S_LENGTH, (uint32_t)len);
}

static void s2mm_start(unsigned i, uint64_t phys, size_t len)
{
    uint32_t hi = (uint32_t)(phys >> 32);
    wr(dmar[i], S2MM_DMASR, DMASR_IOC);
    wr(dmar[i], S2MM_DA, (uint32_t)phys);
    if (hi != s2mm_msb[i]) { wr(dmar[i], S2MM_DA_MSB, hi); s2mm_msb[i] = hi; }
    wr(dmar[i], S2MM_LENGTH, (uint32_t)len);
}

static int wait_done(unsigned i, unsigned neurons, uint32_t *last)
{
    double t0 = now_ms();
    for (;;) {
        uint32_t s = rd(gpio[i], GPIO2_DATA);
        *last = s;
        if (STATUS_DONE(s) == neurons && (s & STATUS_IDLE)) return 0;
        if (now_ms() - t0 > TIMEOUT_MS) return -2;
    }
}

static int wait_loaded(unsigned i, unsigned beats, uint32_t *last)
{
    double t0 = now_ms();
    for (;;) {
        uint32_t s = rd(gpio[i], GPIO2_DATA);
        *last = s;
        if (STATUS_ACT(s) == beats && (s & STATUS_IDLE)) return 0;
        if (now_ms() - t0 > TIMEOUT_MS) return -2;
    }
}

static int wait_glue(unsigned n, uint32_t *last)
{
    double t0 = now_ms();
    for (;;) {
        uint32_t s = rd(ggpio_a, GPIO2_DATA);
        *last = s;
        if (GSTATUS_DONE(s) == n && (s & GSTATUS_IDLE)) return 0;
        if (now_ms() - t0 > TIMEOUT_MS) return -2;
    }
}

static int engine_phase(const char *what, unsigned B, unsigned batch, unsigned npe, uint64_t act_phys,
                        size_t act_bytes, const uint64_t *w_phys, const uint64_t *r_phys,
                        double *ms_act, double *ms_w,
                        int prewake, unsigned chunks, void (*on_chunk)(void *, unsigned, unsigned), void *carg)
{
    const unsigned CH = chunks < 1 || npe % chunks ? 1u : chunks;
    const unsigned cr = npe / CH;
    const size_t cwbytes = (size_t)cr * B * BEAT_BYTES, crbytes = (size_t)cr * batch * 4u;
    const unsigned act_beats = (unsigned)(batch * act_bytes / BEAT_BYTES);
    const uint32_t bits = CTRL_BEATS(B) | CTRL_BATCH(batch);
    const uint32_t ctrl_act = CTRL_NEURONS(npe) | bits | CTRL_RUN;
    const uint32_t ctrl_w = CTRL_NEURONS(cr) | bits | CTRL_MODE_WEIGHTS | CTRL_RUN;
    unsigned i, c;
    double ta = now_ms();
    CHUNK_WORK_MS = 0;

    for (i = 0; i < E; i++) s2mm_start(i, r_phys[i], crbytes);
    for (i = 0; i < E; i++) {
        wr(gpio[i], GPIO_DATA, ctrl_act);
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
    for (i = 0; i < E; i++) wr(gpio[i], GPIO_DATA, ctrl_w);
    for (i = 0; i < E; i++) mm2s_start(i, w_phys[i], cwbytes);
    for (c = 0; c < CH; c++) {
        if (prewake) {
            unsigned cpn = B * ((batch + 1u) / 2u);
            unsigned left = 50000u / (cpn ? cpn : 1u), thresh = cr > left ? cr - left : 0;
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
            if (wait_done(i, cr, &s)) {
                fprintf(stderr, "  %s: engine %u status 0x%08x (done %u) never reached %u | idle\n", what, i, s, STATUS_DONE(s), cr);
                print_dma_status();
                return 3;
            }
        }
        for (i = 0; i < E; i++) wr(gpio[i], GPIO_DATA, 0);
        if (c + 1 < CH) {
            for (i = 0; i < E; i++) s2mm_start(i, r_phys[i] + (uint64_t)(c + 1) * crbytes, crbytes);
            for (i = 0; i < E; i++) wr(gpio[i], GPIO_DATA, ctrl_w);
            for (i = 0; i < E; i++) mm2s_start(i, w_phys[i] + (uint64_t)(c + 1) * cwbytes, cwbytes);
        }
        if (on_chunk) on_chunk(carg, c, CH);
    }
    double t1 = now_ms();
    *ms_act += t0 - ta;
    *ms_w += t1 - t0 - CHUNK_WORK_MS;
    return 0;
}

static int glue_pass(const char *what, unsigned mode_b, unsigned n, uint32_t params, uint32_t m,
                     uint64_t in_phys, size_t in_bytes, uint64_t out_phys, size_t out_bytes,
                     double *ms, uint32_t *status_out)
{
    uint32_t ctrl = GCTRL_N(n) | (mode_b ? GCTRL_MODE_B : 0) | GCTRL_RUN, s;
    wr(ggpio_b, GPIO_DATA, params);
    wr(ggpio_b, GPIO2_DATA, m);
    wr(ggpio_a, GPIO_DATA, 0);
    double t0 = now_ms();
    s2mm_start(GDMA, out_phys, out_bytes);
    wr(ggpio_a, GPIO_DATA, ctrl);
    mm2s_start(GDMA, in_phys, in_bytes);
    if (wait_ioc(GDMA, MM2S_DMASR, what)) { print_dma_status(); return 3; }
    if (wait_ioc(GDMA, S2MM_DMASR, what)) { print_dma_status(); return 3; }
    if (wait_glue(n, &s)) {
        fprintf(stderr, "  %s: glue status 0x%08x (hmax %u done %u) never reached %u | idle\n", what, s, GSTATUS_HMAX(s), GSTATUS_DONE(s), n);
        print_dma_status();
        return 3;
    }
    *ms += now_ms() - t0;
    *status_out = s;
    wr(ggpio_a, GPIO_DATA, 0);
    return 0;
}

static const char *PROJ_NAME[NPROJ] = { "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj" };

struct mat {
    uint64_t offset;
    unsigned n, k, beats;
    double ws;
};

struct model {
    int hidden, inter, layers, n_q, n_kv, head_dim, kv_size, vocab, groups;
    double theta, eps;
    struct mat mat[64][NPROJ];
    const float *in_norm[64], *post_norm[64], *attn_sub[64], *ffn_sub[64], *final_norm;
    int16_t *G[64];
    uint32_t *Gabs[64];
    int kg[64];
    const uint16_t *embed;
    const int8_t *head_w;
    const float *head_scale;
    const float *head_t_scale;
    unsigned head_t_beats, head_t_npe, head_rows;
    size_t model_bytes, head_t_bytes;
    unsigned wpb, wpby;
    int silu;
    char enc[16], model_file[64], head_file[64];
};

static struct model M;

static const void *map_ro(const char *dir, const char *name, size_t want, size_t *got)
{
    char path[512];
    snprintf(path, sizeof path, "%s/%s", dir, name);
    int fd = open(path, O_RDONLY);
    if (fd < 0) die(path);
    struct stat st;
    if (fstat(fd, &st)) die(path);
    if (want && (size_t)st.st_size != want) {
        fprintf(stderr, "bitnet_kria: %s is %zu bytes, wanted %zu\n", path, (size_t)st.st_size, want);
        exit(2);
    }
    void *m = mmap(0, (size_t)st.st_size, PROT_READ, MAP_SHARED, fd, 0);
    if (m == MAP_FAILED) die(path);
    close(fd);
    if (got) *got = (size_t)st.st_size;
    return m;
}

static void gamma_fixed(const float *gamma, unsigned n, int *kg_out, int16_t *G)
{
    double amax = 0;
    for (unsigned i = 0; i < n; i++) {
        double a = fabs((double)gamma[i]);
        if (a > amax) amax = a;
    }
    int kg = 0;
    for (int k = 0; k < 16; k++)
        if (rint(amax * ldexp(1.0, k)) <= 32767.0) kg = k;
    for (unsigned i = 0; i < n; i++) {
        double v = rint((double)gamma[i] * ldexp(1.0, kg));
        G[i] = (int16_t)v;
    }
    *kg_out = kg;
}

static void rmsnorm(float *out, const float *x, const float *w, int n, double eps)
{
    double acc = 0;
    int i = 0;
#ifdef __aarch64__
    float64x2_t s0 = vdupq_n_f64(0), s1 = vdupq_n_f64(0), s2 = vdupq_n_f64(0), s3 = vdupq_n_f64(0);
    for (; i + 8 <= n; i += 8) {
        float32x4_t x0 = vld1q_f32(x + i), x1 = vld1q_f32(x + i + 4);
        float32x4_t p0 = vmulq_f32(x0, x0), p1 = vmulq_f32(x1, x1);
        s0 = vaddq_f64(s0, vcvt_f64_f32(vget_low_f32(p0)));
        s1 = vaddq_f64(s1, vcvt_f64_f32(vget_high_f32(p0)));
        s2 = vaddq_f64(s2, vcvt_f64_f32(vget_low_f32(p1)));
        s3 = vaddq_f64(s3, vcvt_f64_f32(vget_high_f32(p1)));
    }
    acc = (vaddvq_f64(s0) + vaddvq_f64(s1)) + (vaddvq_f64(s2) + vaddvq_f64(s3));
#endif
    for (; i < n; i++) {
        float s = x[i] * x[i];
        acc += (double)s;
    }
    float v = (float)(acc / (double)n);
    float inv = 1.0f / sqrtf(v + (float)eps);
    int j = 0;
#ifdef __aarch64__
    const float32x4_t iv = vdupq_n_f32(inv);
    for (; j + 8 <= n; j += 8) {
        vst1q_f32(out + j, vmulq_f32(vmulq_f32(vld1q_f32(x + j), iv), vld1q_f32(w + j)));
        vst1q_f32(out + j + 4, vmulq_f32(vmulq_f32(vld1q_f32(x + j + 4), iv), vld1q_f32(w + j + 4)));
    }
#endif
    for (; j < n; j++) out[j] = (x[j] * inv) * w[j];
}

static float absmax_int8(int8_t *q, const float *x, int n)
{
    float amax = 0;
    int i = 0;
#ifdef __aarch64__
    float32x4_t m0 = vdupq_n_f32(0), m1 = vdupq_n_f32(0);
    for (; i + 8 <= n; i += 8) {
        m0 = vmaxq_f32(m0, vabsq_f32(vld1q_f32(x + i)));
        m1 = vmaxq_f32(m1, vabsq_f32(vld1q_f32(x + i + 4)));
    }
    amax = vmaxvq_f32(vmaxq_f32(m0, m1));
#endif
    for (; i < n; i++) {
        float a = fabsf(x[i]);
        if (a > amax) amax = a;
    }
    if (!(amax > 1e-5f)) amax = 1e-5f;
    float scale = 127.0f / amax;
    int j = 0;
#ifdef __aarch64__
    const float32x4_t sv = vdupq_n_f32(scale), hi = vdupq_n_f32(127.0f), lo = vdupq_n_f32(-128.0f);
    for (; j + 16 <= n; j += 16) {
        int32x4_t c0 = vcvtq_s32_f32(vmaxq_f32(vminq_f32(vrndnq_f32(vmulq_f32(vld1q_f32(x + j), sv)), hi), lo));
        int32x4_t c1 = vcvtq_s32_f32(vmaxq_f32(vminq_f32(vrndnq_f32(vmulq_f32(vld1q_f32(x + j + 4), sv)), hi), lo));
        int32x4_t c2 = vcvtq_s32_f32(vmaxq_f32(vminq_f32(vrndnq_f32(vmulq_f32(vld1q_f32(x + j + 8), sv)), hi), lo));
        int32x4_t c3 = vcvtq_s32_f32(vmaxq_f32(vminq_f32(vrndnq_f32(vmulq_f32(vld1q_f32(x + j + 12), sv)), hi), lo));
        int16x8_t h0 = vcombine_s16(vmovn_s32(c0), vmovn_s32(c1));
        int16x8_t h1 = vcombine_s16(vmovn_s32(c2), vmovn_s32(c3));
        vst1q_s8(q + j, vcombine_s8(vmovn_s16(h0), vmovn_s16(h1)));
    }
#endif
    for (; j < n; j++) {
        float v = rintf(x[j] * scale);
        if (v > 127.0f) v = 127.0f;
        if (v < -128.0f) v = -128.0f;
        q[j] = (int8_t)v;
    }
    return scale;
}

static void scale_i32(float *out, const int32_t *in, int n, float sc)
{
    int i = 0;
#ifdef __aarch64__
    const float32x4_t sv = vdupq_n_f32(sc);
    for (; i + 8 <= n; i += 8) {
        vst1q_f32(out + i, vmulq_f32(vcvtq_f32_s32(vld1q_s32(in + i)), sv));
        vst1q_f32(out + i + 4, vmulq_f32(vcvtq_f32_s32(vld1q_s32(in + i + 4)), sv));
    }
#endif
    for (; i < n; i++) out[i] = (float)in[i] * sc;
}

static void add_scaled_i32(float *h, const int32_t *in, int n, float sc)
{
    int i = 0;
#ifdef __aarch64__
    const float32x4_t sv = vdupq_n_f32(sc);
    for (; i + 8 <= n; i += 8) {
        vst1q_f32(h + i, vaddq_f32(vld1q_f32(h + i), vmulq_f32(vcvtq_f32_s32(vld1q_s32(in + i)), sv)));
        vst1q_f32(h + i + 4, vaddq_f32(vld1q_f32(h + i + 4),
                                       vmulq_f32(vcvtq_f32_s32(vld1q_s32(in + i + 4)), sv)));
    }
#endif
    for (; i < n; i++) h[i] = h[i] + (float)in[i] * sc;
}

static void rope_heads(float *p, int nheads, int hd, const float *c, const float *s)
{
    const int half = hd / 2;
    for (int h = 0; h < nheads; h++) {
        float *r = p + (size_t)h * hd;
        int d = 0;
#ifdef __aarch64__
        for (; d + 4 <= half; d += 4) {
            float32x4_t a = vld1q_f32(r + d), b = vld1q_f32(r + d + half);
            float32x4_t cv = vld1q_f32(c + d), sv = vld1q_f32(s + d);
            vst1q_f32(r + d, vsubq_f32(vmulq_f32(a, cv), vmulq_f32(b, sv)));
            vst1q_f32(r + d + half, vaddq_f32(vmulq_f32(b, cv), vmulq_f32(a, sv)));
        }
#endif
        for (; d < half; d++) {
            float a = r[d], b = r[d + half], cc = c[d], ss = s[d];
            r[d] = a * cc + (-b) * ss;
            r[d + half] = b * cc + a * ss;
        }
    }
}

struct timings {
    double eng_qkv, eng_o, eng_gu, eng_down, act, glue_a, glue_b, scan, copyout, attn, norm, head, total;
    double sidewait, sidebusy;
    double head_s1, head_act, head_top, head_s2;
    int tokens, heads;
};

#define CACHE_F32 0
#define CACHE_BF16 1
#define CACHE_I8 2
#define CACHE_FX 3
#define CACHE_FAB 4
#define CACHE_K8 5
#define CACHE_V8 6
#define FAB_KVMAX 16
#define FAB_NQMAX 20
#define FAB_FDMAX 128

struct fabshape {
    int kv, nq, fd, qpk, rows, scb, cells, qfb, topb;
    unsigned block, header, off_top, off_qq;
    int outa, outb;
};

static struct fabshape FAB;

static void fab_shape(int kv, int nq, int fd, int qpk)
{
    FAB.kv = kv; FAB.nq = nq; FAB.fd = fd; FAB.qpk = qpk;
    FAB.rows = fd / 16;
    FAB.scb = (kv + 7) / 8;
    FAB.cells = FAB.rows * kv;
    FAB.qfb = (nq * 16 + 127) / 128;
    FAB.topb = (nq * 24 + 127) / 128;
    FAB.block = (unsigned)(2 * FAB.scb + 2 * FAB.cells) * 16u;
    FAB.header = (unsigned)(1 + FAB.qfb + FAB.topb + nq * FAB.rows) * 16u;
    FAB.off_top = (unsigned)(1 + FAB.qfb) * 16u;
    FAB.off_qq = (unsigned)(1 + FAB.qfb + FAB.topb) * 16u;
    FAB.outa = nq;
    FAB.outb = nq + 2 * nq * fd;
}

#define FAB_KV FAB.kv
#define FAB_NQ FAB.nq
#define FAB_FD FAB.fd
#define FAB_QPK FAB.qpk
#define FAB_BLOCK FAB.block
#define FAB_HEADER FAB.header
#define OFF_ATT_HDR 0x200000u
#define OFF_ATT_RES 0x280000u
#define ATT_STRIDE 0x8000u
#define ATT_SEL (1u << 15)
static unsigned ATT_PORTS;
static size_t FAB_BASE_OFF;
static int GEN_FILL;

struct run {
    int ctx;
    int cache_dtype, k_i8, v_i8;
    float *cache_k, *cache_v;
    size_t ldk, kt_tile, kt_head, kt_layer, v_head, v_layer;
    uint16_t *cache_kb, *cache_vb;
    int8_t *cache_k8, *cache_v8;
    float *cache_ks, *cache_vs;
    int pos;
    int slot_pos[BMAX], slot_seq[BMAX];
    int cur_seq, nseq;
    size_t fab_seq;
    int fab_groups, fab_kv;
    float *hb;
    float *finb;
    int32_t *sqkv, *soi, *sgu, *sd;
    float *h, *x, *xa, *attn, *qf, *kf, *vf, *fin, *ffn;
    int8_t *q8_in;
    int32_t *gu, *hglue;
    int8_t *q8;
    uint32_t *scratch;
    float *logits;
    const int8_t *hw;
    const float *hs;
    int8_t *hq;
    float hinv;
    int head_fabric, head_k;
    unsigned head_chunks, gu_chunks;
    const float *hts;
    float *hcand;
    int *hcandi;
    int *hshort;
    int cur_layer, cur_T;
    const float *cur_q;
    float *cur_out;
    float *scores;
    double *inv_freq;
    float *rope_c, *rope_s;
    float *rope_cb, *rope_sb;
    struct timings t;
};

static struct run R;

#define ATT_TB 32
#define ATT_MAXG 16

static int ATT_W = ATT_TB;
static int ATT_LO;

#define ATT_INLINE static inline __attribute__((always_inline))

ATT_INLINE void scores_kt16_h(float *sc, const float *kt, const float *q, const int hd, float scaling, const int w)
{
#ifdef __aarch64__
    float32x4_t a0 = vdupq_n_f32(0), a1 = vdupq_n_f32(0), a2 = vdupq_n_f32(0), a3 = vdupq_n_f32(0);
    for (int d = 0; d < hd; d++) {
        const float32x4_t qd = vdupq_n_f32(q[d]);
        const float *k = kt + (size_t)d * (size_t)w;
        a0 = vaddq_f32(a0, vmulq_f32(vld1q_f32(k), qd));
        a1 = vaddq_f32(a1, vmulq_f32(vld1q_f32(k + 4), qd));
        a2 = vaddq_f32(a2, vmulq_f32(vld1q_f32(k + 8), qd));
        a3 = vaddq_f32(a3, vmulq_f32(vld1q_f32(k + 12), qd));
    }
    const float32x4_t sv = vdupq_n_f32(scaling);
    vst1q_f32(sc, vmulq_f32(a0, sv));
    vst1q_f32(sc + 4, vmulq_f32(a1, sv));
    vst1q_f32(sc + 8, vmulq_f32(a2, sv));
    vst1q_f32(sc + 12, vmulq_f32(a3, sv));
#else
    for (int j = 0; j < 16; j++) {
        float s = 0;
        for (int d = 0; d < hd; d++) s += kt[(size_t)d * (size_t)w + j] * q[d];
        sc[j] = s * scaling;
    }
#endif
}

static void scores_kt16(float *sc, const float *kt, const float *q, int hd, float scaling, const int w)
{
    if (w == 16) {
        if (hd == 128) scores_kt16_h(sc, kt, q, 128, scaling, 16);
        else scores_kt16_h(sc, kt, q, hd, scaling, 16);
    } else {
        if (hd == 128) scores_kt16_h(sc, kt, q, 128, scaling, 32);
        else scores_kt16_h(sc, kt, q, hd, scaling, 32);
    }
}

static void scores_kt8(float *sc, const float *kt, const float *q, int hd, float scaling, const int w)
{
#ifdef __aarch64__
    float32x4_t a0 = vdupq_n_f32(0), a1 = vdupq_n_f32(0);
    for (int d = 0; d < hd; d++) {
        const float32x4_t qd = vdupq_n_f32(q[d]);
        const float *k = kt + (size_t)d * (size_t)w;
        a0 = vaddq_f32(a0, vmulq_f32(vld1q_f32(k), qd));
        a1 = vaddq_f32(a1, vmulq_f32(vld1q_f32(k + 4), qd));
    }
    const float32x4_t sv = vdupq_n_f32(scaling);
    vst1q_f32(sc, vmulq_f32(a0, sv));
    vst1q_f32(sc + 4, vmulq_f32(a1, sv));
#else
    for (int j = 0; j < 8; j++) {
        float s = 0;
        for (int d = 0; d < hd; d++) s += kt[(size_t)d * (size_t)w + j] * q[d];
        sc[j] = s * scaling;
    }
#endif
}

static void scores_kt4(float *sc, const float *kt, const float *q, int hd, float scaling, const int w)
{
#ifdef __aarch64__
    float32x4_t a0 = vdupq_n_f32(0);
    for (int d = 0; d < hd; d++)
        a0 = vaddq_f32(a0, vmulq_f32(vld1q_f32(kt + (size_t)d * (size_t)w), vdupq_n_f32(q[d])));
    vst1q_f32(sc, vmulq_f32(a0, vdupq_n_f32(scaling)));
#else
    for (int j = 0; j < 4; j++) {
        float s = 0;
        for (int d = 0; d < hd; d++) s += kt[(size_t)d * (size_t)w + j] * q[d];
        sc[j] = s * scaling;
    }
#endif
}

ATT_INLINE void scores_kt16x4_h(float *s0, float *s1, float *s2, float *s3, const float *kt,
                                const float *q0, const float *q1, const float *q2, const float *q3,
                                const int hd, float scaling)
{
#ifdef __aarch64__
    float32x4_t b00 = vdupq_n_f32(0), b01 = vdupq_n_f32(0), b02 = vdupq_n_f32(0), b03 = vdupq_n_f32(0);
    float32x4_t b10 = vdupq_n_f32(0), b11 = vdupq_n_f32(0), b12 = vdupq_n_f32(0), b13 = vdupq_n_f32(0);
    float32x4_t b20 = vdupq_n_f32(0), b21 = vdupq_n_f32(0), b22 = vdupq_n_f32(0), b23 = vdupq_n_f32(0);
    float32x4_t b30 = vdupq_n_f32(0), b31 = vdupq_n_f32(0), b32 = vdupq_n_f32(0), b33 = vdupq_n_f32(0);
    for (int d = 0; d < hd; d++) {
        const float *k = kt + (size_t)d * ATT_TB;
        float32x4_t k0 = vld1q_f32(k), k1 = vld1q_f32(k + 4), k2 = vld1q_f32(k + 8), k3 = vld1q_f32(k + 12);
        float32x4_t qv = vdupq_n_f32(q0[d]);
        b00 = vaddq_f32(b00, vmulq_f32(k0, qv)); b01 = vaddq_f32(b01, vmulq_f32(k1, qv));
        b02 = vaddq_f32(b02, vmulq_f32(k2, qv)); b03 = vaddq_f32(b03, vmulq_f32(k3, qv));
        qv = vdupq_n_f32(q1[d]);
        b10 = vaddq_f32(b10, vmulq_f32(k0, qv)); b11 = vaddq_f32(b11, vmulq_f32(k1, qv));
        b12 = vaddq_f32(b12, vmulq_f32(k2, qv)); b13 = vaddq_f32(b13, vmulq_f32(k3, qv));
        qv = vdupq_n_f32(q2[d]);
        b20 = vaddq_f32(b20, vmulq_f32(k0, qv)); b21 = vaddq_f32(b21, vmulq_f32(k1, qv));
        b22 = vaddq_f32(b22, vmulq_f32(k2, qv)); b23 = vaddq_f32(b23, vmulq_f32(k3, qv));
        qv = vdupq_n_f32(q3[d]);
        b30 = vaddq_f32(b30, vmulq_f32(k0, qv)); b31 = vaddq_f32(b31, vmulq_f32(k1, qv));
        b32 = vaddq_f32(b32, vmulq_f32(k2, qv)); b33 = vaddq_f32(b33, vmulq_f32(k3, qv));
    }
    const float32x4_t sv = vdupq_n_f32(scaling);
    vst1q_f32(s0, vmulq_f32(b00, sv)); vst1q_f32(s0 + 4, vmulq_f32(b01, sv));
    vst1q_f32(s0 + 8, vmulq_f32(b02, sv)); vst1q_f32(s0 + 12, vmulq_f32(b03, sv));
    vst1q_f32(s1, vmulq_f32(b10, sv)); vst1q_f32(s1 + 4, vmulq_f32(b11, sv));
    vst1q_f32(s1 + 8, vmulq_f32(b12, sv)); vst1q_f32(s1 + 12, vmulq_f32(b13, sv));
    vst1q_f32(s2, vmulq_f32(b20, sv)); vst1q_f32(s2 + 4, vmulq_f32(b21, sv));
    vst1q_f32(s2 + 8, vmulq_f32(b22, sv)); vst1q_f32(s2 + 12, vmulq_f32(b23, sv));
    vst1q_f32(s3, vmulq_f32(b30, sv)); vst1q_f32(s3 + 4, vmulq_f32(b31, sv));
    vst1q_f32(s3 + 8, vmulq_f32(b32, sv)); vst1q_f32(s3 + 12, vmulq_f32(b33, sv));
#else
    scores_kt16_h(s0, kt, q0, hd, scaling, ATT_TB);
    scores_kt16_h(s1, kt, q1, hd, scaling, ATT_TB);
    scores_kt16_h(s2, kt, q2, hd, scaling, ATT_TB);
    scores_kt16_h(s3, kt, q3, hd, scaling, ATT_TB);
#endif
}

static void scores_kt16x4(float *s0, float *s1, float *s2, float *s3, const float *kt,
                          const float *q0, const float *q1, const float *q2, const float *q3,
                          int hd, float scaling)
{
    if (hd == 128) scores_kt16x4_h(s0, s1, s2, s3, kt, q0, q1, q2, q3, 128, scaling);
    else scores_kt16x4_h(s0, s1, s2, s3, kt, q0, q1, q2, q3, hd, scaling);
}

ATT_INLINE void scores_kt16x2_h(float *s0, float *s1, const float *kt,
                                const float *q0, const float *q1, const int hd, float scaling)
{
#ifdef __aarch64__
    float32x4_t b00 = vdupq_n_f32(0), b01 = vdupq_n_f32(0), b02 = vdupq_n_f32(0), b03 = vdupq_n_f32(0);
    float32x4_t b10 = vdupq_n_f32(0), b11 = vdupq_n_f32(0), b12 = vdupq_n_f32(0), b13 = vdupq_n_f32(0);
    for (int d = 0; d < hd; d++) {
        const float *k = kt + (size_t)d * ATT_TB;
        float32x4_t k0 = vld1q_f32(k), k1 = vld1q_f32(k + 4), k2 = vld1q_f32(k + 8), k3 = vld1q_f32(k + 12);
        float32x4_t qv = vdupq_n_f32(q0[d]);
        b00 = vaddq_f32(b00, vmulq_f32(k0, qv)); b01 = vaddq_f32(b01, vmulq_f32(k1, qv));
        b02 = vaddq_f32(b02, vmulq_f32(k2, qv)); b03 = vaddq_f32(b03, vmulq_f32(k3, qv));
        qv = vdupq_n_f32(q1[d]);
        b10 = vaddq_f32(b10, vmulq_f32(k0, qv)); b11 = vaddq_f32(b11, vmulq_f32(k1, qv));
        b12 = vaddq_f32(b12, vmulq_f32(k2, qv)); b13 = vaddq_f32(b13, vmulq_f32(k3, qv));
    }
    const float32x4_t sv = vdupq_n_f32(scaling);
    vst1q_f32(s0, vmulq_f32(b00, sv)); vst1q_f32(s0 + 4, vmulq_f32(b01, sv));
    vst1q_f32(s0 + 8, vmulq_f32(b02, sv)); vst1q_f32(s0 + 12, vmulq_f32(b03, sv));
    vst1q_f32(s1, vmulq_f32(b10, sv)); vst1q_f32(s1 + 4, vmulq_f32(b11, sv));
    vst1q_f32(s1 + 8, vmulq_f32(b12, sv)); vst1q_f32(s1 + 12, vmulq_f32(b13, sv));
#else
    scores_kt16_h(s0, kt, q0, hd, scaling, ATT_TB);
    scores_kt16_h(s1, kt, q1, hd, scaling, ATT_TB);
#endif
}

static void scores_kt16x2(float *s0, float *s1, const float *kt,
                          const float *q0, const float *q1, int hd, float scaling)
{
    if (hd == 128) scores_kt16x2_h(s0, s1, kt, q0, q1, 128, scaling);
    else scores_kt16x2_h(s0, s1, kt, q0, q1, hd, scaling);
}

static void scores_kt_tail(float *sc, const float *kt, const float *q, int from, int T,
                           int hd, float scaling, const int w)
{
    int t = from;
    for (; t + 8 <= T; t += 8) scores_kt8(sc + t, kt + t, q, hd, scaling, w);
    for (; t + 4 <= T; t += 4) scores_kt4(sc + t, kt + t, q, hd, scaling, w);
    for (; t < T; t++) {
        float s = 0;
        for (int d = 0; d < hd; d++) s += kt[(size_t)d * (size_t)w + t] * q[d];
        sc[t] = s * scaling;
    }
}



static float scores_bf16(float *sc, const uint16_t *K, const float *q, int T, int hd, float scaling)
{
    float mx = -INFINITY;
    int t = 0;
    for (; t + 8 <= T; t += 8) {
        const uint16_t *k0 = K + (size_t)t * hd, *k1 = k0 + hd, *k2 = k1 + hd, *k3 = k2 + hd;
        const uint16_t *k4 = k3 + hd, *k5 = k4 + hd, *k6 = k5 + hd, *k7 = k6 + hd;
        float s0 = 0, s1 = 0, s2 = 0, s3 = 0, s4 = 0, s5 = 0, s6 = 0, s7 = 0;
        for (int d = 0; d < hd; d++) {
            float qd = q[d];
            s0 += bf16_dec(k0[d]) * qd; s1 += bf16_dec(k1[d]) * qd;
            s2 += bf16_dec(k2[d]) * qd; s3 += bf16_dec(k3[d]) * qd;
            s4 += bf16_dec(k4[d]) * qd; s5 += bf16_dec(k5[d]) * qd;
            s6 += bf16_dec(k6[d]) * qd; s7 += bf16_dec(k7[d]) * qd;
        }
        sc[t] = s0 * scaling; sc[t + 1] = s1 * scaling; sc[t + 2] = s2 * scaling; sc[t + 3] = s3 * scaling;
        sc[t + 4] = s4 * scaling; sc[t + 5] = s5 * scaling; sc[t + 6] = s6 * scaling; sc[t + 7] = s7 * scaling;
        for (int j = 0; j < 8; j++) if (sc[t + j] > mx) mx = sc[t + j];
    }
    for (; t < T; t++) {
        const uint16_t *kr = K + (size_t)t * hd;
        float s = 0;
        for (int d = 0; d < hd; d++) s += bf16_dec(kr[d]) * q[d];
        sc[t] = s * scaling;
        if (sc[t] > mx) mx = sc[t];
    }
    return mx;
}

static void accum_bf16(float *out, const uint16_t *V, const float *sc, int T, int hd)
{
    int t = 0;
    for (; t + 4 <= T; t += 4) {
        const uint16_t *v0 = V + (size_t)t * hd, *v1 = v0 + hd, *v2 = v1 + hd, *v3 = v2 + hd;
        float p0 = sc[t], p1 = sc[t + 1], p2 = sc[t + 2], p3 = sc[t + 3];
        for (int d = 0; d < hd; d++)
            out[d] = (((out[d] + p0 * bf16_dec(v0[d])) + p1 * bf16_dec(v1[d])) + p2 * bf16_dec(v2[d]))
                     + p3 * bf16_dec(v3[d]);
    }
    for (; t < T; t++) {
        const uint16_t *vr = V + (size_t)t * hd;
        float p = sc[t];
        for (int d = 0; d < hd; d++) out[d] += p * bf16_dec(vr[d]);
    }
}

static void accum_f32(float *out, const float *V, const float *sc, int T, int hd)
{
    int t = 0;
    for (; t + 4 <= T; t += 4) {
        const float *v0 = V + (size_t)t * hd, *v1 = v0 + hd, *v2 = v1 + hd, *v3 = v2 + hd;
        float p0 = sc[t], p1 = sc[t + 1], p2 = sc[t + 2], p3 = sc[t + 3];
        int d = 0;
#ifdef __aarch64__
        for (; d + 8 <= hd; d += 8) {
            float32x4_t a = vld1q_f32(out + d), b = vld1q_f32(out + d + 4);
            a = vaddq_f32(a, vmulq_n_f32(vld1q_f32(v0 + d), p0));
            b = vaddq_f32(b, vmulq_n_f32(vld1q_f32(v0 + d + 4), p0));
            a = vaddq_f32(a, vmulq_n_f32(vld1q_f32(v1 + d), p1));
            b = vaddq_f32(b, vmulq_n_f32(vld1q_f32(v1 + d + 4), p1));
            a = vaddq_f32(a, vmulq_n_f32(vld1q_f32(v2 + d), p2));
            b = vaddq_f32(b, vmulq_n_f32(vld1q_f32(v2 + d + 4), p2));
            a = vaddq_f32(a, vmulq_n_f32(vld1q_f32(v3 + d), p3));
            b = vaddq_f32(b, vmulq_n_f32(vld1q_f32(v3 + d + 4), p3));
            vst1q_f32(out + d, a);
            vst1q_f32(out + d + 4, b);
        }
#endif
        for (; d < hd; d++)
            out[d] = (((out[d] + p0 * v0[d]) + p1 * v1[d]) + p2 * v2[d]) + p3 * v3[d];
    }
    for (; t < T; t++) {
        const float *vr = V + (size_t)t * hd;
        float p = sc[t];
        for (int d = 0; d < hd; d++) out[d] += p * vr[d];
    }
}

static void accum_i8(float *out, const int8_t *V, const float *vs, const float *sc, int T, int hd)
{
    for (int t = 0; t < T; t++) {
        const int8_t *vr = V + (size_t)t * hd;
        float p = sc[t] * vs[t];
        int d = 0;
#ifdef __aarch64__
        const float32x4_t pv = vdupq_n_f32(p);
        for (; d + 16 <= hd; d += 16) {
            const int8x16_t x = vld1q_s8(vr + d);
            const int16x8_t lo = vmovl_s8(vget_low_s8(x)), hi = vmovl_s8(vget_high_s8(x));
            vst1q_f32(out + d, vaddq_f32(vld1q_f32(out + d), vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(lo))), pv)));
            vst1q_f32(out + d + 4, vaddq_f32(vld1q_f32(out + d + 4), vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s16(lo))), pv)));
            vst1q_f32(out + d + 8, vaddq_f32(vld1q_f32(out + d + 8), vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(hi))), pv)));
            vst1q_f32(out + d + 12, vaddq_f32(vld1q_f32(out + d + 12), vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s16(hi))), pv)));
        }
#endif
        for (; d < hd; d++) out[d] += p * (float)vr[d];
    }
}

static inline int32_t dot_i8(const int8_t *w, const int8_t *q, int k);

static void scores_i8_group(float **scp, float *mx, const int8_t *K, const float *ks,
                            const int8_t *qq, const float *qs, int nh, int tb, int tn, int hd)
{
    for (int t = 0; t < tn; t++) {
        const int8_t *kr = K + (size_t)t * hd;
        const float kt = ks[t];
        for (int h = 0; h < nh; h++) {
            const float v = (float)dot_i8(kr, qq + (size_t)h * hd, hd) * (kt * qs[h]);
            scp[h][tb + t] = v;
            if (v > mx[h]) mx[h] = v;
        }
    }
}

#define FX_FRAC 12
#define FX_CUT 20
#define FX_ONE ((1u << 20) - 1)
static uint32_t FX_EXPF[1 << FX_FRAC], FX_EXPI[FX_CUT + 1];
static long FX_CLAMP[4][8];

static void fx_tables(void)
{
    for (int f = 0; f < (1 << FX_FRAC); f++)
        FX_EXPF[f] = (uint32_t)lrint((double)FX_ONE * exp(-(double)f / (double)(1 << FX_FRAC)));
    for (int i = 0; i <= FX_CUT; i++)
        FX_EXPI[i] = (uint32_t)lrint((double)FX_ONE * exp(-(double)i));
}

static uint32_t fx_u16(double x, long *clamped)
{
    long v = lrint(x * 65536.0);
    if (v > 65535) { (*clamped)++; return 65535; }
    return v < 0 ? 0 : (uint32_t)v;
}

static void fx_group(float **scp, float **outp, const int8_t *K, const float *ks, const int8_t *V,
                     const float *vs, const int8_t *qq, const float *qs, int nh, int T, int hd, int id)
{
    uint32_t qf[ATT_MAXG];
    int64_t top[ATT_MAXG];
    for (int h = 0; h < nh; h++) {
        long v = lrint((double)qs[h] * 1048576.0);
        if (v > 65535) { FX_CLAMP[0][id]++; v = 65535; }
        qf[h] = v < 0 ? 0 : (uint32_t)v;
        top[h] = INT64_MIN;
    }
    for (int t = 0; t < T; t++) {
        const int8_t *kr = K + (size_t)t * hd;
        const int64_t k16 = fx_u16(ks[t], &FX_CLAMP[1][id]);
        for (int h = 0; h < nh; h++) {
            int64_t s = ((int64_t)dot_i8(kr, qq + (size_t)h * hd, hd) * k16 * (int64_t)qf[h]) >> (36 - FX_FRAC);
            if (s > (1 << 23) - 1) { FX_CLAMP[3][id]++; s = (1 << 23) - 1; }
            if (s < -(1 << 23)) { FX_CLAMP[3][id]++; s = -(1 << 23); }
            scp[h][t] = (float)s;
            if (s > top[h]) top[h] = s;
        }
    }
    uint64_t sum[ATT_MAXG];
    for (int h = 0; h < nh; h++) {
        sum[h] = 0;
        for (int t = 0; t < T; t++) {
            int64_t d = top[h] - (int64_t)scp[h][t];
            int64_t whole = d >> FX_FRAC;
            uint32_t w = whole > FX_CUT
                ? 0
                : (uint32_t)(((uint64_t)FX_EXPF[d & ((1 << FX_FRAC) - 1)] * FX_EXPI[whole]) >> 20);
            scp[h][t] = (float)w;
            sum[h] += w;
        }
    }
    int64_t acc[ATT_MAXG][256];
    for (int h = 0; h < nh; h++)
        for (int d = 0; d < hd; d++) acc[h][d] = 0;
    for (int t = 0; t < T; t++) {
        const int8_t *vr = V + (size_t)t * hd;
        const uint64_t v16 = fx_u16(vs[t], &FX_CLAMP[2][id]);
        for (int h = 0; h < nh; h++) {
            const uint64_t w = (uint64_t)scp[h][t];
            if (!w) continue;
            const int64_t u = (int64_t)((w * v16) >> 16);
            for (int d = 0; d < hd; d++) acc[h][d] += u * (int64_t)vr[d];
        }
    }
    for (int h = 0; h < nh; h++)
        for (int d = 0; d < hd; d++)
            outp[h][d] = sum[h] ? (float)((double)acc[h][d] / (double)sum[h]) : 0.0f;
}

static uint32_t ATT_WORDS[4][5140];

static int attn_fabric(void)
{
    const int hd = M.head_dim, T = R.cur_T, l = R.cur_layer, nq = M.n_q;
    const unsigned P = ATT_PORTS < (unsigned)T ? ATT_PORTS : (unsigned)T;
    const float scaling = 1.0f / sqrtf((float)hd);
    uint8_t hdr[(1 + 3 + 4 + FAB_NQMAX * (FAB_FDMAX / 16)) * 16];
    int8_t qq[FAB_NQMAX * FAB_FDMAX];
    uint32_t qf[20];
    int32_t top[20];
    uint64_t sum[20];
    int64_t acc[20 * 128];
    unsigned k;
    for (int h = 0; h < nq; h++) {
        long v = lrint((double)(scaling / absmax_int8(qq + h * hd, R.cur_q + h * hd, hd)) * 1048576.0);
        qf[h] = v > 65535 ? 65535u : v < 0 ? 0u : (uint32_t)v;
        top[h] = INT32_MIN;
        sum[h] = 0;
    }
    memset(acc, 0, sizeof acc);
    for (unsigned pass = 0; pass < 2; pass++) {
        const size_t words = pass ? (size_t)FAB.outb : (size_t)FAB.outa;
        memset(hdr, 0, sizeof hdr);
        hdr[0] = 0x7E; hdr[1] = 0xA7; hdr[2] = (uint8_t)pass;
        for (int h = 0; h < nq; h++) {
            hdr[16 + 2 * h] = (uint8_t)qf[h];
            hdr[17 + 2 * h] = (uint8_t)(qf[h] >> 8);
            uint32_t t24 = (uint32_t)top[h] & 0xFFFFFFu;
            hdr[FAB.off_top + 3 * h] = (uint8_t)t24;
            hdr[FAB.off_top + 1 + 3 * h] = (uint8_t)(t24 >> 8);
            hdr[FAB.off_top + 2 + 3 * h] = (uint8_t)(t24 >> 16);
        }
        memcpy(hdr + FAB.off_qq, qq, (size_t)nq * (size_t)hd);
        for (k = 0; k < P; k++) {
            const unsigned n = (k + 1) * (unsigned)T / P - k * (unsigned)T / P;
            hdr[4] = (uint8_t)n;
            hdr[5] = (uint8_t)(n >> 8);
            copy_in(B1.va + OFF_ATT_HDR + k * ATT_STRIDE, hdr, FAB_HEADER);
        }
        for (k = 0; k < P; k++) {
            wr(gpio[k], GPIO_DATA, ATT_SEL);
            s2mm_start(k, B1.phys + OFF_ATT_RES + k * ATT_STRIDE, words * 4u);
            mm2s_start(k, B1.phys + OFF_ATT_HDR + k * ATT_STRIDE, FAB_HEADER);
        }
        for (k = 0; k < P; k++)
            if (wait_ioc(k, MM2S_DMASR, "attention header")) return -1;
        for (k = 0; k < P; k++) {
            const unsigned first = k * (unsigned)T / P, n = (k + 1) * (unsigned)T / P - first;
            mm2s_start(k, B0.phys + FAB_BASE_OFF + (uint64_t)R.cur_seq * R.fab_seq
                          + ((uint64_t)l * (uint64_t)R.ctx + first) * FAB_BLOCK,
                       (size_t)n * FAB_BLOCK);
        }
        for (k = 0; k < P; k++)
            if (wait_ioc(k, MM2S_DMASR, "attention cache")) return -1;
        for (k = 0; k < P; k++)
            if (wait_ioc(k, S2MM_DMASR, "attention answers")) return -1;
        for (k = 0; k < P; k++) {
            copy_out(ATT_WORDS[k], &B1, OFF_ATT_RES + k * ATT_STRIDE, words * 4u);
            for (int h = 0; h < nq; h++) {
                if (!pass) {
                    int32_t t = (int32_t)ATT_WORDS[k][h];
                    if (t > top[h]) top[h] = t;
                } else {
                    sum[h] += ATT_WORDS[k][h];
                    for (int d = 0; d < hd; d++) {
                        const size_t e = (size_t)FAB.outa + 2u * ((size_t)h * (size_t)FAB_FD + (size_t)d);
                        acc[h * hd + d] += (int64_t)(((uint64_t)(int64_t)(int32_t)ATT_WORDS[k][e + 1] << 32)
                                                     | ATT_WORDS[k][e]);
                    }
                }
            }
        }
    }
    for (k = 0; k < P; k++) wr(gpio[k], GPIO_DATA, 0);
    for (int h = 0; h < nq; h++)
        for (int d = 0; d < hd; d++)
            R.cur_out[h * hd + d] = sum[h] ? (float)((double)acc[h * hd + d] / (double)sum[h]) : 0.0f;
    return 0;
}

static void attn_worker(void *arg, int id, int nt);

static uint32_t ATT_QF[4][FAB_NQMAX];
static int32_t ATT_TOP[4][FAB_NQMAX];
static uint64_t ATT_SUM[4][FAB_NQMAX];
static int64_t ATT_ACC[4][FAB_NQMAX * FAB_FDMAX];
static int8_t ATT_QQ[4][FAB_NQMAX * FAB_FDMAX];

static int attn_fabric_groups(void)
{
    const int hd = M.head_dim, T = R.cur_T, l = R.cur_layer;
    const int G = R.fab_groups, gq = M.groups;
    const float scaling = 1.0f / sqrtf((float)hd);
    const unsigned P = ATT_PORTS < (unsigned)G ? ATT_PORTS : (unsigned)G;
    uint8_t hdr[(1 + 3 + 4 + FAB_NQMAX * (FAB_FDMAX / 16)) * 16];
    unsigned i;

    for (int c0 = 0; c0 < G; c0 += (int)P) {
        const int nc = c0 + (int)P <= G ? (int)P : G - c0;
        for (i = 0; i < (unsigned)nc; i++) {
            const int c = c0 + (int)i;
            memset(ATT_QQ[i], 0, sizeof ATT_QQ[i]);
            memset(ATT_ACC[i], 0, sizeof ATT_ACC[i]);
            for (int s = 0; s < FAB_NQ; s++) {
                ATT_QF[i][s] = 0;
                ATT_TOP[i][s] = INT32_MIN;
                ATT_SUM[i][s] = 0;
            }
            for (int g = 0; g < FAB_KV; g++) {
                const int kvh = c * FAB_KV + g;
                if (kvh >= M.n_kv) break;
                for (int j = 0; j < gq; j++) {
                    const int qh = kvh * gq + j, slot = g * FAB_QPK + j;
                    if (qh >= M.n_q || slot >= FAB_NQ) break;
                    const double e = (double)(scaling / absmax_int8(ATT_QQ[i] + (size_t)slot * FAB_FD,
                                                                   R.cur_q + (size_t)qh * hd, hd));
                    const long v = lrint(e * 1048576.0);
                    ATT_QF[i][slot] = v > 65535 ? 65535u : v < 0 ? 0u : (uint32_t)v;
                }
            }
        }
        for (unsigned pass = 0; pass < 2; pass++) {
            const size_t words = pass ? (size_t)FAB.outb : (size_t)FAB.outa;
            for (i = 0; i < (unsigned)nc; i++) {
                memset(hdr, 0, sizeof hdr);
                hdr[0] = 0x7E; hdr[1] = 0xA7; hdr[2] = (uint8_t)pass;
                hdr[4] = (uint8_t)T; hdr[5] = (uint8_t)((unsigned)T >> 8);
                for (int s = 0; s < FAB_NQ; s++) {
                    hdr[16 + 2 * s] = (uint8_t)ATT_QF[i][s];
                    hdr[17 + 2 * s] = (uint8_t)(ATT_QF[i][s] >> 8);
                    const uint32_t t24 = (uint32_t)ATT_TOP[i][s] & 0xFFFFFFu;
                    hdr[FAB.off_top + 3 * s] = (uint8_t)t24;
                    hdr[FAB.off_top + 1 + 3 * s] = (uint8_t)(t24 >> 8);
                    hdr[FAB.off_top + 2 + 3 * s] = (uint8_t)(t24 >> 16);
                }
                memcpy(hdr + FAB.off_qq, ATT_QQ[i], (size_t)FAB_NQ * FAB_FD);
                copy_in(B1.va + OFF_ATT_HDR + i * ATT_STRIDE, hdr, FAB_HEADER);
            }
            for (i = 0; i < (unsigned)nc; i++) {
                wr(gpio[i], GPIO_DATA, ATT_SEL);
                s2mm_start(i, B1.phys + OFF_ATT_RES + i * ATT_STRIDE, words * 4u);
                mm2s_start(i, B1.phys + OFF_ATT_HDR + i * ATT_STRIDE, FAB_HEADER);
            }
            for (i = 0; i < (unsigned)nc; i++)
                if (wait_ioc(i, MM2S_DMASR, "attention header")) return -1;
            for (i = 0; i < (unsigned)nc; i++) {
                const uint64_t base = (uint64_t)((size_t)l * (size_t)G + (size_t)(c0 + (int)i))
                                      * (uint64_t)R.ctx * FAB_BLOCK;
                mm2s_start(i, B0.phys + FAB_BASE_OFF + (uint64_t)R.cur_seq * R.fab_seq + base,
                           (size_t)T * FAB_BLOCK);
            }
            if (!pass && M.n_kv > R.fab_kv) parallel_for(attn_worker, 0);
            for (i = 0; i < (unsigned)nc; i++)
                if (wait_ioc(i, MM2S_DMASR, "attention cache")) return -1;
            for (i = 0; i < (unsigned)nc; i++)
                if (wait_ioc(i, S2MM_DMASR, "attention answers")) return -1;
            for (i = 0; i < (unsigned)nc; i++) {
                copy_out(ATT_WORDS[i], &B1, OFF_ATT_RES + i * ATT_STRIDE, words * 4u);
                for (int s = 0; s < FAB_NQ; s++) {
                    if (!pass) {
                        const int32_t v = (int32_t)ATT_WORDS[i][s];
                        if (v > ATT_TOP[i][s]) ATT_TOP[i][s] = v;
                    } else {
                        ATT_SUM[i][s] += ATT_WORDS[i][s];
                        for (int d = 0; d < hd; d++) {
                            const size_t e = (size_t)FAB.outa + 2u * ((size_t)s * FAB_FD + (size_t)d);
                            ATT_ACC[i][(size_t)s * FAB_FD + d] +=
                                (int64_t)(((uint64_t)(int64_t)(int32_t)ATT_WORDS[i][e + 1] << 32)
                                          | ATT_WORDS[i][e]);
                        }
                    }
                }
            }
        }
        for (i = 0; i < (unsigned)nc; i++) wr(gpio[i], GPIO_DATA, 0);
        for (i = 0; i < (unsigned)nc; i++) {
            const int c = c0 + (int)i;
            for (int g = 0; g < FAB_KV; g++) {
                const int kvh = c * FAB_KV + g;
                if (kvh >= M.n_kv) break;
                for (int j = 0; j < gq; j++) {
                    const int qh = kvh * gq + j, slot = g * FAB_QPK + j;
                    if (qh >= M.n_q || slot >= FAB_NQ) break;
                    const double sm = (double)ATT_SUM[i][slot];
                    for (int d = 0; d < hd; d++)
                        R.cur_out[(size_t)qh * hd + d] =
                            sm > 0 ? (float)((double)ATT_ACC[i][(size_t)slot * FAB_FD + d] / sm) : 0.0f;
                }
            }
        }
    }
    return 0;
}

static void attn_worker(void *arg, int id, int nt)
{
    (void)arg;
    const int hd = M.head_dim, T = R.cur_T, groups = M.groups, W = ATT_W;
    const float scaling = 1.0f / sqrtf((float)hd);
    const int base = ATT_LO, span = M.n_q - base;
    const int lo = base + span * id / nt, hi = base + span * (id + 1) / nt;
    float *scb = R.scores + (size_t)id * groups * R.ctx;
    for (int hq = lo; hq < hi;) {
        const int g = hq / groups;
        int end = (g + 1) * groups;
        if (end > hi) end = hi;
        const int nh = end - hq;
        const size_t vbase = (size_t)R.cur_layer * R.v_layer + (size_t)g * R.v_head;
        const size_t ktbase = (size_t)R.cur_layer * R.kt_layer + (size_t)g * R.kt_head;
        const size_t sbase = ((size_t)R.cur_layer * M.n_kv + (size_t)g) * (size_t)R.ctx;
        float mx[ATT_MAXG];
        float *scp[ATT_MAXG], *outp[ATT_MAXG];
        for (int h = 0; h < nh; h++) {
            mx[h] = -INFINITY;
            scp[h] = scb + (size_t)h * R.ctx;
            outp[h] = R.cur_out + (size_t)(hq + h) * hd;
        }
        int8_t qq[ATT_MAXG * 256];
        float qs[ATT_MAXG];
        if (R.k_i8 || R.cache_dtype == CACHE_FX)
            for (int h = 0; h < nh; h++)
                qs[h] = scaling / absmax_int8(qq + (size_t)h * hd, R.cur_q + (size_t)(hq + h) * hd, hd);
        if (R.cache_dtype == CACHE_FX) {
            fx_group(scp, outp, R.cache_k8 + vbase, R.cache_ks + sbase, R.cache_v8 + vbase, R.cache_vs + sbase,
                     qq, qs, nh, T, hd, id);
            hq = end;
            continue;
        }
        for (int tb = 0; tb < T; tb += W) {
            const int tn = tb + W < T ? W : T - tb;
            if (R.k_i8) {
                scores_i8_group(scp, mx, R.cache_k8 + vbase + (size_t)tb * hd, R.cache_ks + sbase + tb,
                                qq, qs, nh, tb, tn, hd);
            } else if (R.cache_dtype == CACHE_BF16) {
                for (int h = 0; h < nh; h++) {
                    float *sc = scp[h] + tb;
                    const float *q = R.cur_q + (size_t)(hq + h) * hd;
                    float m = scores_bf16(sc, R.cache_kb + vbase + (size_t)tb * hd, q, tn, hd, scaling);
                    if (m > mx[h]) mx[h] = m;
                }
            } else {
                const float *kt = R.cache_k + ktbase + (size_t)(tb / W) * R.kt_tile;
                int t = 0;
                for (; t + 16 <= tn; t += 16) {
                    int h = 0;
                    for (; h + 4 <= nh; h += 4)
                        scores_kt16x4(scp[h] + tb + t, scp[h + 1] + tb + t, scp[h + 2] + tb + t,
                                      scp[h + 3] + tb + t, kt + t,
                                      R.cur_q + (size_t)(hq + h) * hd, R.cur_q + (size_t)(hq + h + 1) * hd,
                                      R.cur_q + (size_t)(hq + h + 2) * hd, R.cur_q + (size_t)(hq + h + 3) * hd,
                                      hd, scaling);
                    for (; h + 2 <= nh; h += 2)
                        scores_kt16x2(scp[h] + tb + t, scp[h + 1] + tb + t, kt + t,
                                      R.cur_q + (size_t)(hq + h) * hd, R.cur_q + (size_t)(hq + h + 1) * hd,
                                      hd, scaling);
                    for (; h < nh; h++)
                        scores_kt16(scp[h] + tb + t, kt + t, R.cur_q + (size_t)(hq + h) * hd,
                                    hd, scaling, W);
                }
                for (int h = 0; h < nh; h++)
                    scores_kt_tail(scp[h] + tb, kt, R.cur_q + (size_t)(hq + h) * hd, t, tn,
                                   hd, scaling, W);
                for (int h = 0; h < nh; h++)
                    for (int j = 0; j < tn; j++)
                        if (scp[h][tb + j] > mx[h]) mx[h] = scp[h][tb + j];
            }
        }
        for (int h = 0; h < nh; h++) {
            float *sc = scp[h];
            float sum = 0;
            for (int t = 0; t < T; t++) { sc[t] = expf(sc[t] - mx[h]); sum += sc[t]; }
            float inv = 1.0f / sum;
            for (int t = 0; t < T; t++) sc[t] *= inv;
            for (int d = 0; d < hd; d++) outp[h][d] = 0;
        }
        for (int tb = 0; tb < T; tb += W) {
            const int tn = tb + W < T ? W : T - tb;
            if (!R.v_i8 && R.cache_dtype != CACHE_BF16) {
                const float *V = R.cache_v + vbase + (size_t)tb * hd;
                for (int h = 0; h < nh; h++) accum_f32(outp[h], V, scp[h] + tb, tn, hd);
            } else {
                for (int h = 0; h < nh; h++) {
                    float *sc = scp[h] + tb;
                    if (R.cache_dtype == CACHE_BF16)
                        accum_bf16(outp[h], R.cache_vb + vbase + (size_t)tb * hd, sc, tn, hd);
                    else
                        accum_i8(outp[h], R.cache_v8 + vbase + (size_t)tb * hd,
                                 R.cache_vs + sbase + tb, sc, tn, hd);
                }
            }
        }
        hq = end;
    }
}

struct head_job {
    int32_t *best_i;
    float *best_v;
};


static inline int32_t dot_i8(const int8_t *w, const int8_t *q, int k)
{
#ifdef __aarch64__
    int32x4_t a0 = vdupq_n_s32(0), a1 = vdupq_n_s32(0), a2 = vdupq_n_s32(0), a3 = vdupq_n_s32(0);
    int i = 0;
    for (; i + 32 <= k; i += 32) {
        int8x16_t w0 = vld1q_s8(w + i), q0 = vld1q_s8(q + i);
        int8x16_t w1 = vld1q_s8(w + i + 16), q1 = vld1q_s8(q + i + 16);
        a0 = vpadalq_s16(a0, vmull_s8(vget_low_s8(w0), vget_low_s8(q0)));
        a1 = vpadalq_s16(a1, vmull_s8(vget_high_s8(w0), vget_high_s8(q0)));
        a2 = vpadalq_s16(a2, vmull_s8(vget_low_s8(w1), vget_low_s8(q1)));
        a3 = vpadalq_s16(a3, vmull_s8(vget_high_s8(w1), vget_high_s8(q1)));
    }
    int32_t s = vaddvq_s32(vaddq_s32(vaddq_s32(a0, a1), vaddq_s32(a2, a3)));
    for (; i < k; i++) s += (int32_t)w[i] * q[i];
    return s;
#else
    int32_t s = 0;
    for (int i = 0; i < k; i++) s += (int32_t)w[i] * q[i];
    return s;
#endif
}

static void head_worker(void *arg, int id, int nt)
{
    struct head_job *j = arg;
    const int K = M.hidden;
    const int rows = M.vocab;
    int lo = (int)((long)rows * id / nt) & ~3, hi = id + 1 == nt ? rows : (int)((long)rows * (id + 1) / nt) & ~3;
    float best = -INFINITY;
    int besti = lo;
    float *out = R.logits;
    for (int r = lo; r < hi; r++) {
        int32_t acc = dot_i8(R.hw + (size_t)r * K, R.hq, K);
        float v = (float)acc * R.hs[r] * R.hinv;
        out[r] = v;
        if (v > best) { best = v; besti = r; }
    }
    j->best_v[id] = best;
    j->best_i[id] = besti;
}

static inline int hbetter(float va, int ra, float vb, int rb)
{
    return va > vb || (va == vb && ra < rb);
}

static void heap_down(float *v, int *r, int n, int i)
{
    for (;;) {
        int l = 2 * i + 1, m = i;
        if (l < n && hbetter(v[m], r[m], v[l], r[l])) m = l;
        if (l + 1 < n && hbetter(v[m], r[m], v[l + 1], r[l + 1])) m = l + 1;
        if (m == i) return;
        float tv = v[i];
        int tr = r[i];
        v[i] = v[m]; r[i] = r[m];
        v[m] = tv; r[m] = tr;
        i = m;
    }
}

static void heap_push(float *v, int *r, int *n, int k, float val, int row)
{
    if (*n < k) {
        int i = (*n)++;
        v[i] = val;
        r[i] = row;
        while (i > 0) {
            int p = (i - 1) / 2;
            if (!hbetter(v[p], r[p], v[i], r[i])) break;
            float tv = v[i];
            int tr = r[i];
            v[i] = v[p]; r[i] = r[p];
            v[p] = tv; r[p] = tr;
            i = p;
        }
        return;
    }
    if (!hbetter(val, row, v[0], r[0])) return;
    v[0] = val;
    r[0] = row;
    heap_down(v, r, k, 0);
}

struct topk_job {
    const int32_t *sum;
    const float *scale;
    int k;
    float *v;
    int *r;
    int n[MAXT];
    unsigned npe, c, ch;
};

static void topk_chunk_worker(void *arg, int id, int nt)
{
    struct topk_job *j = arg;
    const int k = j->k;
    const unsigned npe = j->npe, cr = npe / j->ch;
    float *v = j->v + (size_t)id * k;
    int *r = j->r + (size_t)id * k;
    int n = j->n[id];
    for (unsigned e = 0; e < E; e++) {
        const unsigned base = e * npe + j->c * cr;
        const unsigned lo = base + (unsigned)((uint64_t)cr * (unsigned)id / (unsigned)nt);
        const unsigned hi = base + (unsigned)((uint64_t)cr * (unsigned)(id + 1) / (unsigned)nt);
        const unsigned end = hi < (unsigned)M.vocab ? hi : (unsigned)M.vocab;
        for (unsigned i = lo; i < end; i++)
            heap_push(v, r, &n, k, (float)j->sum[i] * j->scale[i], (int)i);
    }
    j->n[id] = n;
}

struct rescore_job {
    const int *rows;
    int n;
    int32_t *best_i;
    float *best_v;
};

static void rescore_worker(void *arg, int id, int nt)
{
    struct rescore_job *j = arg;
    const int K = M.hidden;
    int lo = (int)((long)j->n * id / nt), hi = (int)((long)j->n * (id + 1) / nt);
    float best = -INFINITY;
    int besti = -1;
    for (int i = lo; i < hi; i++) {
        int r = j->rows[i];
        float v = (float)dot_i8(R.hw + (size_t)r * K, R.hq, K) * R.hs[r] * R.hinv;
        if (v > best) { best = v; besti = r; }
    }
    j->best_v[id] = best;
    j->best_i[id] = besti;
}

static int cmp_row(const void *a, const void *b)
{
    int x = *(const int *)a, y = *(const int *)b;
    return x < y ? -1 : x > y;
}

static unsigned bitlen32(uint32_t v) { unsigned n = 0; while (v) { n++; v >>= 1; } return n; }
static unsigned stage_shift(uint32_t max) { unsigned b = bitlen32(max); return b > 15 ? b - 15 : 0; }

struct shifts { unsigned s_g, s_u, s_S, s_T, s_V, hmax, m; };

struct ffnprep {
    volatile uint8_t *beats;
    const int32_t *g, *u;
    const int16_t *G;
    const uint32_t *Gabs;
    uint32_t *a, *b;
    unsigned n;
    unsigned r0, r1, len;
    uint32_t mg[MAXT], mu[MAXT], mx[3][MAXT];
    struct shifts s;
    unsigned bgen;
    int bcount;
};

#define BAR_SPIN 10000

static void prep_barrier(struct ffnprep *p, int nt)
{
    if (nt <= 1) return;
    unsigned g = __atomic_load_n(&p->bgen, __ATOMIC_RELAXED);
    if (__atomic_add_fetch(&p->bcount, 1, __ATOMIC_ACQ_REL) == nt) {
        __atomic_store_n(&p->bcount, 0, __ATOMIC_RELAXED);
        __atomic_store_n(&p->bgen, g + 1, __ATOMIC_RELEASE);
        return;
    }
    unsigned spins = BAR_SPIN;
    while (__atomic_load_n(&p->bgen, __ATOMIC_ACQUIRE) == g) {
        if (spins) { spins--; cpu_relax(); }
        else sched_yield();
    }
}

#ifdef POOL_TRACE
static void prep_barrier_t(struct ffnprep *p, int nt, int id)
{
    double t = now_ms();
    PF_LOOP[id] += t - PF_MARK[id];
    prep_barrier(p, nt);
    double u = now_ms();
    PF_BAR[id] += u - t;
    PF_MARK[id] = u;
}
#define PREP_BARRIER(p, nt, id) prep_barrier_t(p, nt, id)
#else
#define PREP_BARRIER(p, nt, id) prep_barrier(p, nt)
#endif

static void ffn_prep_a_worker(void *arg, int id, int nt)
{
    struct ffnprep *p = arg;
    const unsigned len = p->len;
    const unsigned lo = (unsigned)((uint64_t)len * (unsigned)id / (unsigned)nt) & ~7u;
    const unsigned hi = id + 1 == nt ? len : (unsigned)((uint64_t)len * (unsigned)(id + 1) / (unsigned)nt) & ~7u;
    const int32_t *g = p->g, *u = p->u;
    uint32_t max_g = p->mg[id], max_u = p->mu[id];
    volatile uint64_t *q = (volatile uint64_t *)p->beats;
    const int16_t *G = p->G;

    for (unsigned h = 0; h < 2; h++) {
        const unsigned base = h ? p->r1 : p->r0;
        unsigned i;
        if (!M.silu)
            for (i = base + lo; i < base + hi; i++) {
                q[2 * i] = (uint64_t)(uint32_t)g[i] | (uint64_t)(uint32_t)u[i] << 32;
                q[2 * i + 1] = (uint64_t)(uint16_t)G[i];
            }
        i = base + lo;
#ifdef __aarch64__
        {
            const int32x4_t zero = vdupq_n_s32(0);
            uint32x4_t vg = vdupq_n_u32(max_g), vu = vdupq_n_u32(max_u);
            for (; i + 8 <= base + hi; i += 8) {
                vg = vmaxq_u32(vg, vreinterpretq_u32_s32(vmaxq_s32(vld1q_s32(g + i), zero)));
                vu = vmaxq_u32(vu, vreinterpretq_u32_s32(vabsq_s32(vld1q_s32(u + i))));
                vg = vmaxq_u32(vg, vreinterpretq_u32_s32(vmaxq_s32(vld1q_s32(g + i + 4), zero)));
                vu = vmaxq_u32(vu, vreinterpretq_u32_s32(vabsq_s32(vld1q_s32(u + i + 4))));
            }
            max_g = vmaxvq_u32(vg);
            max_u = vmaxvq_u32(vu);
        }
#endif
        for (; i < base + hi; i++) {
            uint32_t rg = g[i] > 0 ? (uint32_t)g[i] : 0;
            uint32_t au = u[i] < 0 ? (uint32_t)(-(int64_t)u[i]) : (uint32_t)u[i];
            if (rg > max_g) max_g = rg;
            if (au > max_u) max_u = au;
        }
    }
    barrier();
    p->mg[id] = max_g;
    p->mu[id] = max_u;
}

static void ffn_prep_worker(void *arg, int id, int nt)
{
    struct ffnprep *p = arg;
    const unsigned n = p->n;
    const unsigned lo = (unsigned)((uint64_t)n * (unsigned)id / (unsigned)nt) & ~7u;
    const unsigned hi = id + 1 == nt ? n : (unsigned)((uint64_t)n * (unsigned)(id + 1) / (unsigned)nt) & ~7u;
    const int32_t *g = p->g, *u = p->u;
    uint32_t *a = p->a, *b = p->b;
    uint32_t max = 0;
    unsigned i;

#ifdef POOL_TRACE
    PF_MARK[id] = now_ms();
#endif
    uint32_t xg = 0, xu = 0;
    for (int t = 0; t < nt; t++) {
        if (p->mg[t] > xg) xg = p->mg[t];
        if (p->mu[t] > xu) xu = p->mu[t];
    }
    const unsigned s_g = stage_shift(xg), s_u = stage_shift(xu);
    if (id == 0) { p->s.s_g = s_g; p->s.s_u = s_u; }
    for (i = lo; i < hi; i++) {
        uint32_t rg = g[i] > 0 ? (uint32_t)g[i] : 0;
        uint32_t au = u[i] < 0 ? (uint32_t)(-(int64_t)u[i]) : (uint32_t)u[i];
        uint32_t av = rg >> s_g;
        b[i] = au >> s_u;
        a[i] = av * av;
        if (a[i] > max) max = a[i];
    }
    p->mx[0][id] = max;
    PREP_BARRIER(p, nt, id);
    uint32_t x = 0;
    for (int t = 0; t < nt; t++) if (p->mx[0][t] > x) x = p->mx[0][t];
    const unsigned s_S = stage_shift(x);
    if (id == 0) p->s.s_S = s_S;

    max = 0;
    for (i = lo; i < hi; i++) { a[i] = (a[i] >> s_S) * b[i]; if (a[i] > max) max = a[i]; }
    p->mx[1][id] = max;
    PREP_BARRIER(p, nt, id);
    x = 0;
    for (int t = 0; t < nt; t++) if (p->mx[1][t] > x) x = p->mx[1][t];
    const unsigned s_T = stage_shift(x);
    if (id == 0) p->s.s_T = s_T;

    const uint32_t *Gabs = p->Gabs;
    max = 0;
    for (i = lo; i < hi; i++) { a[i] = (a[i] >> s_T) * Gabs[i]; if (a[i] > max) max = a[i]; }
    p->mx[2][id] = max;
    PREP_BARRIER(p, nt, id);
    if (id == 0) {
        x = 0;
        for (int t = 0; t < nt; t++) if (p->mx[2][t] > x) x = p->mx[2][t];
        p->s.s_V = stage_shift(x);
        p->s.hmax = x >> p->s.s_V;
        p->s.m = p->s.hmax == 0 ? 65535u
                 : ((127u << 15) / p->s.hmax > 65535u ? 65535u : (127u << 15) / p->s.hmax);
    }
#ifdef POOL_TRACE
    PF_LOOP[id] += now_ms() - PF_MARK[id];
#endif
}

static struct ffnprep FP;

static void gu_chunk(void *arg, unsigned c, unsigned ch)
{
    struct ffnprep *p = arg;
    double t0 = now_ms();
    const unsigned half = p->n / 2, cr = half / ch;
    const size_t cb = (size_t)cr * 4u;
    for (unsigned e = 0; e < E; e++)
        inval_range(B1.cva + OFF_GU + (size_t)e * half * 4u + (size_t)c * cb, cb);
    p->r0 = c * cr;
    p->r1 = half + c * cr;
    p->len = cr;
    parallel_for(ffn_prep_a_worker, p);
    double d = now_ms() - t0;
    CHUNK_WORK_MS += d;
    R.t.scan += d;
}

struct sidejob {
    const int32_t *g, *u;
    const float *gam;
    unsigned n;
    double wsg, wsu, wsd, sx2, eps;
    double absmax_y;
    float fs;
};

static struct sidejob SJ;

static void ffn_scale(struct sidejob *j)
{
    const int32_t *gi = j->g, *ui = j->u;
    const float *gam = j->gam;
    const int I = (int)j->n;
    double s0 = 0, s1 = 0, s2 = 0, s3 = 0, a0 = 0, a1 = 0, a2 = 0, a3 = 0;
    int i = 0;
    for (; i + 4 <= I; i += 4) {
        double r0 = gi[i] > 0 ? (double)gi[i] : 0.0, P0 = (r0 * r0) * (double)ui[i];
        double r1 = gi[i + 1] > 0 ? (double)gi[i + 1] : 0.0, P1 = (r1 * r1) * (double)ui[i + 1];
        double r2 = gi[i + 2] > 0 ? (double)gi[i + 2] : 0.0, P2 = (r2 * r2) * (double)ui[i + 2];
        double r3 = gi[i + 3] > 0 ? (double)gi[i + 3] : 0.0, P3 = (r3 * r3) * (double)ui[i + 3];
        s0 += P0 * P0; s1 += P1 * P1; s2 += P2 * P2; s3 += P3 * P3;
        double q0 = fabs((double)gam[i] * P0), q1 = fabs((double)gam[i + 1] * P1);
        double q2 = fabs((double)gam[i + 2] * P2), q3 = fabs((double)gam[i + 3] * P3);
        if (q0 > a0) a0 = q0;
        if (q1 > a1) a1 = q1;
        if (q2 > a2) a2 = q2;
        if (q3 > a3) a3 = q3;
    }
    double sumsq = (s0 + s1) + (s2 + s3);
    double amax = a0 > a1 ? a0 : a1, am2 = a2 > a3 ? a2 : a3;
    if (am2 > amax) amax = am2;
    for (; i < I; i++) {
        double rg = gi[i] > 0 ? (double)gi[i] : 0.0;
        double P = (rg * rg) * (double)ui[i];
        sumsq += P * P;
        double a = fabs((double)gam[i] * P);
        if (a > amax) amax = a;
    }
    double Kc = j->wsg * j->wsg * j->wsu / (j->sx2 * j->sx2 * j->sx2);
    double denom = sqrt(sumsq / (double)I + j->eps / (Kc * Kc));
    j->absmax_y = denom > 0 ? amax / denom : 0.0;
    j->fs = (float)(j->wsd * j->absmax_y / 127.0);
}

struct silujob {
    const int32_t *g;
    const float *gam;
    float *y;
    int8_t *q8;
    unsigned n, r0, len;
    int stride, boff, nb;
    float kg, ku, qs;
    float kgv[BMAX], kuv[BMAX];
    double eps, wsd, absmax_y;
    double sq[BMAX][MAXT];
    float mx[BMAX][MAXT];
    float fs;
};

static struct silujob SB;

static void silu_a_worker(void *arg, int id, int nt)
{
    struct silujob *j = arg;
    const unsigned len = j->len;
    const unsigned lo = j->r0 + (unsigned)((uint64_t)len * (unsigned)id / (unsigned)nt);
    const unsigned hi = j->r0 + (unsigned)((uint64_t)len * (unsigned)(id + 1) / (unsigned)nt);
    const int32_t *g = j->g;
    const float *gam = j->gam;
    const int st = j->stride, bo = j->boff, b = j->nb > 1 ? bo : 0;
    const unsigned ub = j->n;
    float *y = j->y;
    const float kg = j->kgv[b], ku = j->kuv[b];
    double sq = j->sq[b][id];
    float mx = j->mx[b][id];
    for (unsigned i = lo; i < hi; i++) {
        float G = (float)g[(size_t)i * st + bo] * kg;
        float v = (G / (1.0f + expf(-G))) * ((float)g[(size_t)(ub + i) * st + bo] * ku);
        float a = fabsf(v * gam[i]);
        y[i] = v;
        sq += (double)v * (double)v;
        if (a > mx) mx = a;
    }
    j->sq[b][id] = sq;
    j->mx[b][id] = mx;
}

static void silu_b_worker(void *arg, int id, int nt)
{
    struct silujob *j = arg;
    const unsigned n = j->n;
    const unsigned lo = (unsigned)((uint64_t)n * (unsigned)id / (unsigned)nt);
    const unsigned hi = (unsigned)((uint64_t)n * (unsigned)(id + 1) / (unsigned)nt);
    const float *y = j->y, *gam = j->gam;
    const float qs = j->qs;
    int8_t *q8 = j->q8;
    for (unsigned i = lo; i < hi; i++) {
        float v = y[i] * gam[i] * qs;
        long r = lrintf(v);
        if (r > 127) r = 127;
        if (r < -128) r = -128;
        q8[i] = (int8_t)r;
    }
}

static void silu_chunk(void *arg, unsigned c, unsigned ch)
{
    struct silujob *j = arg;
    double t0 = now_ms();
    const unsigned half = j->n / 2, cr = half / ch;
    const int nb = j->nb;
    const size_t cb = (size_t)cr * 4u * (size_t)nb;
    for (unsigned e = 0; e < E; e++)
        inval_range(B1.cva + OFF_GU + (size_t)e * half * 4u * (size_t)nb + (size_t)c * cb, cb);
    for (int b = 0; b < nb; b++) {
        j->boff = nb > 1 ? b : 0;
        j->y = R.ffn + (size_t)b * j->n;
        j->r0 = c * cr;
        j->len = cr;
        parallel_for(silu_a_worker, j);
        j->r0 = half + c * cr;
        parallel_for(silu_a_worker, j);
    }
    double d = now_ms() - t0;
    CHUNK_WORK_MS += d;
    R.t.scan += d;
}

static void silu_ffn(struct silujob *j, int nt, int done)
{
    const int b = j->nb > 1 ? j->boff : 0;
    if (!done) {
        for (int i = 0; i < nt; i++) { j->sq[b][i] = 0; j->mx[b][i] = 0; }
        j->r0 = 0;
        j->len = j->n;
        parallel_for(silu_a_worker, j);
    }
    double sq = 0;
    float mx = 0;
    for (int i = 0; i < nt; i++) { sq += j->sq[b][i]; if (j->mx[b][i] > mx) mx = j->mx[b][i]; }
    double inv = 1.0 / sqrt(sq / (double)j->n + j->eps);
    double amax = (double)mx * inv;
    if (!(amax > 1e-5)) amax = 1e-5;
    j->absmax_y = amax;
    j->qs = amax > (double)mx * inv ? (float)(inv * 127.0 / amax) : (float)(127.0 / (double)mx);
    j->fs = (float)(j->wsd * amax / 127.0);
    parallel_for(silu_b_worker, j);
}

static void side_ffn(void)
{
    double t0 = now_ms();
    ffn_scale(&SJ);
    R.t.sidebusy += now_ms() - t0;
    side_mark(2);
}

struct qkvjob {
    const int32_t *qi, *ki, *vi;
    float scq, sck, scv;
    int layer, pos, seq;
};

static struct qkvjob QJ;

static void qkv_post_worker(void *arg, int id, int nt)
{
    const struct qkvjob *j = arg;
    const int hd = M.head_dim, l = j->layer, pos = j->pos;
    int lo = M.n_q * id / nt, hi = M.n_q * (id + 1) / nt;
    for (int h = lo; h < hi; h++) {
        float *qh = R.qf + (size_t)h * hd;
        scale_i32(qh, j->qi + (size_t)h * hd, hd, j->scq);
        rope_heads(qh, 1, hd, R.rope_c, R.rope_s);
    }
    lo = M.n_kv * id / nt;
    hi = M.n_kv * (id + 1) / nt;
    for (int g = lo; g < hi; g++) {
        float *kg = R.kf + (size_t)g * hd, *vg = R.vf + (size_t)g * hd;
        scale_i32(kg, j->ki + (size_t)g * hd, hd, j->sck);
        scale_i32(vg, j->vi + (size_t)g * hd, hd, j->scv);
        rope_heads(kg, 1, hd, R.rope_c, R.rope_s);
        const size_t vbase = (size_t)l * R.v_layer + (size_t)g * R.v_head + (size_t)pos * hd;
        if (R.cache_dtype == CACHE_BF16) {
            for (int d = 0; d < hd; d++) { kg[d] = bf16_dec(bf16_enc(kg[d])); vg[d] = bf16_dec(bf16_enc(vg[d])); }
            for (int d = 0; d < hd; d++) R.cache_kb[vbase + d] = bf16_enc(kg[d]);
            for (int d = 0; d < hd; d++) R.cache_vb[vbase + d] = bf16_enc(vg[d]);
        } else if (R.cache_dtype == CACHE_FAB && g < R.fab_kv) {
            const int c = g / FAB_KV, s = g % FAB_KV;
            const size_t blk = FAB_BASE_OFF + (size_t)j->seq * R.fab_seq
                             + (((size_t)l * (size_t)R.fab_groups + (size_t)c) * (size_t)R.ctx
                                + (size_t)pos) * FAB_BLOCK;
            static const uint8_t zero[FAB_FDMAX] = {0};
            const size_t kbase = (size_t)FAB.scb * 32u;
            const size_t sk = (size_t)(s / 8) * 16u + (size_t)(s % 8) * 2u;
            int8_t row[FAB_FDMAX];
            uint8_t two[2];
            long clamped = 0;
            uint32_t s16 = fx_u16(1.0f / absmax_int8(row, kg, hd), &clamped);
            copy_in(B0.va + blk + kbase + (size_t)s * FAB_FD, row, (size_t)hd);
            if (hd < FAB_FD)
                copy_in(B0.va + blk + kbase + (size_t)s * FAB_FD + (size_t)hd, zero, (size_t)(FAB_FD - hd));
            two[0] = (uint8_t)s16; two[1] = (uint8_t)(s16 >> 8);
            copy_in(B0.va + blk + sk, two, 2);
            s16 = fx_u16(1.0f / absmax_int8(row, vg, hd), &clamped);
            copy_in(B0.va + blk + kbase + (size_t)(FAB_KV + s) * FAB_FD, row, (size_t)hd);
            if (hd < FAB_FD)
                copy_in(B0.va + blk + kbase + (size_t)(FAB_KV + s) * FAB_FD + (size_t)hd, zero,
                        (size_t)(FAB_FD - hd));
            two[0] = (uint8_t)s16; two[1] = (uint8_t)(s16 >> 8);
            copy_in(B0.va + blk + (size_t)FAB.scb * 16u + sk, two, 2);
            if (s == 0) {
                static const uint8_t pad[16] = {0};
                const size_t tail = (size_t)FAB.scb * 16u - (size_t)FAB_KV * 2u;
                if (tail) {
                    copy_in(B0.va + blk + (size_t)FAB_KV * 2u, pad, tail);
                    copy_in(B0.va + blk + (size_t)FAB.scb * 16u + (size_t)FAB_KV * 2u, pad, tail);
                }
            }
        } else if (R.cache_dtype == CACHE_FX) {
            const size_t sbase = ((size_t)l * M.n_kv + (size_t)g) * (size_t)R.ctx + (size_t)pos;
            R.cache_ks[sbase] = 1.0f / absmax_int8(R.cache_k8 + vbase, kg, hd);
            R.cache_vs[sbase] = 1.0f / absmax_int8(R.cache_v8 + vbase, vg, hd);
        } else {
            const size_t sbase = ((size_t)l * M.n_kv + (size_t)g) * (size_t)R.ctx + (size_t)pos;
            if (R.k_i8) R.cache_ks[sbase] = 1.0f / absmax_int8(R.cache_k8 + vbase, kg, hd);
            else {
                float *kt = R.cache_k + (size_t)l * R.kt_layer + (size_t)g * R.kt_head
                            + (size_t)(pos / (unsigned)ATT_W) * R.kt_tile + (size_t)(pos % (unsigned)ATT_W);
                for (int d = 0; d < hd; d++) kt[(size_t)d * R.ldk] = kg[d];
            }
            if (R.v_i8) R.cache_vs[sbase] = 1.0f / absmax_int8(R.cache_v8 + vbase, vg, hd);
            else memcpy(R.cache_v + vbase, vg, sizeof(float) * hd);
        }
    }
}

struct stage_sink {
    int active;
    int slot;
    float *l0_attn_in, *l0_attn_out, *l0_attn_sub, *l0_o, *l0_mlp_in, *l0_q, *l0_k, *l0_v, *l0_ffn;
    int8_t *l0_xq, *l0_aq, *l0_x2q, *l0_q8;
    int32_t *l0_qi, *l0_ki, *l0_vi, *l0_oi, *l0_g, *l0_u, *l0_d;
    int16_t *l0_h;
    int32_t l0_shifts[8];
    float l0_sx, l0_sa, l0_sx2, l0_absmax_y;
    float *hidden;
};

static struct stage_sink SINK;

static const int32_t *sums_slot(size_t off, unsigned n, int nb, int b, int32_t *scratch)
{
    if (nb == 1) return (const int32_t *)(const void *)(B1.cva + off);
    return scratch + (size_t)b * n;
}

static void sums_spread(size_t off, unsigned n, int nb, int32_t *scratch)
{
    if (nb == 1) return;
    const int32_t *p = (const int32_t *)(const void *)(B1.cva + off);
    for (unsigned i = 0; i < n; i++) {
        const int32_t *s = p + (size_t)i * (unsigned)nb;
        int32_t *d = scratch + i;
        for (int b = 0; b < nb; b++) d[(size_t)b * n] = s[b];
    }
}

static int forward_slots(const int *tok, int nb, int nlayers)
{
    const int H = M.hidden, I = M.inter, hd = M.head_dim;
    double t_all = now_ms(), t0;
    float sx[BMAX], sa[BMAX], sx2[BMAX], fs[BMAX];
#ifdef POOL_TRACE
    struct pfslow snap0;
    PF_MAXDISP = 0;
    PF_MAXFN = 0;
    pf_snap(&snap0);
#endif

    for (int b = 0; b < nb; b++) {
        const uint16_t *row = M.embed + (size_t)tok[b] * H;
        float *h = R.hb + (size_t)b * H;
        int i = 0;
#ifdef __aarch64__
        for (; i + 8 <= H; i += 8) {
            uint16x8_t v = vld1q_u16(row + i);
            vst1q_f32(h + i, vreinterpretq_f32_u32(vshll_n_u16(vget_low_u16(v), 16)));
            vst1q_f32(h + i + 4, vreinterpretq_f32_u32(vshll_n_u16(vget_high_u16(v), 16)));
        }
#endif
        for (; i < H; i++) h[i] = bf16_dec(row[i]);
    }
    if (SINK.active) memcpy(SINK.hidden, R.h, sizeof(float) * H);

    for (int b = 0; b < nb; b++)
        for (int d = 0; d < hd / 2; d++) {
            double ang = (double)R.slot_pos[b] * R.inv_freq[d];
            R.rope_cb[(size_t)b * (hd / 2) + d] = (float)cos(ang);
            R.rope_sb[(size_t)b * (hd / 2) + d] = (float)sin(ang);
        }

    const uint64_t qkv_res = B1.phys + OFF_QKV, o_res = B1.phys + OFF_OI, gu_res = B1.phys + OFF_GU;
    const uint64_t d_res = B1.phys + OFF_D, act_phys = B1.phys + OFF_ACT, q8_phys = B1.phys + OFF_Q8;
    const uint64_t h_phys = B1.phys + OFF_H, beats_phys = B1.phys + OFF_BEATS;
    uint64_t w[E], r[E];
    int8_t *actbuf = R.q8_in;
    const unsigned nbu = (unsigned)nb;
    const unsigned B1_ = M.mat[0][PROJ_Q].beats;
    const unsigned Bd = M.mat[0][PROJ_DOWN].beats;
    const size_t vec_h = (size_t)B1_ * M.wpb;
    const size_t vec_i = (size_t)Bd * M.wpb;
    const unsigned nqkv = (unsigned)(M.hidden + 2 * M.kv_size), ngu = 2u * (unsigned)I;

    for (int l = 0; l < nlayers; l++) {
        const struct mat *mq = &M.mat[l][PROJ_Q];
        int trace = SINK.active && l == 0;

        t0 = now_ms();
        for (int b = 0; b < nb; b++) {
            rmsnorm(R.x, R.hb + (size_t)b * H, M.in_norm[l], H, M.eps);
            sx[b] = absmax_int8(actbuf, R.x, H);
            if (trace) { memcpy(SINK.l0_attn_in, R.x, sizeof(float) * H); memcpy(SINK.l0_xq, actbuf, H); SINK.l0_sx = sx[b]; }
            copy_in(B1.va + OFF_ACT + (size_t)b * vec_h, actbuf, (size_t)H);
        }
        R.t.norm += now_ms() - t0;

        {
            unsigned npe = (M.mat[l][PROJ_Q].n + M.mat[l][PROJ_K].n + M.mat[l][PROJ_V].n) / E;
            for (unsigned i = 0; i < E; i++) {
                w[i] = B0.phys + mq->offset + (uint64_t)i * npe * B1_ * BEAT_BYTES;
                r[i] = qkv_res + (uint64_t)i * npe * nbu * 4u;
            }
            if (engine_phase("qkv", B1_, nbu, npe, act_phys, vec_h, w, r, &R.t.act, &R.t.eng_qkv, 1, 1, 0, 0)) return 3;
        }
        t0 = now_ms();
        sums_in(&B1, OFF_QKV, (size_t)nqkv * nbu * 4);
        sums_spread(OFF_QKV, nqkv, nb, R.sqkv);
        R.t.copyout += now_ms() - t0;

        for (int b = 0; b < nb; b++) {
            t0 = now_ms();
            const int32_t *qi = sums_slot(OFF_QKV, nqkv, nb, b, R.sqkv);
            const int32_t *ki = qi + H, *vi = ki + M.kv_size;
            R.t.copyout += now_ms() - t0;
            if (trace) {
                memcpy(SINK.l0_qi, qi, sizeof(int32_t) * H);
                memcpy(SINK.l0_ki, ki, sizeof(int32_t) * M.kv_size);
                memcpy(SINK.l0_vi, vi, sizeof(int32_t) * M.kv_size);
            }

            t0 = now_ms();
            R.rope_c = R.rope_cb + (size_t)b * (hd / 2);
            R.rope_s = R.rope_sb + (size_t)b * (hd / 2);
            QJ.qi = qi; QJ.ki = ki; QJ.vi = vi; QJ.layer = l;
            QJ.pos = R.slot_pos[b]; QJ.seq = R.slot_seq[b];
            QJ.scq = (float)M.mat[l][PROJ_Q].ws / sx[b];
            QJ.sck = (float)M.mat[l][PROJ_K].ws / sx[b];
            QJ.scv = (float)M.mat[l][PROJ_V].ws / sx[b];
            parallel_for(qkv_post_worker, &QJ);
            if (trace) {
                memcpy(SINK.l0_q, R.qf, sizeof(float) * H);
                memcpy(SINK.l0_k, R.kf, sizeof(float) * M.kv_size);
                memcpy(SINK.l0_v, R.vf, sizeof(float) * M.kv_size);
            }
            R.t.norm += now_ms() - t0;

            t0 = now_ms();
            R.cur_layer = l; R.cur_T = R.slot_pos[b] + 1; R.cur_q = R.qf; R.cur_out = R.attn;
            R.cur_seq = R.slot_seq[b];
            if (R.cache_dtype == CACHE_FAB) {
                if (R.fab_groups > 1 ? attn_fabric_groups() : attn_fabric()) { print_dma_status(); return 3; }
                if (M.n_kv > R.fab_kv && R.fab_groups == 1) parallel_for(attn_worker, 0);
            } else {
                parallel_for(attn_worker, 0);
            }
            R.t.attn += now_ms() - t0;
            if (trace) memcpy(SINK.l0_attn_out, R.attn, sizeof(float) * H);

            t0 = now_ms();
            rmsnorm(R.xa, R.attn, M.attn_sub[l], H, M.eps);
            sa[b] = absmax_int8(actbuf, R.xa, H);
            R.t.norm += now_ms() - t0;
            if (trace) { memcpy(SINK.l0_attn_sub, R.xa, sizeof(float) * H); memcpy(SINK.l0_aq, actbuf, H); SINK.l0_sa = sa[b]; }
            copy_in(B1.va + OFF_ACT + (size_t)b * vec_h, actbuf, (size_t)H);
        }
        {
            unsigned npe = M.mat[l][PROJ_O].n / E;
            for (unsigned i = 0; i < E; i++) {
                w[i] = B0.phys + M.mat[l][PROJ_O].offset + (uint64_t)i * npe * B1_ * BEAT_BYTES;
                r[i] = o_res + (uint64_t)i * npe * nbu * 4u;
            }
            if (engine_phase("o_proj", B1_, nbu, npe, act_phys, vec_h, w, r, &R.t.act, &R.t.eng_o, 0, 1, 0, 0)) return 3;
        }
        t0 = now_ms();
        sums_in(&B1, OFF_OI, (size_t)H * nbu * 4);
        sums_spread(OFF_OI, (unsigned)H, nb, R.soi);
        R.t.copyout += now_ms() - t0;
        for (int b = 0; b < nb; b++) {
            t0 = now_ms();
            const int32_t *oi = sums_slot(OFF_OI, (unsigned)H, nb, b, R.soi);
            float *h = R.hb + (size_t)b * H;
            float sco = (float)M.mat[l][PROJ_O].ws / sa[b];
            if (trace) {
                memcpy(SINK.l0_oi, oi, sizeof(int32_t) * H);
                for (int i = 0; i < H; i++) SINK.l0_o[i] = (float)oi[i] * sco;
            }
            add_scaled_i32(h, oi, H, sco);

            rmsnorm(R.x, h, M.post_norm[l], H, M.eps);
            sx2[b] = absmax_int8(actbuf, R.x, H);
            R.t.norm += now_ms() - t0;
            if (trace) { memcpy(SINK.l0_mlp_in, R.x, sizeof(float) * H); memcpy(SINK.l0_x2q, actbuf, H); SINK.l0_sx2 = sx2[b]; }
            copy_in(B1.va + OFF_ACT + (size_t)b * vec_h, actbuf, (size_t)H);
        }
        FP.G = M.G[l]; FP.Gabs = M.Gabs[l];
        FP.a = R.scratch; FP.b = R.scratch + I; FP.n = (unsigned)I;
        {
            unsigned npe = (M.mat[l][PROJ_GATE].n + M.mat[l][PROJ_UP].n) / E;
            for (unsigned i = 0; i < E; i++) {
                w[i] = B0.phys + M.mat[l][PROJ_GATE].offset + (uint64_t)i * npe * B1_ * BEAT_BYTES;
                r[i] = gu_res + (uint64_t)i * npe * nbu * 4u;
            }
            if (M.silu) {
                SB.g = (const int32_t *)(const void *)(B1.cva + OFF_GU);
                SB.gam = M.ffn_sub[l]; SB.n = (unsigned)I; SB.q8 = R.q8; SB.eps = M.eps;
                SB.nb = nb; SB.stride = nb; SB.boff = 0;
                SB.wsd = M.mat[l][PROJ_DOWN].ws;
                for (int b = 0; b < nb; b++) {
                    SB.kgv[b] = (float)(M.mat[l][PROJ_GATE].ws / sx2[b]);
                    SB.kuv[b] = (float)(M.mat[l][PROJ_UP].ws / sx2[b]);
                    for (int i = 0; i < MAXT; i++) { SB.sq[b][i] = 0; SB.mx[b][i] = 0; }
                }
                if (engine_phase("gate+up", B1_, nbu, npe, act_phys, vec_h, w, r, &R.t.act, &R.t.eng_gu, 1,
                                 R.gu_chunks, silu_chunk, &SB)) return 3;
            } else if (nb == 1) {
                FP.beats = B1.va + OFF_BEATS;
                FP.g = (const int32_t *)(const void *)(B1.cva + OFF_GU); FP.u = FP.g + I;
                memset(FP.mg, 0, sizeof FP.mg);
                memset(FP.mu, 0, sizeof FP.mu);
                if (engine_phase("gate+up", B1_, 1u, npe, act_phys, vec_h, w, r, &R.t.act, &R.t.eng_gu, 1,
                                 R.gu_chunks, gu_chunk, &FP)) return 3;
            } else if (engine_phase("gate+up", B1_, nbu, npe, act_phys, vec_h, w, r, &R.t.act, &R.t.eng_gu, 1,
                                    1, 0, 0)) return 3;
        }
        if (nb > 1) {
            t0 = now_ms();
            sums_in(&B1, OFF_GU, (size_t)ngu * nbu * 4);
            sums_spread(OFF_GU, ngu, nb, R.sgu);
            R.t.copyout += now_ms() - t0;
        }

        struct shifts shb[BMAX];
        for (int b = 0; b < nb; b++) {
            const int32_t *gi, *ui;
            if (nb == 1) { gi = (const int32_t *)(const void *)(B1.cva + OFF_GU); ui = gi + I; }
            else {
                t0 = now_ms();
                gi = sums_slot(OFF_GU, ngu, nb, b, R.sgu);
                ui = gi + I;
                R.t.copyout += now_ms() - t0;
                FP.beats = B1.va + OFF_BEATS + (size_t)b * I * 16;
                FP.g = gi; FP.u = ui;
                memset(FP.mg, 0, sizeof FP.mg);
                memset(FP.mu, 0, sizeof FP.mu);
                t0 = now_ms();
                FP.r0 = 0; FP.r1 = (unsigned)I / 2u; FP.len = (unsigned)I / 2u;
                parallel_for(ffn_prep_a_worker, &FP);
                R.t.scan += now_ms() - t0;
            }
            if (trace) { memcpy(SINK.l0_g, gi, sizeof(int32_t) * I); memcpy(SINK.l0_u, ui, sizeof(int32_t) * I); }

            if (M.silu) {
                t0 = now_ms();
                SB.boff = nb > 1 ? b : 0;
                SB.y = R.ffn + (size_t)b * (size_t)I;
                silu_ffn(&SB, POOL.n > 0 ? POOL.n : 1, 1);
                fs[b] = SB.fs;
                R.t.scan += now_ms() - t0;
                t0 = now_ms();
                copy_in(B1.va + OFF_Q8 + (size_t)b * vec_i, R.q8, (size_t)I);
                R.t.copyout += now_ms() - t0;
                if (trace) {
                    memcpy(SINK.l0_q8, R.q8, (size_t)I);
                    SINK.l0_absmax_y = (float)SB.absmax_y;
                }
                continue;
            }

            t0 = now_ms();
            parallel_for(ffn_prep_worker, &FP);
            R.t.scan += now_ms() - t0;
            shb[b] = FP.s;

            struct sidejob *sj = &SJ;
            sj->g = gi; sj->u = ui; sj->gam = M.ffn_sub[l]; sj->n = (unsigned)I;
            sj->wsg = M.mat[l][PROJ_GATE].ws; sj->wsu = M.mat[l][PROJ_UP].ws; sj->wsd = M.mat[l][PROJ_DOWN].ws;
            sj->sx2 = (double)sx2[b]; sj->eps = M.eps;
            if (nb == 1) side_post(side_ffn);
            else {
                t0 = now_ms();
                ffn_scale(sj);
                R.t.sidebusy += now_ms() - t0;
                fs[b] = sj->fs;
                if (trace) SINK.l0_absmax_y = (float)sj->absmax_y;
            }
        }

        for (int b = 0; b < nb && !M.silu; b++) {
            const struct shifts sh = shb[b];
            uint32_t gs;
            uint32_t params = PARAMS(sh.s_g, sh.s_u, sh.s_S, sh.s_T, sh.s_V);
            if (glue_pass("pass A", 0, (unsigned)I, params, 0, beats_phys + (uint64_t)b * I * 16,
                          (size_t)I * 16, h_phys, (size_t)I * 4, &R.t.glue_a, &gs)) return 3;
            if (GSTATUS_HMAX(gs) != sh.hmax) {
                fprintf(stderr, "  layer %d pass A: the glue's hmax %u, the A53's %u\n", l, GSTATUS_HMAX(gs), sh.hmax);
                return 3;
            }
            if (glue_pass("pass B", 1, (unsigned)I, params, sh.m, h_phys, (size_t)I * 4,
                          q8_phys + (uint64_t)b * vec_i, (size_t)I, &R.t.glue_b, &gs)) return 3;

            if (nb == 1) {
                side_wait(2, &R.t.sidewait);
                fs[b] = SJ.fs;
            }

            if (trace) {
                t0 = now_ms();
                copy_out(R.hglue, &B1, OFF_H, (size_t)I * 4);
                copy_out(R.q8, &B1, OFF_Q8, (size_t)I);
                R.t.copyout += now_ms() - t0;
                for (int i = 0; i < I; i++) SINK.l0_h[i] = (int16_t)R.hglue[i];
                memcpy(SINK.l0_q8, R.q8, (size_t)I);
                SINK.l0_shifts[0] = (int32_t)sh.s_g; SINK.l0_shifts[1] = (int32_t)sh.s_u; SINK.l0_shifts[2] = M.kg[l];
                SINK.l0_shifts[3] = (int32_t)sh.s_S; SINK.l0_shifts[4] = (int32_t)sh.s_T; SINK.l0_shifts[5] = (int32_t)sh.s_V;
                SINK.l0_shifts[6] = (int32_t)sh.hmax; SINK.l0_shifts[7] = (int32_t)sh.m;
                SINK.l0_absmax_y = (float)SJ.absmax_y;
            }
        }

        {
            unsigned npe = M.mat[l][PROJ_DOWN].n / E;
            for (unsigned i = 0; i < E; i++) {
                w[i] = B0.phys + M.mat[l][PROJ_DOWN].offset + (uint64_t)i * npe * Bd * BEAT_BYTES;
                r[i] = d_res + (uint64_t)i * npe * nbu * 4u;
            }
            if (engine_phase("down", Bd, nbu, npe, q8_phys, vec_i, w, r, &R.t.act, &R.t.eng_down, 0, 1, 0, 0)) return 3;
        }
        t0 = now_ms();
        sums_in(&B1, OFF_D, (size_t)H * nbu * 4);
        sums_spread(OFF_D, (unsigned)H, nb, R.sd);
        R.t.copyout += now_ms() - t0;
        for (int b = 0; b < nb; b++) {
            t0 = now_ms();
            const int32_t *di = sums_slot(OFF_D, (unsigned)H, nb, b, R.sd);
            float *h = R.hb + (size_t)b * H;
            if (trace) {
                memcpy(SINK.l0_d, di, sizeof(int32_t) * H);
                for (int i = 0; i < H; i++) SINK.l0_ffn[i] = (float)di[i] * fs[b];
            }
            add_scaled_i32(h, di, H, fs[b]);
            R.t.norm += now_ms() - t0;
        }
        if (SINK.active) memcpy(SINK.hidden + (size_t)(l + 1) * H, R.h, sizeof(float) * H);
    }

    t0 = now_ms();
    for (int b = 0; b < nb; b++)
        rmsnorm(R.finb + (size_t)b * H, R.hb + (size_t)b * H, M.final_norm, H, M.eps);
    memcpy(R.fin, R.finb + (size_t)(nb - 1) * H, sizeof(float) * (size_t)H);
    R.t.norm += now_ms() - t0;
    R.t.total += now_ms() - t_all;
    R.t.tokens += nb;
#ifdef POOL_TRACE
    {
        double d = R.t.scan - SCAN_LAST;
        if (SCAN_N < (int)(sizeof SCAN_FWD / sizeof SCAN_FWD[0])) SCAN_FWD[SCAN_N++] = d;
        SCAN_LAST = R.t.scan;
        if (d > 6.0 && PF_NSLOW < (int)(sizeof PF_SLOW / sizeof PF_SLOW[0])) {
            struct pfslow *s = &PF_SLOW[PF_NSLOW++];
            pf_snap(s);
            struct timespec rt;
            clock_gettime(CLOCK_REALTIME, &rt);
            s->when = (double)rt.tv_sec + rt.tv_nsec / 1e9;
            s->idx = PF_FWD;
            s->scan = d;
            s->wall = now_ms() - t_all;
            s->maxdisp = PF_MAXDISP;
            s->fn = PF_MAXFN;
            s->minflt -= snap0.minflt;
            s->majflt -= snap0.majflt;
            for (int i = 0; i <= POOL.n; i++) { s->run[i] -= snap0.run[i]; s->wait[i] -= snap0.wait[i]; }
            for (int i = 0; i < 8; i++) {
                s->busy[i] -= snap0.busy[i];
                s->idle[i] -= snap0.idle[i];
                s->syst[i] -= snap0.syst[i];
            }
            for (int i = 0; i < MAXT; i++) { s->bar[i] -= snap0.bar[i]; s->loop[i] -= snap0.loop[i]; }
        }
        PF_FWD++;
    }
#endif
    return 0;
}

static int forward_n(const int *tok, int pos0, int nb, int nlayers)
{
    for (int b = 0; b < nb; b++) { R.slot_pos[b] = pos0 + b; R.slot_seq[b] = R.cur_seq; }
    return forward_slots(tok, nb, nlayers);
}

static int forward(int token, int pos, int nlayers)
{
    return forward_n(&token, pos, 1, nlayers);
}

static int head_argmax_arm(void)
{
    double t0 = now_ms();
    float s = absmax_int8(R.hq, R.fin, M.hidden);
    R.hinv = 1.0f / s;
    int32_t bi[MAXT];
    float bv[MAXT];
    struct head_job j = { bi, bv };
    parallel_for(head_worker, &j);
    int best = -1;
    float bestv = -INFINITY;
    for (int i = 0; i < POOL.n; i++)
        if (bv[i] > bestv) { bestv = bv[i]; best = bi[i]; }
    for (int r = 0; r < best; r++)
        if (R.logits[r] == bestv) { best = r; break; }
    R.t.head += now_ms() - t0;
    R.t.heads++;
    return best;
}

static void head_chunk(void *arg, unsigned c, unsigned ch)
{
    struct topk_job *j = arg;
    double t0 = now_ms();
    const size_t cb = (size_t)j->npe * 4u / ch;
    for (unsigned e = 0; e < E; e++)
        inval_range(B1.cva + OFF_HEADSUM + (size_t)e * j->npe * 4u + (size_t)c * cb, cb);
    j->c = c;
    j->ch = ch;
    parallel_for(topk_chunk_worker, j);
    double d = now_ms() - t0;
    CHUNK_WORK_MS += d;
    R.t.head_top += d;
}

static int head_argmax_fabric(void)
{
    double t0 = now_ms(), t1;
    float s = absmax_int8(R.hq, R.fin, M.hidden);
    R.hinv = 1.0f / s;
    copy_in(B1.va + OFF_ACT, R.hq, (size_t)M.hidden);

    uint64_t w[E], r[E];
    const unsigned npe = M.head_t_npe, B = M.head_t_beats;
    for (unsigned i = 0; i < E; i++) {
        w[i] = B2.phys + (uint64_t)i * npe * B * 16u;
        r[i] = B1.phys + OFF_HEADSUM + (uint64_t)i * npe * 4u;
    }
    const int K = R.head_k;
    const int32_t *sum = (const int32_t *)(const void *)(B1.cva + OFF_HEADSUM);
    struct topk_job tj = { sum, R.hts, K, R.hcand, R.hcandi, { 0 }, npe, 0, 1 };
    if (engine_phase("head", B, 1u, npe, B1.phys + OFF_ACT, (size_t)B * M.wpb, w, r,
                     &R.t.head_act, &R.t.head_s1, 1, R.head_chunks, head_chunk, &tj)) {
        fprintf(stderr, "bitnet_kria: the ternary head stream failed\n");
        exit(3);
    }

    t1 = now_ms();
    float *mv = R.hcand + (size_t)MAXT * K;
    int *mr = R.hcandi + (size_t)MAXT * K;
    int mn = 0;
    for (int t = 0; t < POOL.n; t++)
        for (int i = 0; i < tj.n[t]; i++)
            heap_push(mv, mr, &mn, K, R.hcand[(size_t)t * K + i], R.hcandi[(size_t)t * K + i]);
    memcpy(R.hshort, mr, sizeof(int) * (size_t)mn);
    qsort(R.hshort, (size_t)mn, sizeof(int), cmp_row);
    R.t.head_top += now_ms() - t1;

    t1 = now_ms();
    int32_t bi[MAXT];
    float bv[MAXT];
    struct rescore_job rj = { R.hshort, mn, bi, bv };
    parallel_for(rescore_worker, &rj);
    int best = -1;
    float bestv = -INFINITY;
    for (int i = 0; i < POOL.n; i++)
        if (bv[i] > bestv) { bestv = bv[i]; best = bi[i]; }
    R.t.head_s2 += now_ms() - t1;
    R.t.head += now_ms() - t0;
    R.t.heads++;
    return best;
}

static int head_argmax(void)
{
    return R.head_fabric ? head_argmax_fabric() : head_argmax_arm();
}

static const float *SORT_P;
static int cmp_desc(const void *a, const void *b)
{
    int x = *(const int *)a, y = *(const int *)b;
    if (SORT_P[x] > SORT_P[y]) return -1;
    if (SORT_P[x] < SORT_P[y]) return 1;
    return x - y;
}

static int sample_token(double temp, double top_p, unsigned *seed)
{
    static int *idx = 0;
    static float *pr = 0;
    const int V = M.vocab;
    if (!idx) { idx = xmalloc(sizeof(int) * (size_t)V); pr = xmalloc(sizeof(float) * (size_t)V); }
    float mx = -INFINITY;
    for (int r = 0; r < V; r++) if (R.logits[r] > mx) mx = R.logits[r];
    double sum = 0;
    for (int r = 0; r < V; r++) { pr[r] = (float)exp((R.logits[r] - mx) / temp); sum += pr[r]; idx[r] = r; }
    SORT_P = pr;
    qsort(idx, (size_t)V, sizeof(int), cmp_desc);
    double keep = top_p >= 1.0 ? sum : top_p * sum, acc = 0;
    int n = 0;
    while (n < V) { acc += pr[idx[n]]; n++; if (acc >= keep) break; }
    double u = (double)rand_r(seed) / ((double)RAND_MAX + 1.0) * acc, c = 0;
    for (int i = 0; i < n; i++) { c += pr[idx[i]]; if (c > u) return idx[i]; }
    return idx[n - 1];
}

struct check_stats { size_t n, differ; double maxabs, maxrel, dot, na, nb; };

static void cmp_f32(const struct refstages *st, const char *name, size_t from, const float *got, size_t n,
                    double atol, double rtol, double cosfloor, int *fail)
{
    size_t have = 0;
    const float *want = refstages_f32(st, name, &have);
    if (!want || from + n > have) { printf("  %-14s MISSING from the dump\n", name); *fail = 1; return; }
    struct check_stats c = { n, 0, 0, 0, 0, 0, 0 };
    double maxref = 0;
    for (size_t i = 0; i < n; i++) {
        double a = got[i], b = want[from + i], d = fabs(a - b);
        if (d > c.maxabs) c.maxabs = d;
        if (fabs(b) > maxref) maxref = fabs(b);
        double rel = fabs(b) > 0 ? d / fabs(b) : (d > 0 ? INFINITY : 0);
        if (rel > c.maxrel) c.maxrel = rel;
        if (!(d <= atol + rtol * fabs(b))) c.differ++;
        c.dot += a * b; c.na += a * a; c.nb += b * b;
    }
    double cos = (c.na > 0 && c.nb > 0) ? c.dot / sqrt(c.na * c.nb) : 1.0;
    printf("  %-14s %6zu float   outside %.0e+%.0e|ref| %6zu   max|diff| %-11.4g of max|ref| %-11.4g cosine %.9f  %s\n",
           name, n, atol, rtol, c.differ, c.maxabs, maxref, cos, cos < cosfloor ? "[BELOW THE FLOOR]" : "ok");
    if (cos < cosfloor) *fail = 1;
}

static void cmp_i32(const struct refstages *st, const char *name, size_t from, const int32_t *got, size_t n, int *fail)
{
    size_t have = 0;
    const int32_t *want = refstages_i32(st, name, &have);
    if (!want || from + n > have) { printf("  %-14s MISSING from the dump\n", name); *fail = 1; return; }
    size_t differ = 0;
    long first = -1;
    int64_t maxd = 0;
    for (size_t i = 0; i < n; i++)
        if (got[i] != want[from + i]) {
            if (first < 0) first = (long)i;
            int64_t d = (int64_t)got[i] - want[from + i];
            if (d < 0) d = -d;
            if (d > maxd) maxd = d;
            differ++;
        }
    printf("  %-14s %6zu int32   differ %6zu %s\n", name, n, differ,
           differ ? "[NOT EXACT]" : "exact");
    if (differ) {
        printf("                 first at %ld: got %d, reference %d; largest |diff| %" PRId64 "\n",
               first, got[first], want[from + first], maxd);
        *fail = 1;
    }
}

static void cmp_i8(const struct refstages *st, const char *name, size_t from, const int8_t *got, size_t n, int *fail)
{
    size_t have = 0;
    const int8_t *want = refstages_i8(st, name, &have);
    if (!want || from + n > have) { printf("  %-14s MISSING from the dump\n", name); *fail = 1; return; }
    size_t differ = 0, within1 = 0;
    long first = -1;
    for (size_t i = 0; i < n; i++)
        if (got[i] != want[from + i]) {
            if (first < 0) first = (long)i;
            int d = got[i] - want[from + i];
            if (d >= -1 && d <= 1) within1++;
            differ++;
        }
    printf("  %-14s %6zu int8    differ %6zu %s\n", name, n, differ, differ ? "[NOT EXACT]" : "exact");
    if (differ) {
        printf("                 first at %ld: got %d, reference %d; within one %zu of %zu\n",
               first, got[first], want[from + first], within1, differ);
        *fail = 1;
    }
}

static void cmp_i16(const struct refstages *st, const char *name, size_t from, const int16_t *got, size_t n, int *fail)
{
    size_t have = 0;
    const int16_t *want = refstages_i16(st, name, &have);
    if (!want || from + n > have) { printf("  %-14s MISSING from the dump\n", name); *fail = 1; return; }
    size_t differ = 0;
    long first = -1;
    for (size_t i = 0; i < n; i++)
        if (got[i] != want[from + i]) { if (first < 0) first = (long)i; differ++; }
    printf("  %-14s %6zu int16   differ %6zu %s\n", name, n, differ, differ ? "[NOT EXACT]" : "exact");
    if (differ) {
        printf("                 first at %ld: got %d, reference %d\n", first, got[first], want[from + first]);
        *fail = 1;
    }
}

static double load_buf(struct udmabuf *b, const char *path, size_t want, int verify)
{
    struct stat st;
    if (stat(path, &st)) die(path);
    if ((size_t)st.st_size != want) {
        fprintf(stderr, "bitnet_kria: %s is %zu bytes, wanted %zu\n", path, (size_t)st.st_size, want);
        exit(2);
    }
    if (want > b->size) {
        fprintf(stderr, "bitnet_kria: %s is %zu bytes, need %zu\n", b->dev, b->size, want);
        exit(2);
    }
    double t0 = now_ms();
    const size_t chunk = 4u << 20;
    uint8_t *stage = xmalloc(chunk);
    int fd = open(path, O_RDONLY);
    if (fd < 0) die(path);
    size_t off = 0;
    for (;;) {
        ssize_t got = read(fd, stage, chunk);
        if (got < 0) die(path);
        if (got == 0) break;
        copy_in(b->va + off, stage, (size_t)got);
        off += (size_t)got;
    }
    close(fd);
    if (off != want) {
        fprintf(stderr, "bitnet_kria: read %zu of %zu bytes of %s\n", off, want, path);
        exit(2);
    }
    double s = (now_ms() - t0) / 1e3;
    if (verify) {
        int fd2 = open(path, O_RDONLY);
        if (fd2 < 0) die(path);
        size_t o = 0, bad = 0;
        for (;;) {
            ssize_t got = read(fd2, stage, chunk);
            if (got <= 0) break;
            if (memcmp((const void *)(b->cva + o), stage, (size_t)got)) bad++;
            o += (size_t)got;
        }
        close(fd2);
        printf("  %s verified against the file: %s\n", path, bad ? "DIFFER" : "identical");
        if (bad) exit(2);
    }
    free(stage);
    return s;
}

static void alloc_run(int ctx, int cache_dtype)
{
    const int H = M.hidden, I = M.inter;
    R.ctx = ctx;
    R.cache_dtype = cache_dtype;
    ATT_LO = 0;
    R.fab_kv = 0;
    R.k_i8 = cache_dtype == CACHE_I8 || cache_dtype == CACHE_K8;
    R.v_i8 = cache_dtype == CACHE_I8 || cache_dtype == CACHE_V8;
    ATT_W = M.groups == 1 ? 16 : ATT_TB;
    const size_t tiles = ((size_t)ctx + (size_t)ATT_W - 1) / (size_t)ATT_W;
    R.ldk = (size_t)ATT_W;
    R.kt_tile = (size_t)M.head_dim * (size_t)ATT_W;
    R.kt_head = tiles * R.kt_tile;
    R.kt_layer = R.kt_head * (size_t)M.n_kv;
    R.v_head = (size_t)ctx * (size_t)M.head_dim;
    R.v_layer = R.v_head * (size_t)M.n_kv;
    size_t cells = (size_t)M.layers * M.n_kv * ctx * M.head_dim;
    size_t ktcells = (size_t)M.layers * R.kt_layer;
    if (cache_dtype == CACHE_BF16) {
        R.cache_kb = xmalloc(cells * 2);
        R.cache_vb = xmalloc(cells * 2);
    } else if (cache_dtype == CACHE_FAB) {
        if (!ATT_PORTS) {
            const char *env = getenv("ATTN_PORTS");
            ATT_PORTS = env ? (unsigned)atoi(env) : 2u;
        }
        {
            const char *sh = getenv("ATTN_SHAPE");
            if (sh && !strcmp(sh, "16x96")) fab_shape(16, 16, 96, 1);
            else if (sh && !strcmp(sh, "16x128")) fab_shape(16, 16, 128, 1);
            else fab_shape(5, 20, 128, 4);
        }
        if (M.head_dim > FAB_FD || M.groups > FAB_QPK || ATT_PORTS < 1 || ATT_PORTS > 4 || ctx > 65535) {
            fprintf(stderr, "bitnet_kria: --cache-dtype fab takes a head dimension up to %d and up to %d query"
                            " heads a group; this model has %d and %d. --attn-ports is 1..4 and the context"
                            " must be under 65536\n",
                    FAB_FD, FAB_QPK, M.head_dim, M.groups);
            exit(2);
        }
        fx_tables();
        R.fab_groups = (M.n_kv + FAB_KV - 1) / FAB_KV;
        if ((unsigned)R.fab_groups > ATT_PORTS) R.fab_groups = (int)ATT_PORTS;
        R.fab_kv = R.fab_groups * FAB_KV;
        if (R.fab_kv > M.n_kv) R.fab_kv = M.n_kv;
        ATT_LO = R.fab_kv * M.groups;
        FAB_BASE_OFF = (M.model_bytes + 0xFFFFFu) & ~(size_t)0xFFFFFu;
        R.fab_seq = (size_t)M.layers * (size_t)R.fab_groups * (size_t)ctx * FAB_BLOCK;
        const size_t need = FAB_BASE_OFF + (size_t)(R.nseq < 1 ? 1 : R.nseq) * R.fab_seq;
        if (need > B0.size) {
            fprintf(stderr, "bitnet_kria: %d sequence%s of %d positions need %zu bytes of %s, which has %zu: "
                            "fewer with --gen-batch or a smaller --context\n",
                    R.nseq < 1 ? 1 : R.nseq, R.nseq == 1 ? "" : "s", ctx, need, B0.dev, B0.size);
            exit(2);
        }
        if (M.n_kv > R.fab_kv) {
            R.k_i8 = R.v_i8 = 1;
            R.cache_k8 = xmalloc(cells);
            R.cache_v8 = xmalloc(cells);
            R.cache_ks = xmalloc(sizeof(float) * (size_t)M.layers * M.n_kv * ctx);
            R.cache_vs = xmalloc(sizeof(float) * (size_t)M.layers * M.n_kv * ctx);
        }
    } else if (cache_dtype == CACHE_I8 || cache_dtype == CACHE_FX
               || cache_dtype == CACHE_K8 || cache_dtype == CACHE_V8) {
        if (cache_dtype == CACHE_FX) fx_tables();
        if (M.head_dim > 256) {
            fprintf(stderr, "bitnet_kria: the int8 cache takes a head dimension up to 256, this model has %d\n", M.head_dim);
            exit(2);
        }
        const size_t sc = sizeof(float) * (size_t)M.layers * M.n_kv * ctx;
        if (cache_dtype == CACHE_V8) R.cache_k = xmalloc(ktcells * 4);
        else { R.cache_k8 = xmalloc(cells); R.cache_ks = xmalloc(sc); }
        if (cache_dtype == CACHE_K8) R.cache_v = xmalloc(cells * 4);
        else { R.cache_v8 = xmalloc(cells); R.cache_vs = xmalloc(sc); }
    } else {
        R.cache_k = xmalloc(ktcells * 4);
        R.cache_v = xmalloc(cells * 4);
    }
    {
        volatile char *p[4] = { (volatile char *)R.cache_k, (volatile char *)R.cache_v,
                                (volatile char *)R.cache_kb, (volatile char *)R.cache_vb };
        size_t n[4] = { R.cache_k ? ktcells * 4 : 0, R.cache_v ? cells * 4 : 0,
                        R.cache_kb ? cells * 2 : 0, R.cache_vb ? cells * 2 : 0 };
        if (R.cache_k8) { p[2] = (volatile char *)R.cache_k8; n[2] = cells; }
        if (R.cache_v8) { p[3] = (volatile char *)R.cache_v8; n[3] = cells; }
        for (int b = 0; b < 4; b++)
            for (size_t o = 0; o < n[b]; o += 4096) p[b][o] = 0;
    }
    R.hb = xmalloc(sizeof(float) * (size_t)H * BMAX);
    R.h = R.hb;
    R.sqkv = xmalloc(sizeof(int32_t) * (size_t)(H + 2 * M.kv_size) * BMAX);
    R.soi = xmalloc(sizeof(int32_t) * (size_t)H * BMAX);
    R.sgu = xmalloc(sizeof(int32_t) * (size_t)(2 * I) * BMAX);
    R.sd = xmalloc(sizeof(int32_t) * (size_t)H * BMAX);
    R.x = xmalloc(sizeof(float) * H);
    R.xa = xmalloc(sizeof(float) * H);
    R.attn = xmalloc(sizeof(float) * H);
    R.qf = xmalloc(sizeof(float) * H);
    R.kf = xmalloc(sizeof(float) * (size_t)M.kv_size);
    R.vf = xmalloc(sizeof(float) * (size_t)M.kv_size);
    R.fin = xmalloc(sizeof(float) * H);
    R.finb = xmalloc(sizeof(float) * (size_t)H * BMAX);
    R.q8_in = xmalloc((size_t)I);
    R.gu = xmalloc(sizeof(int32_t) * (size_t)(2 * I));
    R.hglue = xmalloc(sizeof(int32_t) * (size_t)I);
    R.q8 = xmalloc((size_t)I);
    R.scratch = xmalloc(sizeof(uint32_t) * (size_t)(2 * I));
    R.ffn = xmalloc(sizeof(float) * (size_t)I * BMAX);
    R.logits = xmalloc(sizeof(float) * (size_t)M.vocab);
    R.hq = xmalloc((size_t)H);
    R.hcand = xmalloc(sizeof(float) * (size_t)(MAXT + 1) * (size_t)R.head_k);
    R.hcandi = xmalloc(sizeof(int) * (size_t)(MAXT + 1) * (size_t)R.head_k);
    R.hshort = xmalloc(sizeof(int) * (size_t)R.head_k);
    R.scores = xmalloc(sizeof(float) * (size_t)MAXT * (size_t)M.groups * ctx);
    R.inv_freq = xmalloc(sizeof(double) * (size_t)(M.head_dim / 2));
    R.rope_cb = xmalloc(sizeof(float) * (size_t)(M.head_dim / 2) * BMAX);
    R.rope_sb = xmalloc(sizeof(float) * (size_t)(M.head_dim / 2) * BMAX);
    R.rope_c = R.rope_cb;
    R.rope_s = R.rope_sb;
    for (int d = 0; d < M.head_dim / 2; d++)
        R.inv_freq[d] = 1.0 / pow(M.theta, (double)(2 * d) / (double)M.head_dim);
}

static void alloc_sink(void)
{
    const int H = M.hidden, I = M.inter, KV = M.kv_size;
    SINK.l0_attn_in = xmalloc(sizeof(float) * H);
    SINK.l0_attn_out = xmalloc(sizeof(float) * H);
    SINK.l0_attn_sub = xmalloc(sizeof(float) * H);
    SINK.l0_o = xmalloc(sizeof(float) * H);
    SINK.l0_mlp_in = xmalloc(sizeof(float) * H);
    SINK.l0_ffn = xmalloc(sizeof(float) * H);
    SINK.l0_q = xmalloc(sizeof(float) * H);
    SINK.l0_k = xmalloc(sizeof(float) * KV);
    SINK.l0_v = xmalloc(sizeof(float) * KV);
    SINK.l0_xq = xmalloc(H);
    SINK.l0_aq = xmalloc(H);
    SINK.l0_x2q = xmalloc(H);
    SINK.l0_q8 = xmalloc(I);
    SINK.l0_qi = xmalloc(sizeof(int32_t) * H);
    SINK.l0_ki = xmalloc(sizeof(int32_t) * KV);
    SINK.l0_vi = xmalloc(sizeof(int32_t) * KV);
    SINK.l0_oi = xmalloc(sizeof(int32_t) * H);
    SINK.l0_g = xmalloc(sizeof(int32_t) * I);
    SINK.l0_u = xmalloc(sizeof(int32_t) * I);
    SINK.l0_d = xmalloc(sizeof(int32_t) * H);
    SINK.l0_h = xmalloc(sizeof(int16_t) * I);
    SINK.hidden = xmalloc(sizeof(float) * (size_t)(M.layers + 1) * H);
}

static void print_timing(void)
{
    struct timings *t = &R.t;
    double k = t->tokens ? 1.0 / t->tokens : 0, kh = t->heads ? 1.0 / t->heads : 0;
    printf("per forward over %d (ms): engines qkv %.3f, o %.3f, gate+up %.3f, down %.3f (%.3f together);"
           " activation loads %.3f; glue A %.3f B %.3f; shift scan %.3f; sums out %.3f;"
           " attention %.3f; norms, quant, RoPE and residuals %.3f; waiting on the side thread %.3f;"
           " forward total %.3f. side thread busy %.3f ms a forward (beats and the FFN scale, off the"
           " critical path). head %.3f ms a call over %d calls\n",
           t->tokens, t->eng_qkv * k, t->eng_o * k, t->eng_gu * k, t->eng_down * k,
           (t->eng_qkv + t->eng_o + t->eng_gu + t->eng_down) * k, t->act * k,
           t->glue_a * k, t->glue_b * k, t->scan * k, t->copyout * k, t->attn * k,
           t->norm * k, t->sidewait * k, t->total * k, t->sidebusy * k, t->head * kh, t->heads);
#ifdef POOL_TRACE
    {
        static const struct { const void *fn; const char *name; } NM[] = {
            { (const void *)qkv_post_worker, "qkv_post" },
            { (const void *)attn_worker, "attention" },
            { (const void *)ffn_prep_a_worker, "scan pass A" },
            { (const void *)ffn_prep_worker, "scan passes B-D" },
        };
        int n = SCAN_N, first = -1, over = 0;
        double *v = malloc(sizeof(double) * (size_t)(n ? n : 1));
        v[0] = 0;
        for (int i = 0; i < n; i++) v[i] = SCAN_FWD[i];
        for (int i = 1; i < n; i++) { double x = v[i]; int j = i - 1; while (j >= 0 && v[j] > x) { v[j + 1] = v[j]; j--; } v[j + 1] = x; }
        for (int i = 0; i < n; i++)
            if (SCAN_FWD[i] > 2 * v[n / 2]) { over++; if (first < 0) first = i; }
        printf("pooltrace: %d forwards, scan ms min %.3f p50 %.3f p90 %.3f max %.3f; %d forwards over"
               " twice the median, the first at %d\n",
               n, n ? v[0] : 0, n ? v[n / 2] : 0, n ? v[(9 * n) / 10] : 0, n ? v[n - 1] : 0, over, first);
        free(v);
        for (int i = 0; i < PFN && PFS[i].fn; i++) {
            const char *name = "?";
            for (unsigned j = 0; j < sizeof NM / sizeof NM[0]; j++) if (NM[j].fn == PFS[i].fn) name = NM[j].name;
            struct pfsite *s = &PFS[i];
            printf("pooltrace: %-16s calls %6u  own slice %8.1f ms  waiting on the workers %8.1f ms"
                   "  (max %6.3f)  worker wake %8.1f ms (max %6.3f)  worker work %8.1f ms  yields %u  over half a ms %u\n",
                   name, s->calls, s->self, s->join, s->maxjoin, s->wake, s->maxwake, s->work, s->yields, s->slow);
        }
        for (int i = 0; i <= POOL.n; i++) {
            int tid = i < POOL.n ? PF_TID[i] : PF_SIDE_TID;
            long a[3];
            pf_sched(tid, a);
            printf("pooltrace: thread %d tid %6d %-4s on cpu ran %9.1f ms  waited on a run queue %9.1f ms  slices %ld\n",
                   i, tid, i == 0 ? "main" : (i < POOL.n ? "pool" : "side"),
                   (a[0] - PF_S0[i][0]) / 1e6, (a[1] - PF_S0[i][1]) / 1e6, a[2] - PF_S0[i][2]);
        }
        {
            long c1[8][8];
            memset(c1, 0, sizeof c1);
            pf_cpustat(c1);
            for (int c = 0; c < PF_NCPU; c++) {
                double busy = 0, idle = 0;
                for (int i = 0; i < 8; i++) {
                    double d = (double)(c1[c][i] - PF_C0[c][i]) * 10.0;
                    if (i == 3 || i == 4) idle += d; else busy += d;
                }
                printf("pooltrace: cpu%d busy %8.1f ms idle %8.1f ms (user %.1f nice %.1f sys %.1f irq %.1f softirq %.1f)\n",
                       c, busy, idle, (double)(c1[c][0] - PF_C0[c][0]) * 10.0, (double)(c1[c][1] - PF_C0[c][1]) * 10.0,
                       (double)(c1[c][2] - PF_C0[c][2]) * 10.0, (double)(c1[c][5] - PF_C0[c][5]) * 10.0,
                       (double)(c1[c][6] - PF_C0[c][6]) * 10.0);
            }
        }
        for (int i = 0; i < PF_NSLOW; i++) {
            struct pfslow *s = &PF_SLOW[i];
            const char *name = "?";
            for (unsigned j = 0; j < sizeof NM / sizeof NM[0]; j++) if (NM[j].fn == s->fn) name = NM[j].name;
            printf("pooltrace: slow forward %4d at %.3f scan %9.3f ms of a %9.3f ms forward; worst dispatch"
                   " %8.3f ms (%s); faults min %ld maj %ld\n",
                   s->idx, s->when, s->scan, s->wall, s->maxdisp, name, s->minflt, s->majflt);
            printf("pooltrace:      threads ran/waited ms");
            for (int t = 0; t <= POOL.n; t++) printf("  %.1f/%.1f", s->run[t], s->wait[t]);
            printf("; scan barrier/loop ms");
            for (int t = 0; t < POOL.n; t++) printf("  %.1f/%.1f", s->bar[t], s->loop[t]);
            printf("\npooltrace:      cpu busy/kernel/idle ms");
            for (int c = 0; c < PF_NCPU; c++) printf("  %.0f/%.0f/%.0f", s->busy[c], s->syst[c], s->idle[c]);
            printf("\n");
        }
        printf("pooltrace: ps %.1f C, pl %.1f C\n", pf_temp("0_ps_temp"), pf_temp("2_pl_temp"));
    }
#endif
    if (R.head_fabric)
        printf("head stages (ms a call over %d calls): stage 1 ternary stream %.3f, its activation load"
               " %.3f, sync and top-%d %.3f, stage 2 exact rescore %.3f, head total %.3f\n",
               t->heads, t->head_s1 * kh, t->head_act * kh, R.head_k, t->head_top * kh,
               t->head_s2 * kh, t->head * kh);
    fflush(stdout);
}

static int usage(void)
{
    fprintf(stderr, "usage: sudo ./bitnet_kria [--dir DIR] [--max-new N] [--context N] [--threads N] [--layers N]\n"
                    "                          [--temp T] [--top-p P] [--seed N] [--timing] [--quiet] [--no-mlock]\n"
                    "                          [--cache-dtype f32|bf16|i8|k8|v8] [--stage-check FILE [--check-layers N]]\n"
                    "                          [--verify-weights] [--timeout MS] [--gpio HEX] [--dma HEX]\n"
                    "                          [--stride HEX] [--glue-gpio HEX] [--glue-params HEX]\n"
                    "                          [--glue-dma HEX] [--buf0 DEV] [--buf1 DEV] [--buf2 DEV]\n"
                    "                          [--head arm|fabric] [--head-k N] [--head-chunks N] [--gu-chunks N]\n"
                    "                          [--prompt-batch 1..4] [--gen-batch 1..4] [--gen-batch-fill]\n");
    return 2;
}

int main(int argc, char **argv)
{
    const char *dir = "/home/ubuntu/bitnet-kria", *buf0 = "/dev/udmabuf0", *buf1 = "/dev/udmabuf1";
    const char *buf2 = "/dev/udmabuf2", *head_mode = "arm";
    const char *stage_file = 0;
    int head_k = 256, head_chunks = 8, gu_chunks = 2;
    uint64_t gpio_base = 0xA0000000ull, dma_base = 0xA0040000ull, stride = 0x10000ull;
    uint64_t ga = 0xA0080000ull, gb = 0xA0090000ull, gd = 0xA00A0000ull;
    int max_new = 64, ctx = 2048, threads = 4, cache_dtype = CACHE_F32, timing = 0, quiet = 0, ignore_eos = 0;
    unsigned pbatch = BMAX, gbatch = 1;
    int check_layers = -1, verify_weights = 0, layers_opt = -1, lock_head = 1;
    double temp = 0, top_p = 1.0;
    unsigned seed = 1;

    for (int a = 1; a < argc; a++) {
        const char *k = argv[a], *v = a + 1 < argc ? argv[a + 1] : 0;
        if (!strcmp(k, "--timing")) timing = 1;
        else if (!strcmp(k, "--quiet")) quiet = 1;
        else if (!strcmp(k, "--ignore-eos")) ignore_eos = 1;
        else if (!strcmp(k, "--verify-weights")) verify_weights = 1;
        else if (!strcmp(k, "--no-mlock")) lock_head = 0;
        else if (!strcmp(k, "--gen-batch-fill")) GEN_FILL = 1;
        else if (!v) return usage();
        else if (!strcmp(k, "--dir")) dir = v, a++;
        else if (!strcmp(k, "--buf0")) buf0 = v, a++;
        else if (!strcmp(k, "--buf1")) buf1 = v, a++;
        else if (!strcmp(k, "--buf2")) buf2 = v, a++;
        else if (!strcmp(k, "--head")) head_mode = v, a++;
        else if (!strcmp(k, "--head-k")) head_k = atoi(v), a++;
        else if (!strcmp(k, "--attn-ports")) ATT_PORTS = (unsigned)atoi(v), a++;
        else if (!strcmp(k, "--head-chunks")) head_chunks = atoi(v), a++;
        else if (!strcmp(k, "--gu-chunks")) gu_chunks = atoi(v), a++;
        else if (!strcmp(k, "--prompt-batch")) {
            int pb = atoi(v);
            if (pb < 1 || pb > (int)BMAX) { fprintf(stderr, "bitnet_kria: --prompt-batch takes 1..%u, not %s\n", BMAX, v); return 2; }
            pbatch = (unsigned)pb;
            a++;
        }
        else if (!strcmp(k, "--gen-batch")) {
            int gb = atoi(v);
            if (gb < 1 || gb > (int)BMAX) { fprintf(stderr, "bitnet_kria: --gen-batch takes 1..%u, not %s\n", BMAX, v); return 2; }
            gbatch = (unsigned)gb;
            a++;
        }
        else if (!strcmp(k, "--stage-check")) stage_file = v, a++;
        else if (!strcmp(k, "--check-layers")) check_layers = atoi(v), a++;
        else if (!strcmp(k, "--layers")) layers_opt = atoi(v), a++;
        else if (!strcmp(k, "--max-new")) max_new = atoi(v), a++;
        else if (!strcmp(k, "--context")) ctx = atoi(v), a++;
        else if (!strcmp(k, "--threads")) threads = atoi(v), a++;
        else if (!strcmp(k, "--temp")) temp = atof(v), a++;
        else if (!strcmp(k, "--top-p")) top_p = atof(v), a++;
        else if (!strcmp(k, "--seed")) seed = (unsigned)strtoul(v, 0, 0), a++;
        else if (!strcmp(k, "--timeout")) TIMEOUT_MS = atof(v), a++;
        else if (!strcmp(k, "--cache-dtype")) {
            if (!strcmp(v, "f32")) cache_dtype = CACHE_F32;
            else if (!strcmp(v, "bf16")) cache_dtype = CACHE_BF16;
            else if (!strcmp(v, "i8")) cache_dtype = CACHE_I8;
            else if (!strcmp(v, "k8")) cache_dtype = CACHE_K8;
            else if (!strcmp(v, "v8")) cache_dtype = CACHE_V8;
            else if (!strcmp(v, "fx")) cache_dtype = CACHE_FX;
            else if (!strcmp(v, "fab")) cache_dtype = CACHE_FAB;
            else { fprintf(stderr, "bitnet_kria: --cache-dtype takes f32, bf16 or i8, not %s\n", v); return 2; }
            a++;
        }
        else if (!strcmp(k, "--gpio")) gpio_base = strtoull(v, 0, 0), a++;
        else if (!strcmp(k, "--dma")) dma_base = strtoull(v, 0, 0), a++;
        else if (!strcmp(k, "--stride")) stride = strtoull(v, 0, 0), a++;
        else if (!strcmp(k, "--glue-gpio")) ga = strtoull(v, 0, 0), a++;
        else if (!strcmp(k, "--glue-params")) gb = strtoull(v, 0, 0), a++;
        else if (!strcmp(k, "--glue-dma")) gd = strtoull(v, 0, 0), a++;
        else return usage();
    }
    if (strcmp(head_mode, "arm") && strcmp(head_mode, "fabric")) {
        fprintf(stderr, "bitnet_kria: --head takes arm or fabric, not %s\n", head_mode);
        return 2;
    }
    R.head_fabric = !strcmp(head_mode, "fabric");
    if (head_k < 1 || head_k > HEAD_KMAX) {
        fprintf(stderr, "bitnet_kria: --head-k must be between 1 and %d\n", HEAD_KMAX);
        return 2;
    }
    R.head_k = head_k;
    R.head_chunks = head_chunks < 1 ? 1u : (unsigned)head_chunks;
    R.gu_chunks = gu_chunks < 1 ? 1u : (unsigned)gu_chunks;
    if (R.head_fabric && temp > 0) {
        fprintf(stderr, "bitnet_kria: --temp needs every logit, which the fabric head does not"
                        " compute; falling back to --head arm\n");
        R.head_fabric = 0;
    }

    double t_start = now_ms();
    char path[512];
    snprintf(path, sizeof path, "%s/manifest.json", dir);
    struct jnode *mf = jparse_file(path);
    struct jnode *geo = jget(mf, "geometry");
    if (!geo) { fprintf(stderr, "bitnet_kria: manifest has no geometry\n"); return 2; }
    M.hidden = (int)jnum(geo, "hidden_size", 0);
    M.inter = (int)jnum(geo, "intermediate_size", 0);
    M.layers = (int)jnum(geo, "num_hidden_layers", 0);
    M.n_q = (int)jnum(geo, "num_attention_heads", 0);
    M.n_kv = (int)jnum(geo, "num_key_value_heads", 0);
    M.head_dim = (int)jnum(geo, "head_dim", 0);
    M.kv_size = (int)jnum(geo, "kv_size", M.n_kv * M.head_dim);
    M.vocab = (int)jnum(geo, "vocab_size", 0);
    M.theta = jnum(geo, "rope_theta", 500000.0);
    M.eps = jnum(geo, "rms_norm_eps", 1e-5);
    {
        const char *act = jstr(geo, "hidden_act");
        M.silu = act && !strcmp(act, "silu");
        if (act && strcmp(act, "silu") && strcmp(act, "relu2")) {
            fprintf(stderr, "bitnet_kria: the manifest asks for the gate %s, which this runtime"
                            " does not compute; it knows relu2 and silu\n", act);
            return 2;
        }
    }
    M.groups = M.n_q / M.n_kv;
    if (layers_opt > 0 && layers_opt < M.layers) M.layers = layers_opt;
    if (M.layers > 64) { fprintf(stderr, "bitnet_kria: more than 64 layers\n"); return 2; }
    if (M.groups < 1 || M.groups > ATT_MAXG || M.groups * M.n_kv != M.n_q) {
        fprintf(stderr, "bitnet_kria: %d query heads over %d key/value heads is not a group size this build handles\n",
                M.n_q, M.n_kv);
        return 2;
    }

    {
        struct jnode *sf = jget(mf, "stream_format");
        const char *name = sf ? jstr(sf, "name") : 0;
        const char *file = sf ? jstr(sf, "file") : 0;
        M.wpb = sf ? (unsigned)jnum(sf, "weights_per_beat", 0) : 0;
        M.wpby = sf ? (unsigned)jnum(sf, "weights_per_byte", 0) : 0;
        snprintf(M.enc, sizeof M.enc, "%s", name ? name : "(none)");
        snprintf(M.model_file, sizeof M.model_file, "%s", file ? file : "model.bin");
        if (!name || strcmp(name, "base3") || M.wpb != 80 || M.wpby != 5) {
            fprintf(stderr, "bitnet_kria: this build streams base 3, five weights a byte and eighty a beat; "
                            "the manifest says %s with %u a beat and %u a byte (%s). Repack with "
                            "pack_model.py --encoding base3.\n",
                    M.enc, M.wpb, M.wpby, M.model_file);
            return 2;
        }
    }

    struct jnode *mats = jget(mf, "matrices");
    int seen = 0;
    for (struct jnode *c = mats ? mats->first : 0; c; c = c->next) {
        int l = (int)jnum(c, "layer", -1);
        const char *nm = jstr(c, "name");
        if (l < 0 || l >= M.layers || !nm) continue;
        for (int p = 0; p < NPROJ; p++)
            if (!strcmp(nm, PROJ_NAME[p])) {
                M.mat[l][p].offset = (uint64_t)jnum(c, "offset", 0);
                M.mat[l][p].n = (unsigned)jnum(c, "n", 0);
                M.mat[l][p].k = (unsigned)jnum(c, "k", 0);
                M.mat[l][p].beats = (unsigned)jnum(c, "beats_per_neuron", 0);
                M.mat[l][p].ws = jnum(c, "weight_scale", 0);
                seen++;
            }
    }
    if (seen != M.layers * NPROJ) { fprintf(stderr, "bitnet_kria: found %d of %d matrices\n", seen, M.layers * NPROJ); return 2; }
    for (int l = 0; l < M.layers; l++) {
        const struct mat *m = M.mat[l];
        uint64_t rq = (uint64_t)m[PROJ_Q].beats * BEAT_BYTES, rg = (uint64_t)m[PROJ_GATE].beats * BEAT_BYTES;
        if (m[PROJ_Q].offset + m[PROJ_Q].n * rq != m[PROJ_K].offset ||
            m[PROJ_K].offset + m[PROJ_K].n * rq != m[PROJ_V].offset ||
            m[PROJ_GATE].offset + m[PROJ_GATE].n * rg != m[PROJ_UP].offset ||
            m[PROJ_K].beats != m[PROJ_Q].beats || m[PROJ_V].beats != m[PROJ_Q].beats ||
            m[PROJ_UP].beats != m[PROJ_GATE].beats) {
            fprintf(stderr, "bitnet_kria: layer %d is not laid out q,k,v then gate,up at one stride\n", l);
            return 2;
        }
        for (int p = 0; p < NPROJ; p++) {
            unsigned want = (m[p].k + M.wpb - 1u) / M.wpb;
            if (m[p].beats != want || m[p].beats > 0x7Fu) {
                fprintf(stderr, "bitnet_kria: layer %d %s says %u beats for K = %u; base 3 wants %u (and at most 127)\n",
                        l, PROJ_NAME[p], m[p].beats, m[p].k, want);
                return 2;
            }
        }
    }

    size_t norms_bytes = 0;
    const float *norms = map_ro(dir, "norms.bin", 0, &norms_bytes);
    struct jnode *nm = jget(mf, "norms");
    struct jnode *ents = nm ? jget(nm, "entries") : 0;
    for (struct jnode *c = ents ? ents->first : 0; c; c = c->next) {
        int l = (int)jnum(c, "layer", -1);
        const char *name = jstr(c, "name");
        size_t off = (size_t)jnum(c, "offset", 0);
        const float *p = norms + off / 4;
        if (!name) continue;
        if (l < 0 && !strcmp(name, "model.norm")) M.final_norm = p;
        else if (l >= 0 && l < M.layers) {
            if (!strcmp(name, "input_layernorm")) M.in_norm[l] = p;
            else if (!strcmp(name, "post_attention_layernorm")) M.post_norm[l] = p;
            else if (!strcmp(name, "attn_sub_norm")) M.attn_sub[l] = p;
            else if (!strcmp(name, "ffn_sub_norm")) M.ffn_sub[l] = p;
        }
    }
    for (int l = 0; l < M.layers; l++)
        if (!M.in_norm[l] || !M.post_norm[l] || !M.attn_sub[l] || !M.ffn_sub[l]) {
            fprintf(stderr, "bitnet_kria: layer %d is missing a norm\n", l);
            return 2;
        }
    if (!M.final_norm) { fprintf(stderr, "bitnet_kria: no model.norm\n"); return 2; }
    for (int l = 0; l < M.layers; l++) {
        M.G[l] = xmalloc(sizeof(int16_t) * (size_t)M.inter);
        gamma_fixed(M.ffn_sub[l], (unsigned)M.inter, &M.kg[l], M.G[l]);
        M.Gabs[l] = xmalloc(sizeof(uint32_t) * (size_t)M.inter);
        for (int i = 0; i < M.inter; i++)
            M.Gabs[l][i] = M.G[l][i] < 0 ? (uint32_t)(-(int32_t)M.G[l][i]) : (uint32_t)M.G[l][i];
    }

    M.embed = map_ro(dir, "embed_bf16.bin", (size_t)M.vocab * M.hidden * 2, 0);
    M.head_w = map_ro(dir, "head_i8.bin", (size_t)M.vocab * M.hidden, 0);
    M.head_scale = map_ro(dir, "head_scale.bin", (size_t)M.vocab * 4, 0);
    R.hw = M.head_w;
    R.hs = M.head_scale;

    open_udmabuf(&B0, buf0);
    open_udmabuf(&B1, buf1);
    if (NEED1 > B1.size) { fprintf(stderr, "bitnet_kria: %s is %zu bytes, need %u\n", B1.dev, B1.size, NEED1); return 2; }
    {
        unsigned dbeats = (unsigned)(M.inter + M.wpb - 1) / M.wpb;
        size_t act_all = (size_t)dbeats * M.wpb * BMAX;
        if (OFF_ACT + act_all > OFF_QKV || OFF_Q8 + act_all > OFF_D ||
            OFF_BEATS + (size_t)M.inter * 16u * BMAX > OFF_HEADSUM) {
            fprintf(stderr, "bitnet_kria: %zu bytes of activation slots and %zu of beats do not fit the "
                            "udmabuf1 layout\n", act_all, (size_t)M.inter * 16u * BMAX);
            return 2;
        }
        for (size_t i = 0; i < act_all; i++) { B1.va[OFF_ACT + i] = 0; B1.va[OFF_Q8 + i] = 0; }
        barrier();
    }

    if (R.head_fabric) {
        open_udmabuf(&B2, buf2);
        M.head_rows = (((unsigned)M.vocab + HEAD_ROW_PAD - 1u) / HEAD_ROW_PAD) * HEAD_ROW_PAD;
        M.head_t_scale = map_ro(dir, "head_t_scale.bin", (size_t)M.head_rows * 4, 0);
        M.head_t_beats = ((unsigned)M.hidden + M.wpb - 1u) / M.wpb;
        M.head_t_npe = M.head_rows / E;
        M.head_t_bytes = (size_t)M.head_rows * M.head_t_beats * BEAT_BYTES;
        snprintf(M.head_file, sizeof M.head_file, "head3_t.bin");
        if (M.head_t_npe % R.head_chunks) {
            fprintf(stderr, "bitnet_kria: %u neurons an engine do not split into %u chunks;"
                            " --head-chunks must divide it\n", M.head_t_npe, R.head_chunks);
            return 2;
        }
        if (M.head_t_npe > 0xFFFFu || M.head_t_beats > 0x7Fu) {
            fprintf(stderr, "bitnet_kria: %u neurons of %u beats an engine does not fit the control register\n",
                    M.head_t_npe, M.head_t_beats);
            return 2;
        }
        if (OFF_HEADSUM + (size_t)M.head_rows * 4 > B1.size) {
            fprintf(stderr, "bitnet_kria: %s is %zu bytes, the head sums need %zu\n",
                    B1.dev, B1.size, OFF_HEADSUM + (size_t)M.head_rows * 4);
            return 2;
        }
        R.hts = M.head_t_scale;
    }

    int memfd = open("/dev/mem", O_RDWR | O_SYNC);
    if (memfd < 0) die("open /dev/mem (run with sudo)");
    for (unsigned i = 0; i < E; i++) {
        gpio_addr[i] = gpio_base + i * stride;
        dma_addr[i] = dma_base + i * stride;
        gpio[i] = map_phys(memfd, gpio_addr[i], PAGE);
        dmar[i] = map_phys(memfd, dma_addr[i], PAGE);
    }
    ggpio_a_addr = ga; ggpio_b_addr = gb; dma_addr[GDMA] = gd;
    ggpio_a = map_phys(memfd, ga, PAGE);
    ggpio_b = map_phys(memfd, gb, PAGE);
    dmar[GDMA] = map_phys(memfd, gd, PAGE);

    snprintf(path, sizeof path, "%s/%s", dir, M.model_file);
    struct stat mst;
    if (stat(path, &mst)) die(path);
    M.model_bytes = (size_t)mst.st_size;
    size_t need0 = (size_t)M.mat[M.layers - 1][PROJ_DOWN].offset + (size_t)M.mat[M.layers - 1][PROJ_DOWN].n * M.mat[M.layers - 1][PROJ_DOWN].beats * 16;
    if (need0 > B0.size) { fprintf(stderr, "bitnet_kria: %s is %zu bytes, need %zu\n", B0.dev, B0.size, need0); return 2; }
    double t_load0 = now_ms();
    {
        int fd = open(path, O_RDONLY);
        if (fd < 0) die(path);
        const size_t chunk = 4u << 20;
        uint8_t *stage = xmalloc(chunk);
        size_t off = 0;
        for (;;) {
            ssize_t got = read(fd, stage, chunk);
            if (got < 0) die(path);
            if (got == 0) break;
            copy_in(B0.va + off, stage, (size_t)got);
            off += (size_t)got;
        }
        close(fd);
        if (off != M.model_bytes) { fprintf(stderr, "bitnet_kria: read %zu of %zu bytes of %s\n", off, M.model_bytes, M.model_file); return 2; }
        if (verify_weights) {
            int fd2 = open(path, O_RDONLY);
            size_t o = 0, bad = 0;
            for (;;) {
                ssize_t got = read(fd2, stage, chunk);
                if (got <= 0) break;
                if (memcmp((const void *)(B0.cva + o), stage, (size_t)got)) bad++;
                o += (size_t)got;
            }
            close(fd2);
            printf("weights verified against the file: %s\n", bad ? "DIFFER" : "identical");
            if (bad) return 2;
        }
        free(stage);
    }
    double load_s = (now_ms() - t_load0) / 1e3;

    double head_load_s = 0;
    if (R.head_fabric) {
        snprintf(path, sizeof path, "%s/%s", dir, M.head_file);
        head_load_s = load_buf(&B2, path, M.head_t_bytes, verify_weights);
    }

    double lock_s = 0;
    if (lock_head) {
        double tl = now_ms();
        size_t hb = (size_t)M.vocab * M.hidden, sb = (size_t)M.vocab * 4;
        int bad = mlock(M.head_w, hb) || mlock(M.head_scale, sb) || mlock(norms, norms_bytes);
        lock_s = (now_ms() - tl) / 1e3;
        if (bad) { fprintf(stderr, "bitnet_kria: mlock failed (%s); carrying on unlocked\n", strerror(errno)); lock_head = 0; }
        else if (!quiet) printf("  locked %.0f MB of head weights, scales and norms into RAM in %.2f s\n",
                                (double)(hb + sb + norms_bytes) / 1e6, lock_s);
    }

    cache_line_size();
    pool_start(threads);
    side_start(threads > 1);
    if (gbatch > 1 && cache_dtype != CACHE_FAB) {
        fprintf(stderr, "bitnet_kria: --gen-batch over 1 wants --cache-dtype fab, which is the one cache "
                        "that keeps a sequence's keys and values where the engines read them\n");
        return 2;
    }
    R.nseq = (int)gbatch;
    alloc_run(ctx, cache_dtype);
    for (unsigned i = 0; i <= E; i++) if (dma_reset(i)) return 3;
    for (unsigned i = 0; i < E; i++) wr(gpio[i], GPIO_DATA, 0);
    wr(ggpio_a, GPIO_DATA, 0);

    if (!quiet) {
        printf("BitNet b1.58 on the KV260: %d layers, hidden %d, intermediate %d, %d q heads over %d kv heads of %d, vocab %d, %s gate\n",
               M.layers, M.hidden, M.inter, M.n_q, M.n_kv, M.head_dim, M.vocab,
               M.silu ? "silu on the A53s" : "relu2 in the glue");
        printf("  %s phys 0x%010" PRIx64 " size %zu: %s %zu bytes read in %.2f s (%.0f MB/s)\n",
               B0.dev, B0.phys, B0.size, M.model_file, M.model_bytes, load_s, M.model_bytes / load_s / 1e6);
        printf("  %s phys 0x%010" PRIx64 " size %zu: activations, sums and glue beats need %u bytes\n", B1.dev, B1.phys, B1.size, NEED1);
        if (R.head_fabric)
            printf("  %s phys 0x%010" PRIx64 " size %zu: %s %zu bytes read in %.2f s (%.0f MB/s),"
                   " %u neurons of %u beats an engine; the head sums take %zu bytes of %s at 0x%06x;"
                   " shortlist K = %d\n",
                   B2.dev, B2.phys, B2.size, M.head_file, M.head_t_bytes, head_load_s,
                   M.head_t_bytes / head_load_s / 1e6, M.head_t_npe, M.head_t_beats,
                   (size_t)M.vocab * 4, B1.dev, OFF_HEADSUM, R.head_k);
        printf("  engines at 0x%010" PRIx64 " + i*0x%" PRIx64 ", DMAs at 0x%010" PRIx64 "; glue GPIO 0x%010" PRIx64 " params 0x%010" PRIx64 " DMA 0x%010" PRIx64 "\n",
               gpio_base, stride, dma_base, ga, gb, gd);
        {
            const char *cn = cache_dtype == CACHE_BF16 ? "bf16" : cache_dtype == CACHE_I8 ? "int8"
                             : cache_dtype == CACHE_FX ? "int8, fixed-point attention"
                             : cache_dtype == CACHE_FAB ? "int8 blocks, attention in the fabric" : "float32";
            double bytes = cache_dtype == CACHE_BF16 ? 2.0
                           : cache_dtype == CACHE_I8 || cache_dtype == CACHE_FX || cache_dtype == CACHE_FAB ? 1.0 : 4.0;
            double mb = (double)M.layers * M.n_kv * ctx * M.head_dim * bytes * 2 / 1e6;
            if (cache_dtype == CACHE_F32)
                mb = (double)M.layers * (R.kt_layer + R.v_layer) * 4 / 1e6;
            printf("  KV cache %s, context %d: %.0f MB; %d threads\n", cn, ctx, mb, POOL.n);
        }
        uint64_t lb = 0;
        for (int p = 0; p < NPROJ; p++) lb += (uint64_t)M.mat[0][p].n * M.mat[0][p].beats * BEAT_BYTES;
        printf("  a layer streams %llu weight bytes (%s, %u weights a beat): qkv %u neurons of %u beats, o %u, "
               "gate+up %u, down %u of %u (a quarter an engine)\n",
               (unsigned long long)lb, M.enc, M.wpb,
               M.mat[0][PROJ_Q].n + M.mat[0][PROJ_K].n + M.mat[0][PROJ_V].n, M.mat[0][PROJ_Q].beats,
               M.mat[0][PROJ_O].n, M.mat[0][PROJ_GATE].n + M.mat[0][PROJ_UP].n,
               M.mat[0][PROJ_DOWN].n, M.mat[0][PROJ_DOWN].beats);
        printf("  a token streams %llu weight bytes over %d layers; the prompt runs %u positions a group\n",
               (unsigned long long)lb * (unsigned)M.layers, M.layers, pbatch);
        printf("  setup %.2f s\n", (now_ms() - t_start) / 1e3);
        fflush(stdout);
    }

    int rc = 0;

    if (stage_file) {
        struct refstages st;
        if (refstages_open(stage_file, &st)) return 2;
        size_t ntok = 0;
        const int32_t *toks = refstages_i32(&st, "tokens", &ntok);
        if (!toks) { fprintf(stderr, "bitnet_kria: the dump has no tokens array\n"); return 2; }
        int nl = check_layers > 0 && check_layers < M.layers ? check_layers : M.layers;
        int whole = nl == M.layers;
        alloc_sink();
        SINK.active = 1;
        printf("stage check against %s: %zu positions, %d of %d layers\n", stage_file, ntok, nl, M.layers);
        int fail = 0;
        const int H = M.hidden, I = M.inter, KV = M.kv_size;
        for (size_t t = 0; t < ntok; t++) {
            printf("position %zu (token %d):\n", t, toks[t]);
            if (forward(toks[t], (int)t, nl)) return 3;
            R.pos = (int)t + 1;
            cmp_f32(&st, "l0_attn_in", t * H, SINK.l0_attn_in, H, 1e-6, 1e-5, 0.9999999, &fail);
            cmp_i8(&st, "l0_xq", t * H, SINK.l0_xq, H, &fail);
            cmp_i32(&st, "l0_qi", t * H, SINK.l0_qi, H, &fail);
            cmp_i32(&st, "l0_ki", t * KV, SINK.l0_ki, KV, &fail);
            cmp_i32(&st, "l0_vi", t * KV, SINK.l0_vi, KV, &fail);
            cmp_f32(&st, "l0_q", t * H, SINK.l0_q, H, 1e-6, 1e-5, 0.9999999, &fail);
            cmp_f32(&st, "l0_k", t * KV, SINK.l0_k, KV, 1e-6, 1e-5, 0.9999999, &fail);
            cmp_f32(&st, "l0_v", t * KV, SINK.l0_v, KV, 1e-6, 1e-5, 0.9999999, &fail);
            cmp_f32(&st, "l0_attn_out", t * H, SINK.l0_attn_out, H, 1e-5, 1e-4, 0.9999999, &fail);
            cmp_f32(&st, "l0_attn_sub", t * H, SINK.l0_attn_sub, H, 1e-5, 1e-4, 0.9999999, &fail);
            cmp_i8(&st, "l0_aq", t * H, SINK.l0_aq, H, &fail);
            cmp_i32(&st, "l0_oi", t * H, SINK.l0_oi, H, &fail);
            cmp_f32(&st, "l0_o", t * H, SINK.l0_o, H, 1e-5, 1e-4, 0.9999999, &fail);
            cmp_f32(&st, "l0_mlp_in", t * H, SINK.l0_mlp_in, H, 1e-5, 1e-4, 0.9999999, &fail);
            cmp_i8(&st, "l0_x2q", t * H, SINK.l0_x2q, H, &fail);
            cmp_i32(&st, "l0_g", t * I, SINK.l0_g, I, &fail);
            cmp_i32(&st, "l0_u", t * I, SINK.l0_u, I, &fail);
            cmp_i32(&st, "l0_shifts", t * 8, SINK.l0_shifts, 8, &fail);
            cmp_i16(&st, "l0_h", t * I, SINK.l0_h, I, &fail);
            cmp_i8(&st, "l0_q8", t * I, SINK.l0_q8, I, &fail);
            cmp_i32(&st, "l0_d", t * H, SINK.l0_d, H, &fail);
            cmp_f32(&st, "l0_ffn", t * H, SINK.l0_ffn, H, 1e-5, 1e-4, 0.9999999, &fail);
            {
                size_t have = 0;
                const float *hid = refstages_f32(&st, "hidden", &have);
                if (hid) {
                    double worst = 1.0;
                    int worst_l = 0;
                    printf("  hidden cosine against the reference, layer by layer:");
                    for (int l = 0; l <= nl; l++) {
                        size_t from = ((size_t)l * ntok + t) * H;
                        struct check_stats c = { (size_t)H, 0, 0, 0, 0, 0, 0 };
                        double maxref = 0;
                        for (int i = 0; i < H; i++) {
                            double a = SINK.hidden[(size_t)l * H + i], b = hid[from + i], d = fabs(a - b);
                            if (d > c.maxabs) c.maxabs = d;
                            if (fabs(b) > maxref) maxref = fabs(b);
                            c.dot += a * b; c.na += a * a; c.nb += b * b;
                        }
                        double cos = c.dot / sqrt(c.na * c.nb);
                        if (cos < worst) { worst = cos; worst_l = l; }
                        if (l % 5 == 0 || l == nl) printf(" %d:%.6f", l, cos);
                        if (l == nl)
                            printf("\n  hidden[%d] max|diff| %.4g of max|ref| %.4g; worst cosine %.9f at layer %d\n",
                                   l, c.maxabs, maxref, worst, worst_l);
                    }
                    if (worst < 0.999) fail = 1;
                }
            }
            if (whole) {
                cmp_f32(&st, "final_norm", t * H, R.fin, H, 1e-4, 1e-4, 0.998, &fail);
                int am = head_argmax_arm();
                int am_fab = R.head_fabric ? head_argmax_fabric() : am;
                size_t have = 0;
                const int32_t *ref_am = refstages_i32(&st, "argmax", &have);
                const float *ref_lg = refstages_f32(&st, "logits", &have);
                if (ref_lg) {
                    struct check_stats c = { (size_t)M.vocab, 0, 0, 0, 0, 0, 0 };
                    for (int i = 0; i < M.vocab; i++) {
                        double a = R.logits[i], b = ref_lg[t * M.vocab + i], d = fabs(a - b);
                        if (d > c.maxabs) c.maxabs = d;
                        c.dot += a * b; c.na += a * a; c.nb += b * b;
                    }
                    double cos = c.dot / sqrt(c.na * c.nb);
                    printf("  %-14s %6d float   max|diff| %.4g   cosine %.9f\n", "logits", M.vocab, c.maxabs, cos);
                    if (cos < 0.99) fail = 1;
                }
                if (ref_am) {
                    printf("  %-14s argmax %d, reference %d  %s\n", "top-1", am, ref_am[t], am == ref_am[t] ? "same" : "[DIFFERS]");
                    if (am != ref_am[t]) fail = 1;
                }
                if (R.head_fabric) {
                    printf("  %-14s fabric argmax %d, exact head %d  %s\n", "two-stage", am_fab, am,
                           am_fab == am ? "same" : "[DIFFERS]");
                    if (am_fab != am) fail = 1;
                }
            }
            fflush(stdout);
        }
        refstages_close(&st);
        printf("stage check: %s\n", fail ? "SOME STAGES DIFFERED"
               : "every integer stage exact, every float stage above its cosine floor, every top-1 the reference's");
        rc = fail ? 1 : 0;
        SINK.active = 0;
        if (timing) goto report;
        goto done;
    }

    {
        char *line = 0;
        size_t cap = 0;
        while (getline(&line, &cap, stdin) > 0) {
            int ids[8192], n = 0, cut[BMAX], ncut = 0;
            int run_new = max_new;
            {
                char *p = line;
                while (*p == ' ' || *p == '\t') p++;
                if (*p == '!' && p[1] >= '0' && p[1] <= '9') {
                    long v = strtol(p + 1, &p, 10);
                    if (v > 0 && v < run_new) run_new = (int)v;
                    while (*p == ' ' || *p == '\t') p++;
                    memmove(line, p, strlen(p) + 1);
                }
            }
            for (char *p = line; *p && n < 8192;) {
                if (*p == ';') { if (ncut < (int)BMAX) cut[ncut++] = n; p++; continue; }
                if (*p >= '0' && *p <= '9') { ids[n++] = (int)strtol(p, &p, 10); continue; }
                if (*p == '-' && p[1] >= '0' && p[1] <= '9') { ids[n++] = (int)strtol(p, &p, 10); continue; }
                p++;
            }
            if (n == 0) continue;
            memset(&R.t, 0, sizeof R.t);
#ifdef POOL_TRACE
            memset(PFS, 0, sizeof PFS);
            memset(PF_BAR, 0, sizeof PF_BAR);
            memset(PF_LOOP, 0, sizeof PF_LOOP);
            SCAN_N = 0;
            SCAN_LAST = 0;
            PF_NSLOW = 0;
            PF_FWD = 0;
            for (int i = 0; i <= POOL.n; i++) pf_sched(i < POOL.n ? PF_TID[i] : PF_SIDE_TID, PF_S0[i]);
            pf_cpustat(PF_C0);
#endif
            double t0 = now_ms();
            const int most = R.nseq < 1 ? 1 : R.nseq;
            int S = (ncut == 0) ? (GEN_FILL ? most : 1) : (ncut + 1);
            if (S > most) S = most;
            const int H = M.hidden;
            int spos[BMAX], stok[BMAX], sdone[BMAX], smade[BMAX];
            int poff[BMAX], plen[BMAX];
            R.pos = 0;
            int last = -1;
            for (int s = 0; s < S; s++) {
                if (ncut == 0) { poff[s] = 0; plen[s] = n; }
                else {
                    const int start = s == 0 ? 0 : cut[s - 1];
                    const int end = s < ncut ? cut[s] : n;
                    poff[s] = start;
                    plen[s] = (s <= ncut && end > start) ? end - start : 0;
                }
                if (plen[s] == 0) {
                    fprintf(stderr, "bitnet_kria: prompt %d of the %d on this line is empty; they are "
                                    "separated by a semicolon and there may be up to %d\n", s, S, most);
                    return 2;
                }
            }
            int npt_all = 0;
            for (int s = 0; s < S; s++) {
                const int *pids = ids + poff[s];
                const int pn = plen[s];
                R.cur_seq = s;
                R.pos = 0;
                for (int i = 0; i < pn && R.pos < ctx; ) {
                    int nb = pn - i;
                    if (nb > (int)pbatch) nb = (int)pbatch;
                    if (nb > ctx - R.pos) nb = ctx - R.pos;
                    if (forward_n(pids + i, R.pos, nb, M.layers)) return 3;
                    R.pos += nb;
                    i += nb;
                }
                npt_all += R.pos;
                spos[s] = R.pos;
                sdone[s] = 0;
                smade[s] = 0;
                int nxt = head_argmax();
                if (temp > 0) nxt = sample_token(temp, top_p, &seed);
                stok[s] = nxt;
            }
            double t_prompt = now_ms() - t0;
            double t1 = now_ms();
            int made = 0, alive = S;
            for (int i = 0; i < run_new && alive > 0; i++) {
                int slot_of[BMAX], slot_tok[BMAX], nb = 0;
                for (int s = 0; s < S; s++) {
                    if (sdone[s]) continue;
                    if (S > 1) printf("%d:%d\n", s, stok[s]); else printf("%d\n", stok[s]);
                    last = stok[s];
                    smade[s]++;
                    made++;
                    if ((!ignore_eos && (stok[s] == 128001 || stok[s] == 128009)) || spos[s] >= ctx) {
                        sdone[s] = 1;
                        alive--;
                        continue;
                    }
                    slot_of[nb] = s;
                    R.slot_pos[nb] = spos[s];
                    R.slot_seq[nb] = s;
                    slot_tok[nb] = stok[s];
                    nb++;
                }
                fflush(stdout);
                if (nb == 0) break;
                if (forward_slots(slot_tok, nb, M.layers)) return 3;
                for (int b = 0; b < nb; b++) {
                    const int s = slot_of[b];
                    spos[s]++;
                    memcpy(R.fin, R.finb + (size_t)b * H, sizeof(float) * (size_t)H);
                    int nxt = head_argmax();
                    if (temp > 0) nxt = sample_token(temp, top_p, &seed);
                    stok[s] = nxt;
                }
            }
            double t_gen = now_ms() - t1;
            (void)last;
            printf("done %d %.1f\n", made, now_ms() - t0);
            printf("prompt %d tokens %.1f ms (%.2f tok/s), generated %d %.1f ms (%.2f tok/s)\n",
                   npt_all, t_prompt, npt_all / (t_prompt / 1e3),
                   made, t_gen, made > 0 ? made / (t_gen / 1e3) : 0.0);
            if (S > 1) {
                printf("sequences %d, each", S);
                for (int s = 0; s < S; s++) printf(" %d", smade[s]);
                printf(" tokens; the prompt was run once a sequence\n");
            }
            fflush(stdout);
            if (timing) print_timing();
        }
        free(line);
    }
    goto done;

report:
    if (timing) print_timing();

done:
    for (unsigned i = 0; i < E; i++) wr(gpio[i], GPIO_DATA, 0);
    wr(ggpio_a, GPIO_DATA, 0);
    side_stop();
    pool_stop();
    return rc;
}
