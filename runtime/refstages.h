#ifndef REFSTAGES_H
#define REFSTAGES_H

#include <fcntl.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#define REFSTAGES_MAGIC "REFSTG\x01\x00"
#define REFSTAGES_F32 0
#define REFSTAGES_I32 1
#define REFSTAGES_I16 2
#define REFSTAGES_I8 3

struct refstages_entry {
    char name[32];
    uint32_t dtype;
    uint32_t ndim;
    uint32_t dims[4];
    uint64_t offset;
    uint64_t nbytes;
};

struct refstages {
    const uint8_t *map;
    size_t size;
    uint32_t count;
    uint32_t data_start;
    const struct refstages_entry *entries;
};

static inline int refstages_open(const char *path, struct refstages *st)
{
    struct stat sb;
    int fd = open(path, O_RDONLY);
    if (fd < 0) {
        fprintf(stderr, "refstages: cannot open %s\n", path);
        return -1;
    }
    if (fstat(fd, &sb) != 0 || (size_t)sb.st_size < 16) {
        close(fd);
        return -1;
    }
    void *m = mmap(NULL, (size_t)sb.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);
    if (m == MAP_FAILED) {
        fprintf(stderr, "refstages: cannot mmap %s\n", path);
        return -1;
    }
    st->map = (const uint8_t *)m;
    st->size = (size_t)sb.st_size;
    if (memcmp(st->map, REFSTAGES_MAGIC, 8) != 0) {
        fprintf(stderr, "refstages: %s is not a stage dump\n", path);
        munmap(m, st->size);
        return -1;
    }
    memcpy(&st->count, st->map + 8, 4);
    memcpy(&st->data_start, st->map + 12, 4);
    if (16 + (size_t)st->count * 72 > st->size || st->data_start > st->size) {
        fprintf(stderr, "refstages: %s is truncated\n", path);
        munmap(m, st->size);
        return -1;
    }
    st->entries = (const struct refstages_entry *)(st->map + 16);
    return 0;
}

static inline void refstages_close(struct refstages *st)
{
    if (st->map) {
        munmap((void *)st->map, st->size);
        st->map = NULL;
    }
}

static inline const struct refstages_entry *refstages_entry(const struct refstages *st, const char *name)
{
    for (uint32_t i = 0; i < st->count; i++)
        if (strncmp(st->entries[i].name, name, 32) == 0)
            return &st->entries[i];
    return NULL;
}

static inline const void *refstages_get(const struct refstages *st, const char *name, uint32_t dtype, size_t *n)
{
    static const size_t width[4] = {4, 4, 2, 1};
    const struct refstages_entry *e = refstages_entry(st, name);
    if (!e || e->dtype != dtype)
        return NULL;
    if (st->data_start + e->offset + e->nbytes > st->size)
        return NULL;
    if (n)
        *n = (size_t)e->nbytes / width[dtype];
    return st->map + st->data_start + e->offset;
}

static inline const float *refstages_f32(const struct refstages *st, const char *name, size_t *n)
{
    return (const float *)refstages_get(st, name, REFSTAGES_F32, n);
}

static inline const int32_t *refstages_i32(const struct refstages *st, const char *name, size_t *n)
{
    return (const int32_t *)refstages_get(st, name, REFSTAGES_I32, n);
}

static inline const int16_t *refstages_i16(const struct refstages *st, const char *name, size_t *n)
{
    return (const int16_t *)refstages_get(st, name, REFSTAGES_I16, n);
}

static inline const int8_t *refstages_i8(const struct refstages *st, const char *name, size_t *n)
{
    return (const int8_t *)refstages_get(st, name, REFSTAGES_I8, n);
}

static inline long refstages_check_i32(const struct refstages *st, const char *name, size_t from,
                                       const int32_t *got, size_t n)
{
    size_t have = 0;
    const int32_t *want = refstages_i32(st, name, &have);
    if (!want || from + n > have) {
        fprintf(stderr, "stage-check %s: %s\n", name, want ? "shorter than asked" : "missing");
        return 0;
    }
    for (size_t i = 0; i < n; i++)
        if (got[i] != want[from + i]) {
            printf("stage-check %s[%zu + %zu]: got %d, reference %d\n", name, from, i, got[i], want[from + i]);
            return (long)i;
        }
    return -1;
}

static inline long refstages_check_f32(const struct refstages *st, const char *name, size_t from,
                                       const float *got, size_t n, float atol, float rtol)
{
    size_t have = 0;
    const float *want = refstages_f32(st, name, &have);
    if (!want || from + n > have) {
        fprintf(stderr, "stage-check %s: %s\n", name, want ? "shorter than asked" : "missing");
        return 0;
    }
    for (size_t i = 0; i < n; i++) {
        float w = want[from + i], d = got[i] - w;
        if (d < 0)
            d = -d;
        float lim = atol + rtol * (w < 0 ? -w : w);
        if (!(d <= lim)) {
            printf("stage-check %s[%zu + %zu]: got %.7g, reference %.7g, |diff| %.4g > %.4g\n",
                   name, from, i, (double)got[i], (double)w, (double)d, (double)lim);
            return (long)i;
        }
    }
    return -1;
}

#endif
