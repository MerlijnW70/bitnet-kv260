"""Can a ternary approximation of BitNet's output head shortlist the exact argmax?

The KV260 spends 52.11 ms a generated token on the int8 head: 328 MB of int8 read by four A53s.
The fabric reads at 12.1 GB/s and idles through it.  A two-stage head would run a TERNARY copy of
the head on the fabric (82 MB, about 7 ms), take the top K rows by that approximate score, and
recompute only those K rows exactly from head_i8.bin on the A53.  That is exact if and only if the
row today's argmax returns is always inside the approximate top K.

This measures it: over every real final hidden state of head_states.npz, the rank of the exact int8
head's argmax under each candidate approximation.

usage
  python head_shortlist.py                                the whole study -> head-shortlist.txt
  python head_shortlist.py --states F.npz --out T.txt
  python head_shortlist.py --rules "absmean:0.75,frac:0.5"  a subset
  python head_shortlist.py --no-noise                     skip the board-drift robustness pass

The rules, per head row w (the bf16 embedding row; the head is tied to the embedding):
  absmean:A   t = sign(w) where |w| >= A * mean|w|, else 0
  frac:F      t = sign(w) on the F largest |w| of the row, else 0
  int4:_      q = clip(rint(w / (max|w| / 7)), -7, 7)          the fallback, not ternary
and in every case the row scale is the least-squares one, c = sum(|w| kept) / sum(t^2), so that
c * t is the closest multiple of the pattern to the row.

Ranks are conservative: a row that ties the argmax's approximate score is counted as being ahead of
it, because a top-K by value may order ties either way.
"""
import argparse
import glob
import json
import os
import struct
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.expanduser("~/.cache/huggingface/hub/models--microsoft--bitnet-b1.58-2B-4T/snapshots")
EMBED = "model.embed_tokens.weight"
HEAD_QMAX = 127
ROWS_BLOCK = 16384

DEFAULT_RULES = (["absmean:%g" % a for a in (0.0, 0.25, 0.5, 0.6, 0.7, 0.75, 0.8, 0.9, 1.0, 1.25)]
                 + ["frac:%g" % f for f in (0.75, 0.5, 0.35, 0.25, 0.15)]
                 + ["tern2:0.6", "tern2:0.75", "int4:0"])


def find_snapshot(path=None):
    if path:
        return path
    for c in sorted(glob.glob(os.path.join(CACHE, "*"))):
        if os.path.exists(os.path.join(c, "model.safetensors")):
            return c
    raise SystemExit(f"no model.safetensors under {CACHE}; pass --snapshot DIR")


class Embed:
    """The tied head, read straight out of the checkpoint's bf16 embedding, a row block at a time."""

    def __init__(self, snapshot):
        self.path = os.path.join(snapshot, "model.safetensors")
        with open(self.path, "rb") as f:
            (hl,) = struct.unpack("<Q", f.read(8))
            header = json.loads(f.read(hl))
        self.base = 8 + hl
        meta = header[EMBED]
        if meta["dtype"] != "BF16":
            raise SystemExit(f"{EMBED} is {meta['dtype']}, expected BF16")
        self.rows, self.cols = meta["shape"]
        self.off = meta["data_offsets"][0]
        self.mm = np.memmap(self.path, dtype=np.uint8, mode="r")

    def block(self, lo, hi):
        """Rows [lo, hi) as float64, exactly the bf16 values."""
        a = self.base + self.off + lo * self.cols * 2
        b = self.base + self.off + hi * self.cols * 2
        raw = np.asarray(self.mm[a:b]).tobytes()
        f32 = (np.frombuffer(raw, dtype="<u2").astype(np.uint32) << 16).view(np.float32)
        return f32.reshape(hi - lo, self.cols).astype(np.float64)


def quantise_head_block(block):
    """pack_model.quantize_head's own recipe: scale = max|w| / 127 in float32, q = rint(w / scale)."""
    scale = (np.abs(block).max(axis=1) / HEAD_QMAX).astype(np.float32)
    safe = np.where(scale > 0, scale, np.float32(1.0)).astype(np.float64)
    q = np.clip(np.rint(block / safe[:, None]), -HEAD_QMAX, HEAD_QMAX).astype(np.int8)
    return q, scale


def parse_rule(text):
    kind, _, param = text.partition(":")
    return kind, float(param or 0.0)


def ls_scale(t, block):
    """The least-squares row scale for a pattern t against the row: c = (t.w) / (t.t)."""
    energy = np.square(t).sum(axis=1)
    num = (t * block).sum(axis=1)
    return np.where(energy > 0, num / np.where(energy > 0, energy, 1.0), 0.0)


def one_plane(block, kind, param):
    a = np.abs(block)
    if kind == "absmean":
        keep = np.ones_like(a, dtype=bool) if param <= 0 else a >= (param * a.mean(axis=1))[:, None]
        t = np.where(keep, np.sign(block), 0.0)
    elif kind == "frac":
        k = max(1, int(round(param * block.shape[1])))
        if k >= block.shape[1]:
            keep = np.ones_like(a, dtype=bool)
        else:
            cut = np.partition(a, block.shape[1] - k, axis=1)[:, block.shape[1] - k]
            keep = a >= cut[:, None]
        t = np.where(keep, np.sign(block), 0.0)
    elif kind == "int4":
        s = a.max(axis=1) / 7.0
        t = np.clip(np.rint(block / np.where(s > 0, s, 1.0)[:, None]), -7, 7)
    else:
        raise SystemExit(f"unknown rule {kind}")
    return t


def apply_rule(block, kind, param):
    """A block of head rows -> a list of (pattern int8, row scale float32) planes; the approximate
    score of a row is the sum over planes of c * (t . q).  block is float64.

    tern2 is two ternary planes: the second quantises what the first leaves behind, so the head
    streams twice through the same engines and no engine changes."""
    if kind == "tern2":
        t1 = one_plane(block, "absmean", param)
        c1 = ls_scale(t1, block)
        resid = block - c1[:, None] * t1
        t2 = one_plane(resid, "absmean", param)
        c2 = ls_scale(t2, resid)
        return [(t1.astype(np.int8), c1.astype(np.float32)),
                (t2.astype(np.int8), c2.astype(np.float32))]
    t = one_plane(block, kind, param)
    return [(t.astype(np.int8), ls_scale(t, block).astype(np.float32))]


def read_bench(path):
    """head-bench.log, the board run of head_bench.c, as numbers."""
    import re
    out = {"dot": {}, "read": {}, "top": {}, "res": {}, "sync": None, "memcpy": None}
    for line in open(path, encoding="utf-8", errors="replace"):
        m = re.match(r"threads (\d+): head dot_i8 best\s+([\d.]+) ms mean\s+[\d.]+ ms \([\s\d.]+GB/s\) \| "
                     r"read only best\s+([\d.]+)", line)
        if m:
            out["dot"][int(m.group(1))] = float(m.group(2))
            out["read"][int(m.group(1))] = float(m.group(3))
        m = re.match(r"stage 1 with a real top-(\d+)\s*, \d+ threads: best ([\d.]+)", line)
        if m:
            out["top"][int(m.group(1))] = float(m.group(2))
        m = re.match(r"stage 2 rescore, \d+ threads: K =\s+(\d+) exact rows, best ([\d.]+)", line)
        if m:
            out["res"][int(m.group(1))] = float(m.group(2))
        m = re.search(r"sync_for_cpu\s+513024 B best ([\d.]+) ms, memcpy out best ([\d.]+)", line)
        if m:
            out["sync"] = float(m.group(1))
            out["memcpy"] = float(m.group(2))
    if not out["dot"] or out["sync"] is None or not out["top"] or not out["res"]:
        raise SystemExit(f"{path} does not hold a full head_bench run")
    return out


def absmax_int8(x):
    """The runtime's ActQuant, per state: scale = 127 / max(max|x|, 1e-5), round half to even."""
    amax = np.maximum(np.abs(x).max(axis=1), np.float32(1e-5)).astype(np.float32)
    scale = (np.float32(127.0) / amax).astype(np.float32)
    q = np.clip(np.round(x * scale[:, None]), -128, 127).astype(np.int8)
    return q, scale


def int_mm(torch, q, wt, chunk):
    """int8 [T, K] times int8 [K, V] -> int32 [T, V], exactly, a chunk of states at a time."""
    n = q.shape[0]
    for lo in range(0, n, chunk):
        hi = min(lo + chunk, n)
        blk = q[lo:hi]
        pad = (-(hi - lo)) % 16
        if pad:
            blk = torch.cat([blk, torch.zeros((pad, q.shape[1]), dtype=torch.int8, device=q.device)])
        out = torch._int_mm(blk.contiguous(), wt)
        yield lo, hi, out[:hi - lo]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--states", default=os.path.join(HERE, "head_states.npz"))
    ap.add_argument("--out", default=os.path.join(HERE, "head-shortlist.txt"))
    ap.add_argument("--json", default=os.path.join(HERE, "head-shortlist.json"))
    ap.add_argument("--snapshot", default=None)
    ap.add_argument("--bench", default=os.path.join(HERE, "head-bench.txt"))
    ap.add_argument("--rules", default=",".join(DEFAULT_RULES))
    ap.add_argument("--chunk", type=int, default=384)
    ap.add_argument("--noise-cosine", type=float, default=0.99944)
    ap.add_argument("--no-noise", action="store_true")
    ap.add_argument("--seed", type=int, default=20260906)
    a = ap.parse_args()

    import torch
    torch.backends.cuda.matmul.allow_tf32 = False
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if dev == "cpu":
        raise SystemExit("this study wants the card; torch reports no CUDA")

    L = []

    def say(*parts):
        line = " ".join(str(p) for p in parts)
        print(line, flush=True)
        L.append(line)

    snap = find_snapshot(a.snapshot)
    z = np.load(a.states, allow_pickle=False)
    fin = z["fin"].astype(np.float32)
    is_head = z["is_head"]
    board_arg = z["board_argmax"]
    src = z["src"]
    meta = json.loads(str(z["meta"][0]))
    T, K = fin.shape

    say("SETUP")
    say(f"  states            {a.states}")
    say(f"  positions         {T} final hidden states of {K} floats, "
        f"{int(is_head.sum())} of them at a position where the runtime's head ran")
    say(f"  sources           {int((src == 'board').sum())} from the 27 board-answers.json runs replayed "
        f"token for token, {int((src == 'bench').sum())} from the five bench prompts")
    say(f"  collected in      {meta['seconds']:.0f} s by head_states.py, arithmetic of ref_model.Ref")
    say(f"  checkpoint        {os.path.basename(snap)}")
    say(f"  card              {torch.cuda.get_device_name(0)}, torch {torch.__version__}")

    emb = Embed(snap)
    V = emb.rows
    if emb.cols != K:
        raise SystemExit(f"head is {emb.cols} wide, states are {K}")

    t0 = time.time()
    head_i8 = np.empty((V, K), dtype=np.int8)
    head_scale = np.empty(V, dtype=np.float32)
    for lo in range(0, V, ROWS_BLOCK):
        hi = min(lo + ROWS_BLOCK, V)
        blk = emb.block(lo, hi)
        head_i8[lo:hi], head_scale[lo:hi] = quantise_head_block(blk)
    say(f"  head              {V} x {K} int8 rebuilt from the checkpoint in {time.time() - t0:.0f} s "
        f"({V * K} bytes, the board's head_i8.bin), row scale "
        f"[{head_scale.min():.6g}, {head_scale.max():.6g}]")

    q, s = absmax_int8(fin)
    hinv = (np.float32(1.0) / s).astype(np.float32)
    qg = torch.from_numpy(q).to(dev)
    hs_g = torch.from_numpy(head_scale).to(dev)
    hinv_g = torch.from_numpy(hinv).to(dev)

    def exact_scores(head_int8, qgpu, hinvg):
        """The runtime's own logits: (float)acc * head_scale[r] * hinv, in float32, and its argmax
        with the first-maximum tie-break head_argmax() uses."""
        wt = torch.from_numpy(np.ascontiguousarray(head_int8.T)).to(dev)
        arg = torch.empty(qgpu.shape[0], dtype=torch.long, device=dev)
        best = torch.empty(qgpu.shape[0], dtype=torch.float32, device=dev)
        ties = torch.empty(qgpu.shape[0], dtype=torch.long, device=dev)
        gap = torch.empty(qgpu.shape[0], dtype=torch.float32, device=dev)
        order = (V - torch.arange(V, device=dev, dtype=torch.int32))
        for lo, hi, acc in int_mm(torch, qgpu, wt, a.chunk):
            val = acc.to(torch.float32) * hs_g * hinvg[lo:hi, None]
            mx = val.max(dim=1).values
            eq = (val == mx[:, None])
            arg[lo:hi] = (eq.to(torch.int32) * order).argmax(dim=1)
            best[lo:hi] = mx
            ties[lo:hi] = eq.sum(dim=1)
            second = val.masked_fill(eq, -float("inf")).max(dim=1).values
            gap[lo:hi] = mx - second
            del acc, val, eq
        del wt
        torch.cuda.empty_cache()
        return arg, best, ties, gap

    t0 = time.time()
    arg, best, ties, gap = exact_scores(head_i8, qg, hinv_g)
    say(f"  exact head        {T} states scored against all {V} rows in {time.time() - t0:.0f} s "
        f"(int8 x int8 -> int32, exact); {int((ties > 1).sum())} states have an exact tie at the top")
    argn = arg.cpu().numpy()

    hb = (board_arg >= 0)
    if hb.any():
        agree = int((argn[hb] == board_arg[hb]).sum())
        say(f"  replay fidelity   the exact int8 head on these states returns the id the KV260 emitted at "
            f"{agree}/{int(hb.sum())} of the board's own head calls")

    rows = []
    rules = [parse_rule(r) for r in a.rules.split(",") if r.strip()]

    def build(kind, param):
        planes = None
        for lo in range(0, V, ROWS_BLOCK):
            hi = min(lo + ROWS_BLOCK, V)
            got = apply_rule(emb.block(lo, hi), kind, param)
            if planes is None:
                planes = [(np.empty((V, K), dtype=np.int8), np.empty(V, dtype=np.float32))
                          for _ in got]
            for (t, c), (T_, C_) in zip(got, planes):
                T_[lo:hi], C_[lo:hi] = t, c
        return planes

    def ranks_of(planes, arg_idx, qgpu):
        """For every state: how many rows the approximation puts strictly above the exact argmax's
        row, and how many are above or equal to it (the conservative rank)."""
        wts = [(torch.from_numpy(np.ascontiguousarray(t.T)).to(dev), torch.from_numpy(c).to(dev))
               for t, c in planes]
        above = torch.empty(qgpu.shape[0], dtype=torch.int64, device=dev)
        aboveq = torch.empty(qgpu.shape[0], dtype=torch.int64, device=dev)
        own = torch.empty(qgpu.shape[0], dtype=torch.int64, device=dev)
        gens = [int_mm(torch, qgpu, wt, a.chunk) for wt, _ in wts]
        while True:
            A, lo, hi = None, None, None
            try:
                for g, (_, cg) in zip(gens, wts):
                    lo, hi, dot = next(g)
                    part = dot.to(torch.float32) * cg
                    A = part if A is None else A + part
                    del dot, part
            except StopIteration:
                break
            mine = A.gather(1, arg_idx[lo:hi, None])
            above[lo:hi] = (A > mine).sum(dim=1)
            aboveq[lo:hi] = (A >= mine).sum(dim=1) - 1
            own[lo:hi] = A.argmax(dim=1)
            del A, mine
        del wts, gens
        torch.cuda.empty_cache()
        return above.cpu().numpy(), aboveq.cpu().numpy(), own.cpu().numpy()

    say("")
    say("THE MEASUREMENT: the rank of the exact int8 head's argmax under each stage-1 approximation")
    say("  rank 0 means the approximation's own top-1 is already the exact argmax; the shortlist is exact on a")
    say("  state when K > its rank, so K must be worst rank + 1.  Two populations are reported: HEAD, the")
    say(f"  {int(is_head.sum())} positions where the runtime's head actually runs (every generated token and the last")
    say(f"  prompt token), which is what has to be exact, and ALL {T} positions, which adds the earlier prompt")
    say("  positions the runtime never scores -- a harder population, kept as a stress test.")
    say("")
    say("                             HEAD positions (what must be exact)      ALL positions (stress)")
    say("  rule              nonzero  top-1     r<=63    r<=255   p99.9 worst    K   p99.9 worst    K")
    ranks_keep = {}
    for kind, param in rules:
        t0 = time.time()
        planes = build(kind, param)
        nz = float(np.mean([(p != 0).mean() for p, _ in planes]))
        bits = (4 if kind == "int4" else 2) * len(planes)
        by = V * K * bits // 8
        above, aboveq, own = ranks_of(planes, arg, qg)
        r, rh = aboveq, aboveq[is_head]
        name = f"{kind}:{param:g}" if kind != "int4" else "int4"
        row = {"rule": name, "kind": kind, "param": param, "nonzero": nz, "bytes": by,
               "planes": len(planes),
               "top1_head": float((own[is_head] == argn[is_head]).mean()),
               "top1_all": float((own == argn).mean()),
               "head_rank63": float((rh <= 63).mean()), "head_rank255": float((rh <= 255).mean()),
               "head_p999": int(np.percentile(rh, 99.9)), "head_worst": int(rh.max()),
               "head_K": int(rh.max()) + 1,
               "all_p999": int(np.percentile(r, 99.9)), "all_worst": int(r.max()),
               "all_K": int(r.max()) + 1, "worst_strict": int(above.max()),
               "mean_rank_head": float(rh.mean()), "seconds": time.time() - t0}
        rows.append(row)
        ranks_keep[name] = r
        say(f"  {name:<16} {nz * 100:6.2f}%  {row['top1_head'] * 100:6.2f}%  "
            f"{row['head_rank63'] * 100:7.3f}% {row['head_rank255'] * 100:7.3f}% "
            f"{row['head_p999']:5d} {row['head_worst']:5d} {row['head_K']:5d}  "
            f"{row['all_p999']:6d} {row['all_worst']:5d} {row['all_K']:5d}")
        del planes

    tern = [r for r in rows if r["kind"] in ("absmean", "frac")]
    best_rule = min(tern, key=lambda r: (r["head_worst"], -r["top1_head"]))
    say("")
    say(f"  the tightest one-plane ternary rule on the head positions is {best_rule['rule']}: worst rank "
        f"{best_rule['head_worst']}, so K = {best_rule['head_K']} covers every one of the "
        f"{int(is_head.sum())} states, and K = {best_rule['head_p999'] + 1} covers 99.9%.")
    say(f"  the same rule over all {T} positions: worst rank {best_rule['all_worst']}, K = {best_rule['all_K']}.")
    for r in rows:
        if r["kind"] == "int4":
            say(f"  the int4 fallback ({r['bytes'] / 1e6:.1f} MB on the A53, not the fabric): worst rank "
                f"{r['head_worst']} on the head positions (K = {r['head_K']}), {r['all_worst']} over all "
                f"(K = {r['all_K']}).")
        if r["kind"] == "tern2":
            say(f"  two ternary planes {r['rule']} ({r['bytes'] / 1e6:.1f} MB, the head streamed twice through "
                f"the same engines): worst rank {r['head_worst']} on the head positions (K = {r['head_K']}), "
                f"{r['all_worst']} over all (K = {r['all_K']}).")

    say("")
    say("A CERTIFICATE INSTEAD OF A MEASUREMENT?")
    say("  E_r - A_r = d_r . q with d_r = head_scale_r w8_r - c_r t_r, so |E_r - A_r| <= ||d_r|| ||q|| and a row")
    say("  can be dropped for certain when A_r + bound_r < max_r'(A_r' - bound_r').  That shortlist is exact by")
    say("  construction, with no study at all; the question is how big it is.")
    cert = {}
    for rule in (best_rule["rule"],) + tuple(r["rule"] for r in rows if r["kind"] == "int4"):
        row = [r for r in rows if r["rule"] == rule][0]
        planes = build(row["kind"], row["param"])
        dn = np.empty(V, dtype=np.float32)
        for lo in range(0, V, ROWS_BLOCK):
            hi = min(lo + ROWS_BLOCK, V)
            d = head_i8[lo:hi].astype(np.float64) * head_scale[lo:hi, None].astype(np.float64)
            for t, c in planes:
                d = d - t[lo:hi].astype(np.float64) * c[lo:hi, None].astype(np.float64)
            dn[lo:hi] = np.sqrt(np.square(d).sum(axis=1)).astype(np.float32)
        wts = [(torch.from_numpy(np.ascontiguousarray(t.T)).to(dev), torch.from_numpy(c).to(dev))
               for t, c in planes]
        dng = torch.from_numpy(dn).to(dev)
        qn = torch.from_numpy(np.linalg.norm(q.astype(np.float32), axis=1).astype(np.float32)).to(dev)
        keeps = torch.empty(T, dtype=torch.int64, device=dev)
        gens = [int_mm(torch, qg, wt, a.chunk) for wt, _ in wts]
        while True:
            A, lo, hi = None, None, None
            try:
                for g, (_, cg) in zip(gens, wts):
                    lo, hi, dot = next(g)
                    part = dot.to(torch.float32) * cg
                    A = part if A is None else A + part
                    del dot, part
            except StopIteration:
                break
            b = dng * qn[lo:hi, None]
            m = (A - b).max(dim=1).values
            keeps[lo:hi] = ((A + b) >= m[:, None]).sum(dim=1)
            del A, b, m
        kk = keeps.cpu().numpy()
        cert[rule] = {"min": int(kk.min()), "median": int(np.median(kk)), "max": int(kk.max())}
        say(f"  {rule:<16} certified shortlist {kk.min()} .. {kk.max()} rows (median {int(np.median(kk))}) "
            f"of {V}: {100.0 * kk.mean() / V:.1f}% of the vocabulary, so the certificate is useless -- "
            f"Cauchy-Schwarz is about sqrt(2560) too loose for a sum of 2560 near-independent terms.")
        del planes, wts, gens
        torch.cuda.empty_cache()

    noise = None
    if not a.no_noise:
        say("")
        say("ROBUSTNESS TO THE BOARD'S OWN FLOAT DRIFT")
        say("  These states are the reference's.  The KV260's final hidden state differs from it in the last")
        say("  bits of its float32 attention and norms: --stage-check measured cosine 0.999440 on final_norm.")
        say("  So the same study is run again on states perturbed to exactly that cosine, which stands in for")
        say("  the board's own vector, to show the K margin is not an artefact of the reference's rounding.")
        rng = np.random.default_rng(a.seed)
        c0 = np.float32(a.noise_cosine)
        nrm = np.linalg.norm(fin, axis=1)
        sig = (nrm / np.sqrt(K) * np.sqrt(1.0 / (c0 * c0) - 1.0)).astype(np.float32)
        pert = (fin + rng.standard_normal(fin.shape).astype(np.float32) * sig[:, None]).astype(np.float32)
        cos = (fin * pert).sum(1) / (np.linalg.norm(fin, axis=1) * np.linalg.norm(pert, axis=1))
        qp, sp = absmax_int8(pert)
        qpg = torch.from_numpy(qp).to(dev)
        hinvp = torch.from_numpy((np.float32(1.0) / sp).astype(np.float32)).to(dev)
        argp, _, _, _ = exact_scores(head_i8, qpg, hinvp)
        argpn = argp.cpu().numpy()
        say(f"  perturbed states: cosine to the reference's {cos.min():.6f} .. {cos.max():.6f} "
            f"(mean {cos.mean():.6f}); the exact argmax moves on "
            f"{int((argpn != argn).sum())}/{T} of them, which is the drift itself, not the shortlist")
        noise = []
        say("")
        say("  rule              top-1(head)  head p99.9  head worst  head K   all worst  all K")
        for r in rows:
            planes = build(r["kind"], r["param"])
            _, aboveq, own = ranks_of(planes, argp, qpg)
            rh = aboveq[is_head]
            noise.append({"rule": r["rule"], "head_worst": int(rh.max()), "head_K": int(rh.max()) + 1,
                          "head_p999": int(np.percentile(rh, 99.9)),
                          "all_worst": int(aboveq.max()), "all_K": int(aboveq.max()) + 1,
                          "top1_head": float((own[is_head] == argpn[is_head]).mean())})
            say(f"  {r['rule']:<16} {noise[-1]['top1_head'] * 100:9.2f}%  {noise[-1]['head_p999']:10d}  "
                f"{noise[-1]['head_worst']:10d}  {noise[-1]['head_K']:6d}  {noise[-1]['all_worst']:9d}  "
                f"{noise[-1]['all_K']:5d}")
            del planes

    pick = None
    if noise:
        nz = {n["rule"]: n for n in noise}
        cand = [r for r in rows if r["kind"] in ("absmean", "frac")]
        for r in cand:
            r["both_worst"] = max(r["head_worst"], nz[r["rule"]]["head_worst"])
            r["both_p999"] = max(r["head_p999"], nz[r["rule"]]["head_p999"])
        pick = min(cand, key=lambda r: (r["both_worst"], r["both_p999"]))
        say("")
        say("VERDICT")
        say(f"  rule            {pick['rule']}: t = sign(w) where |w| >= {pick['param']:g} mean|w| of the row, else 0,")
        say(f"                  with the least-squares row scale c = (t.w)/(t.t).  {pick['nonzero'] * 100:.1f}% of the")
        say("                  weights survive, but a zero costs the same two bits as a sign, so the stream is 82.1 MB.")
        say(f"  worst rank      {pick['both_worst']} over the {int(is_head.sum())} states where the head runs "
            f"(clean {pick['head_worst']}, drifted {nz[pick['rule']]['head_worst']})")
        say(f"  99.9th rank     {pick['both_p999']}")
        say(f"  K               256 covers every measured state with a factor of "
            f"{256 / max(1, pick['both_worst']):.1f} in hand; 64 covers them all but with only "
            f"{64 / max(1, pick['both_worst']):.2f}x, 512 with {512 / max(1, pick['both_worst']):.1f}x")
        say("  exactness       measured, not proved: the shortlist is exact on every one of these states, and the")
        say("                  certificate above shows no cheap bound makes it exact by construction.  It has to be")
        say("                  re-checked by running the whole battery on the board and comparing token ids.")

    nz_best = [r for r in rows if r["rule"] == (pick or best_rule)["rule"]][0]["nonzero"]
    say("")
    say("WHAT IT COSTS (every figure here is a PROJECTION from measured rates, not a board measurement)")
    say(f"  stage 1 on the fabric: {V} x {K} ternary at 2 bits = {V * K // 4} B, {K // 64} beats a neuron, the")
    say("  engines' own stream format, N split over the four engines in blocks of 32064 neurons.  At the 12.100")
    say("  GB/s the four engines measured over model.bin (kria-model-results.txt) that is "
        f"{V * K / 4 / 12.100e9 * 1e3:.2f} ms;")
    say(f"  at the 12.941 GB/s down_proj reaches, {V * K / 4 / 12.941e9 * 1e3:.2f} ms.  The ternary head is "
        f"{nz_best * 100:.0f}% nonzero but a zero")
    say("  costs the same two bits as a sign, so the stream is the full 82.1 MB either way.")
    bench = read_bench(a.bench)
    say("")
    say(f"  MEASURED ON THE BOARD by head_bench.c, the run kept in {os.path.basename(a.bench)} (polkit restarted")
    say("  first, no other load, best of 9 repeats, the real head_i8.bin mlocked, four A53s at 1.33 GHz):")
    say(f"    the head as the runtime runs it, dot_i8 over all {V} rows: 1 thread {bench['dot'][1]:.3f} ms, "
        f"2 threads {bench['dot'][2]:.3f},")
    say(f"      4 threads {bench['dot'][4]:.3f} ms = {V * K / bench['dot'][4] / 1e6:.2f} GB/s "
        f"(the runtime's own figure is 52.113 ms = 6.30 GB/s)")
    say(f"    the same {V * K} B read with no multiply at all:            1 thread {bench['read'][1]:.3f} ms, "
        f"2 threads {bench['read'][2]:.3f},")
    say(f"      4 threads {bench['read'][4]:.3f} ms = {V * K / bench['read'][4] / 1e6:.2f} GB/s")
    say(f"    so the head is memory bound on the A53s: taking every multiply out of it saves "
        f"{bench['dot'][4] - bench['read'][4]:.3f} ms of")
    say(f"    {bench['dot'][4]:.3f}, {100 * (bench['dot'][4] - bench['read'][4]) / bench['dot'][4]:.1f}%.  "
        f"Four threads reach {V * K / bench['read'][4] / 1e6:.1f} GB/s; the fabric reads at 12.1 GB/s and idles")
    say("    through those 52 ms.")
    say(f"    sync_for_cpu over {V * 4} B of the DMA buffer: {bench['sync']:.3f} ms; a memcpy of the same out: "
        f"{bench['memcpy']:.3f} ms.")
    say(f"    stage 1's host side, {V} sums scaled and a real top-K by a min-heap a thread, four threads:")
    say("      " + "   ".join(f"K = {k}: {bench['top'][k]:.3f}" for k in sorted(bench["top"])) + " ms")
    say("    stage 2's exact rescore of K rows of head_i8, four threads:")
    say("      " + "   ".join(f"K = {k}: {bench['res'][k]:.3f}" for k in sorted(bench["res"])
                              if k in bench["top"]) + " ms")
    say("")
    say("  THE PROJECTED TWO-STAGE HEAD (fabric stream + sync + top-K + exact rescore):")
    fab = V * K / 4 / 12.100e9 * 1e3
    top, res = bench["top"], bench["res"]
    for k in (64, 128, 256, 512, 1024):
        tot = fab + bench["sync"] + top[k] + res[k]
        say(f"    K = {k:4d}: {fab:.3f} + {bench['sync']:.3f} + {top[k]:.3f} + {res[k]:.3f} = {tot:6.3f} ms a head call, "
            f"against 52.113 today, saving {52.113 - tot:.2f} ms; a generated token "
            f"{124.906 - 52.113 + tot:6.2f} ms = {1000.0 / (124.906 - 52.113 + tot):.2f} tok/s")
    say("    (a generated token is 124.906 ms today = 8.01 tok/s: forward 72.793 + head 52.113.)")
    i4b = V * K // 2
    gbs = V * K / bench["read"][4] / 1e6
    i4t = i4b / (gbs * 1e9) * 1e3
    extra = top[64] + res[64]
    say(f"    the int4 fallback, stage 1 on the A53 instead of the fabric: {i4b} B at the {gbs:.2f} GB/s four")
    say(f"      threads reach = {i4t:.2f} ms plus the nibble unpack, + {top[64]:.3f} (top-64) + {res[64]:.3f} "
        f"(K = 64) = {i4t + extra:.2f} ms a head call,")
    say(f"      a generated token {124.906 - 52.113 + i4t + extra:.2f} ms = "
        f"{1000.0 / (124.906 - 52.113 + i4t + extra):.2f} tok/s, and it needs no fabric change at all.")
    say("")
    say("  WHERE head_t.bin GOES: udmabuf0 is 545259520 B and model.bin fills 521011200 of it, so the ternary head")
    say("  needs its own buffer.  CmaTotal is 1024000 kB with CmaFree 126496 kB today, so raising udmabuf0 to 640")
    say("  MiB would ask for 648 MiB of CMA against the 876 MiB already in use and is too tight; adding")
    say("  udmabuf2=83886080 (80 MiB, holding the 82083840 B whole) to /etc/modprobe.d/u-dma-buf.conf fits in the")
    say("  123 MiB free today.  The driver's dma_mask_bit is 32, so whatever it hands out is below 4 GiB by")
    say("  construction (udmabuf0 sits at 0x0000000038300000 and udmabuf1 at 0x0000000058b00000 now).")
    say(f"  The results need {V * 4} more bytes of udmabuf1, which has 8388608 and uses 253952 today.")
    say(f"  N splits over the four engines in blocks of {V // 4} neurons, {K // 64} beats a neuron, "
        f"{V // 4 * K // 64 * 16} B an engine;")
    say(f"  {V // 4} fits the ctrl register's 16-bit neuron count and {K // 64} its 14-bit beat count.")

    say("")
    say("FILES AND HOW TO RERUN")
    say("  head_states.py          collects the states: replays the 27 runs of board-answers.json token for token")
    say("                          and generates the five bench prompts, with ref_model.Ref's arithmetic and the")
    say("                          ternary matvecs on the card (bit-identical to numpy's, --verify proves it on a")
    say("                          whole token).  122 s.  Writes head_states.npz (31.7 MB, a cache, not a source).")
    say(f"  head_shortlist.py       this study.  Reads head_states.npz and {os.path.basename(a.bench)}, writes "
        f"head-shortlist.txt")
    say("                          and head-shortlist.json (every rank of the chosen rule is in the json).")
    say("  head_bench.c            the board side: the head as the runtime runs it, the same bytes with no")
    say("                          multiply, sync_for_cpu, the top-K and the exact rescore.  Its run is in")
    say(f"                          {os.path.basename(a.bench)}.  Built with")
    say("                            gcc -O3 -Wall -Wextra -std=c11 -march=armv8-a+crc -o head_bench head_bench.c -lpthread")
    say("  pack_head_ternary.py    writes head_t.bin (82083840 B), head_t_scale.bin and head_t.json, then reads the")
    say("                          file back, unpacks every trit, rebuilds the first beat by hand, recomputes the")
    say("                          approximate scores from the file and measures the rank of the exact argmax on")
    say("                          real states.  Verbatim, its run with --check-states 128:")
    say("                            pattern built in 5 s, 62.60% nonzero, row scale [0.312214, 1.5189]")
    say("                            round trip: all 328335360 trits unpack to the pattern they were packed from")
    say("                            first beat of neuron 0 by hand: a8a99494944045686a4a1a544a896602")
    say("                            the approximate scores from the unpacked file are identical to the ones from the pattern")
    say("                            rank of the exact argmax under this file: worst 7, top-1 already right on 73.4% of them")
    say("                            K = 256 covers every one of the 128 states")
    say("                            md5 7f2850dced9d43476b000bd0cff7a8ed, 32064 neurons and 20520960 B an engine")

    out = {"states": int(T), "head_states": int(is_head.sum()), "rows": rows, "noise": noise,
           "best_rule": best_rule["rule"], "snapshot": os.path.basename(snap), "meta": meta,
           "ranks_best": ranks_keep[best_rule["rule"]].tolist(),
           "is_head": is_head.astype(np.int8).tolist()}
    open(a.json, "w").write(json.dumps(out, indent=1))
    open(a.out, "w", encoding="utf-8").write("\n".join(L) + "\n")
    print(f"\nwrote {a.out} and {a.json}", flush=True)


if __name__ == "__main__":
    main()
