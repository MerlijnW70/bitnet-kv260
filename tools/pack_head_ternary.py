"""Write the ternary stage-1 head for the KV260's matvec engines, and prove the file reproduces it.

usage
  python pack_head_ternary.py OUT_DIR                       base3, absmean:0.6, K = 256, self-check
  python pack_head_ternary.py OUT_DIR --encoding 2bit       the old two-bit stream
  python pack_head_ternary.py OUT_DIR --rule absmean:0.8 --k 512
  python pack_head_ternary.py OUT_DIR --check-only          re-check a file already written
  python pack_head_ternary.py OUT_DIR --check-states 128    how many real states the check scores

what it writes into out_dir
  head3_t.bin       (base3, the default) 128256 x 2560 ternary in the engines' stream format: five
                    weights a byte in base 3, byte = sum of (w_j + 1) * 3^j for j = 0..4 lowest
                    power first, 80 weights = one 16-byte beat, 32 beats a neuron, no padding
                    (2560 = 32 * 80).  65667072 B.
  head_t.bin        (2bit) the same trits two bits a weight, 64 a beat, 40 beats a neuron.  82083840 B.
  head_t_scale.bin  float32 [128256], the per-row scale c the A53 multiplies the engine's int32 sum
                    by before the top-K.
  head_t.json       geometry, the encoding and the file it names, the rule, the engine split, md5s,
                    and what the self-check measured.

the self-check
  always: the file is read back, unpacked, and every unpacked row compared with the pattern in
  memory, and the first beat of the first neuron is rebuilt by hand from the trits.
  only with head_states.npz (real final hidden states -- a research artefact this repository does
  NOT ship): the approximate scores are recomputed from the unpacked file and compared with the
  ones the pattern gives directly, and for each state the rank of the exact int8 head's argmax
  under those scores is measured, which is the number the shortlist depends on. Without that file
  this part is skipped with a printed reason and head_t.json records states_skipped; the head is
  written and usable either way. The recorded result of that measurement on the reference board
  was a worst rank of 56 out of K = 256 over 2115 real states.
"""
import argparse
import hashlib
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from head_shortlist import Embed, apply_rule, absmax_int8, find_snapshot, quantise_head_block

ROWS_BLOCK = 8192
WEIGHTS_PER_BYTE = 4
WEIGHTS_PER_BEAT = 64
BEAT_BYTES = 16
ENGINES = 4
ENCODINGS = {
    "2bit": {"file": "head_t.bin", "per_byte": 4, "per_beat": 64,
             "text": "w+1 (0 = -1, 1 = 0, 2 = +1), four a byte lowest first, row-major per neuron"},
    "base3": {"file": "head3_t.bin", "per_byte": 5, "per_beat": 80,
              "text": "five weights a byte in base 3, byte = sum of (w_j + 1) * 3^j for j = 0..4 "
                      "(a byte of five zero weights is 121), row-major per neuron, each neuron "
                      "padded with zero weights to a whole 80-weight beat"},
}


def md5_of(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def beats_for(cols, enc):
    per = ENCODINGS[enc]["per_beat"]
    return (cols + per - 1) // per


def pack_engine(values, enc="2bit"):
    """uint8 [rows, K] of w+1 -> uint8 [rows, beats*16] in the fabric's stream order."""
    spec = ENCODINGS[enc]
    rows, cols = values.shape
    width = beats_for(cols, enc) * spec["per_beat"]
    if width != cols:
        pad = np.ones((rows, width - cols), dtype=np.uint8)
        values = np.concatenate([values, pad], axis=1)
    v = values.reshape(rows, width // spec["per_byte"], spec["per_byte"])
    if enc == "2bit":
        return np.ascontiguousarray(v[:, :, 0] | (v[:, :, 1] << 2) | (v[:, :, 2] << 4) | (v[:, :, 3] << 6))
    out = np.zeros((rows, width // spec["per_byte"]), dtype=np.uint8)
    for j in range(spec["per_byte"] - 1, -1, -1):
        out = (out * 3 + v[:, :, j]).astype(np.uint8)
    return np.ascontiguousarray(out)


def unpack_engine(packed, enc="2bit", cols=None):
    """uint8 [rows, beats*16] -> int8 [rows, cols] of -1, 0, +1 (padding trimmed)."""
    spec = ENCODINGS[enc]
    rows, bytes_per_row = packed.shape
    width = bytes_per_row * spec["per_byte"]
    out = np.empty((rows, width), dtype=np.int8)
    if enc == "2bit":
        for i in range(spec["per_byte"]):
            out[:, i::spec["per_byte"]] = ((packed >> (2 * i)) & 3).astype(np.int8) - 1
    else:
        v = packed.astype(np.uint16)
        for i in range(spec["per_byte"]):
            out[:, i::spec["per_byte"]] = (v % 3).astype(np.int8) - 1
            v = v // 3
    return out if cols is None or cols == width else np.ascontiguousarray(out[:, :cols])


def exact_head_scores(head_i8, head_scale, q):
    """The runtime's own logits up to the positive per-state factor: acc * head_scale, exactly."""
    try:
        import torch
        if torch.cuda.is_available():
            wt = torch.from_numpy(np.ascontiguousarray(head_i8.T)).cuda()
            qg = torch.from_numpy(q).cuda()
            hs = torch.from_numpy(head_scale).cuda()
            pad = (-q.shape[0]) % 16
            if pad:
                qg = torch.cat([qg, torch.zeros((pad, q.shape[1]), dtype=torch.int8, device="cuda")])
            acc = torch._int_mm(qg, wt)[:q.shape[0]]
            return (acc.to(torch.float32) * hs).cpu().numpy()
    except Exception:
        pass
    out = np.zeros((q.shape[0], head_i8.shape[0]), dtype=np.float64)
    for lo in range(0, head_i8.shape[0], ROWS_BLOCK):
        hi = min(lo + ROWS_BLOCK, head_i8.shape[0])
        acc = head_i8[lo:hi].astype(np.float64) @ q.astype(np.float64).T
        out[:, lo:hi] = (acc * head_scale[lo:hi, None].astype(np.float64)).T
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out", help="directory for the head stream (66 MB base3, 82 MB 2bit); not kept in the repository")
    ap.add_argument("--rule", default="absmean:0.6")
    ap.add_argument("--k", type=int, default=256)
    ap.add_argument("--snapshot", default=None)
    ap.add_argument("--states", default=os.path.join(HERE, "head_states.npz"))
    ap.add_argument("--check-states", type=int, default=64)
    ap.add_argument("--check-only", action="store_true")
    ap.add_argument("--encoding", default="base3", choices=sorted(ENCODINGS))
    a = ap.parse_args()

    kind, _, param = a.rule.partition(":")
    param = float(param or 0.0)
    os.makedirs(a.out, exist_ok=True)
    snap = find_snapshot(a.snapshot)
    emb = Embed(snap)
    V, K = emb.rows, emb.cols
    enc = a.encoding
    spec = ENCODINGS[enc]
    beats = beats_for(K, enc)
    row_bytes = beats * BEAT_BYTES
    bin_path = os.path.join(a.out, spec["file"])
    scale_path = os.path.join(a.out, "head_t_scale.bin")
    want_bytes = V * row_bytes

    print(f"checkpoint {os.path.basename(snap)}: head {V} x {K}, rule {a.rule}, K = {a.k}, "
          f"encoding {enc} -> {spec['file']} {want_bytes} B, {beats} beats a neuron, "
          f"{beats * spec['per_beat'] - K} padding weights", flush=True)

    pattern = np.empty((V, K), dtype=np.int8)
    scale = np.empty(V, dtype=np.float32)
    t0 = time.time()
    for lo in range(0, V, ROWS_BLOCK):
        hi = min(lo + ROWS_BLOCK, V)
        planes = apply_rule(emb.block(lo, hi), kind, param)
        if len(planes) != 1:
            raise SystemExit(f"{a.rule} makes {len(planes)} planes; this packer writes one")
        pattern[lo:hi], scale[lo:hi] = planes[0]
    if pattern.min() < -1 or pattern.max() > 1:
        raise SystemExit("the rule did not produce trits")
    print(f"  pattern built in {time.time() - t0:.0f} s, {float((pattern != 0).mean()) * 100:.2f}% nonzero, "
          f"row scale [{scale.min():.6g}, {scale.max():.6g}]", flush=True)

    if not a.check_only:
        t0 = time.time()
        with open(bin_path, "wb") as f:
            for lo in range(0, V, ROWS_BLOCK):
                hi = min(lo + ROWS_BLOCK, V)
                f.write(pack_engine((pattern[lo:hi] + 1).astype(np.uint8), enc).tobytes(order="C"))
        scale.tofile(scale_path)
        print(f"  {bin_path} {os.path.getsize(bin_path)} B and {scale_path} "
              f"{os.path.getsize(scale_path)} B in {time.time() - t0:.0f} s", flush=True)
    if os.path.getsize(bin_path) != want_bytes:
        raise SystemExit(f"{bin_path} is {os.path.getsize(bin_path)} B, expected {want_bytes}")

    checks = {}
    mm = np.memmap(bin_path, dtype=np.uint8, mode="r", shape=(V, row_bytes))

    bad = pad_bad = 0
    for lo in range(0, V, ROWS_BLOCK):
        hi = min(lo + ROWS_BLOCK, V)
        full = unpack_engine(np.asarray(mm[lo:hi]), enc)
        bad += int((full[:, :K] != pattern[lo:hi]).sum())
        pad_bad += int((full[:, K:] != 0).sum())
    if bad:
        raise SystemExit(f"{bad} trits do not survive the round trip")
    if pad_bad:
        raise SystemExit(f"{pad_bad} padding weights are not zero")
    checks["roundtrip_trits"] = int(V) * int(K)
    checks["padding_weights"] = int(beats * spec["per_beat"] - K)
    print(f"  round trip: all {V * K} trits unpack to the pattern they were packed from, "
          f"every padding weight zero", flush=True)

    per = spec["per_byte"]
    if enc == "2bit":
        by_hand = bytes(sum((int(pattern[0, per * b + j]) + 1) << (2 * j) for j in range(per))
                        for b in range(BEAT_BYTES))
    else:
        by_hand = bytes(sum((int(pattern[0, per * b + j]) + 1) * 3 ** j for j in range(per))
                        for b in range(BEAT_BYTES))
    if bytes(np.asarray(mm[0, :BEAT_BYTES])) != by_hand:
        raise SystemExit("the first beat of neuron 0 differs from the by-hand packing")
    checks["first_beat"] = by_hand.hex()
    print(f"  first beat of neuron 0 by hand: {by_hand.hex()}", flush=True)

    if not os.path.exists(a.states):
        print(f"  SKIPPED the shortlist self-check: no {a.states}.\n"
              f"  head_states.npz holds real final hidden states and is a research artefact this\n"
              f"  repository does not ship. Everything above -- the file read back, unpacked and\n"
              f"  compared with the pattern in memory, and the first beat rebuilt by hand -- did run,\n"
              f"  and the head is written and usable. What is skipped is the measurement that the\n"
              f"  K = {a.k} shortlist covers the exact argmax; the recorded result of that\n"
              f"  measurement on the reference board was a worst rank of 56 out of 256 over 2115\n"
              f"  real states. Pass --states FILE if you have one.", flush=True)
        checks["states_skipped"] = True
    else:
        head_i8 = np.empty((V, K), dtype=np.int8)
        head_scale = np.empty(V, dtype=np.float32)
        for lo in range(0, V, ROWS_BLOCK):
            hi = min(lo + ROWS_BLOCK, V)
            head_i8[lo:hi], head_scale[lo:hi] = quantise_head_block(emb.block(lo, hi))

        z = np.load(a.states, allow_pickle=False)
        fin, is_head = z["fin"].astype(np.float32), z["is_head"]
        idx = np.nonzero(is_head)[0]
        step = max(1, len(idx) // a.check_states)
        idx = idx[::step][:a.check_states]
        q, s = absmax_int8(fin[idx])
        print(f"  self-check on {len(idx)} real final hidden states of {a.states}", flush=True)

        unpacked = unpack_engine(np.asarray(mm), enc, K)
        approx_file = (unpacked.astype(np.float32) @ q.astype(np.float32).T).T * scale
        approx_mem = (pattern.astype(np.float32) @ q.astype(np.float32).T).T * scale
        if not np.array_equal(approx_file, approx_mem):
            raise SystemExit("the scores from the file differ from the scores from the pattern")
        checks["scores_match"] = True
        print("  the approximate scores from the unpacked file are identical to the ones from the pattern",
              flush=True)

        exact = exact_head_scores(head_i8, head_scale, q)
        arg = np.array([int(np.argmax(e)) for e in exact])
        mine = approx_file[np.arange(len(idx)), arg]
        rank = (approx_file >= mine[:, None]).sum(axis=1) - 1
        checks["states"] = int(len(idx))
        checks["worst_rank"] = int(rank.max())
        checks["rank0"] = float((rank == 0).mean())
        checks["k"] = int(a.k)
        checks["covered"] = bool((rank < a.k).all())
        print(f"  rank of the exact argmax under this file: worst {rank.max()}, top-1 already right on "
              f"{(rank == 0).mean() * 100:.1f}% of them", flush=True)
        if not checks["covered"]:
            raise SystemExit(f"K = {a.k} does not cover a state whose rank is {rank.max()}")
        print(f"  K = {a.k} covers every one of the {len(idx)} states", flush=True)

    per_engine = V // ENGINES
    if per_engine * ENGINES != V:
        raise SystemExit(f"{V} neurons do not split over {ENGINES} engines")
    manifest = {
        "file": spec["file"], "bytes": want_bytes, "md5": md5_of(bin_path),
        "scale_file": "head_t_scale.bin", "scale_bytes": os.path.getsize(scale_path),
        "scale_md5": md5_of(scale_path),
        "n": V, "k": K, "beats_per_neuron": beats,
        "encoding": enc, "stream_format": spec["text"],
        "weights_per_byte": spec["per_byte"], "weights_per_beat": spec["per_beat"],
        "padding_weights": beats * spec["per_beat"] - K,
        "act_beats_per_vector": beats * (spec["per_beat"] // BEAT_BYTES),
        "engines": ENGINES, "neurons_per_engine": per_engine,
        "bytes_per_engine": per_engine * beats * BEAT_BYTES,
        "result_bytes": V * 4,
        "rule": a.rule, "k_shortlist": a.k,
        "nonzero": float((pattern != 0).mean()),
        "snapshot": os.path.basename(snap), "checks": checks,
    }
    with open(os.path.join(a.out, "head_t.json"), "w") as f:
        json.dump(manifest, f, indent=1)
    print(f"  md5 {manifest['md5']}, {per_engine} neurons and {manifest['bytes_per_engine']} B an engine",
          flush=True)
    print(f"wrote {os.path.join(a.out, 'head_t.json')}", flush=True)


if __name__ == "__main__":
    main()
