"""Pack the whole of BitNet b1.58 2B4T out of the local Hugging Face cache into the files the
KV260 runtime (runtime/bitnet_kria.c) reads: every ternary matrix in the fabric's weight-stream
format, the float32 norms, the raw bf16 embedding table, the tied output head quantised to int8,
and a manifest that names every offset.

usage: python pack_model.py [out_dir] [--encoding base3|2bit] [--layers N] [--no-embed] [--no-head]
                            [--probes N] [--seed N] [--hidden file.npy] [--check-all] [--measure-head]

  out_dir     default $BITNET_PACK_DIR, else <temp>/bitnet-kria
  --encoding  base3 (default) writes model3.bin, five weights a byte in base 3, 80 a beat, each
              neuron padded to a whole beat with zero weights; 2bit writes model.bin, four weights
              a byte, 64 a beat, as before. The manifest's stream_format.name and .file say which,
              so an old file cannot be fed to a new engine or the other way round
  --layers N  pack only the first N layers (a smoke test; the manifest says partial)
  --no-embed  skip embed_bf16.bin        --no-head   skip head_i8.bin/head_scale.bin
  --probes N  hidden states a kind for the head's argmax agreement (default 128)
  --hidden f  also measure the head's agreement on real final hidden states, float32 [T, 2560]
              (hf_dump.py's hf_final_norm.npy); the file may hold one vector or many
  --check-all unpack every matrix back instead of a sample
  --measure-head  pack nothing: read the head already in out_dir, measure its agreement (with
              --hidden if the real states are to hand) and put the numbers in its manifest.json

Files written into out_dir:
  model3.bin       (base3) every ternary matrix, each starting at a 4096-byte boundary, in this
                   order for layer L = 0..29: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj,
                   down_proj. A matrix of N neurons and K inputs is N*16*ceil(K/80) bytes:
                   row-major per neuron, five weights a byte as sum of (w_j + 1) * 3^j for
                   j = 0..4 lowest power first, so 80 weights = one 16-byte beat, a neuron is
                   ceil(K/80) beats and its last 80*beats - K weights are zero (a byte of five
                   zero weights is 121). The whole model is 417546240 bytes, 80.14% of the
                   2-bit one, of which 737280 bytes is down_proj's padding
  model.bin        (2bit) the same matrices two bits a weight, N*K/4 bytes each, 64 weights a
                   beat, K/64 beats a neuron, no padding: 521011200 bytes for the whole model
  manifest.json    the geometry, the 4096-aligned offset/N/K/beats/padding/weight_scale of every
                   matrix, stream_format.name and .file naming the encoding, the offsets of the
                   norms, and the md5 and size of every file
  norms.bin        float32, per layer input_layernorm, post_attention_layernorm, attn_sub_norm,
                   ffn_sub_norm, then model.norm at the end
  embed_bf16.bin   model.embed_tokens.weight verbatim, raw bf16 [128256, 2560]
  head_i8.bin      the tied head quantised per row: scale_r = max|w_r| / 127,
                   w_i8 = clip(round(w_r / scale_r), -127, 127)   int8 [128256, 2560]
  head_scale.bin   float32 [128256], the row scales
  tokenizer.json   copied from the snapshot

Then it checks itself: a sample of matrices is read back out of the model file, unpacked and
compared with the Hugging Face tensor unpacked the transformers way, with every padding weight
zero, with the first beat packed by hand, and, in base 3, with the trits the 2-bit packing of the
same tensor gives; one matvec is recomputed both from the int8 matrix and straight from the packed
bytes; layer 0's gate_proj/up_proj/down_proj hold the trits of the gate.bin/up.bin/down.bin
pack_ffn.py already wrote (byte for byte in 2bit); norms.bin, embed_bf16.bin and a sample of head
rows are read back; and the int8 head's top-1 is compared with the bf16 head's on random hidden
states of three shapes.
"""
import hashlib
import json
import os
import shutil
import struct
import sys
import tempfile
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import pack

REPO_DIR = "models--microsoft--bitnet-b1.58-2B-4T"
SAFETENSORS = "model.safetensors"
ALIGN = 4096
WEIGHTS_PER_BYTE = 4
WEIGHTS_PER_BEAT = 64
BEAT_BYTES = 16
ENCODINGS = {
    "2bit": {"file": "model.bin", "per_byte": 4, "per_beat": 64,
             "text": "w+1 (0 = -1, 1 = 0, 2 = +1), four a byte lowest first, row-major per neuron"},
    "base3": {"file": "model3.bin", "per_byte": 5, "per_beat": 80,
              "text": "five weights a byte in base 3, byte = sum of (w_j + 1) * 3^j for j = 0..4 "
                      "(a byte of five zero weights is 121), row-major per neuron, each neuron "
                      "padded with zero weights to a whole 80-weight beat"},
}
MATRIX_ORDER = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
ATTN = ("q_proj", "k_proj", "v_proj", "o_proj")
NORM_ORDER = ("input_layernorm", "post_attention_layernorm", "attn_sub_norm", "ffn_sub_norm")
FINAL_NORM = "model.norm.weight"
EMBED = "model.embed_tokens.weight"
HEAD_QMAX = 127
BLOCK_ROWS = 8192


def snapshot_dir():
    """The model's snapshot directory in the local Hugging Face cache (never re-download)."""
    home = os.environ.get("HF_HOME") or os.path.join(os.path.expanduser("~"), ".cache", "huggingface")
    hub = os.environ.get("HF_HUB_CACHE") or os.path.join(home, "hub")
    root = os.path.join(hub, REPO_DIR, "snapshots")
    if not os.path.isdir(root):
        raise SystemExit(f"no local snapshot of the model under {root}")
    snaps = sorted(os.listdir(root))
    for snap in snaps:
        if os.path.exists(os.path.join(root, snap, SAFETENSORS)):
            return os.path.join(root, snap), snap
    raise SystemExit(f"no {SAFETENSORS} in any snapshot under {root}: {snaps}")


def read_header(path):
    with open(path, "rb") as f:
        (header_len,) = struct.unpack("<Q", f.read(8))
        header = json.loads(f.read(header_len))
    return header, 8 + header_len


class Store:
    """The safetensors file, memory-mapped, with the header's tensor table."""

    def __init__(self, path):
        self.path = path
        self.header, self.data_start = read_header(path)
        self.mm = np.memmap(path, dtype=np.uint8, mode="r")

    def meta(self, name):
        if name not in self.header:
            raise SystemExit(f"{name} is not in {os.path.basename(self.path)}")
        return self.header[name]

    def raw(self, name, dtype, lo=None, hi=None):
        """The tensor's bytes, or rows [lo, hi) of it, checked against the header's dtype."""
        meta = self.meta(name)
        if meta["dtype"] != dtype:
            raise SystemExit(f"{name}: expected {dtype}, the header says {meta['dtype']}")
        a, b = meta["data_offsets"]
        if lo is not None:
            per_row = (b - a) // meta["shape"][0]
            a, b = a + lo * per_row, a + hi * per_row
        return np.array(self.mm[self.data_start + a:self.data_start + b])

    def bf16(self, name, lo=None, hi=None):
        return bf16_to_f32(self.raw(name, "BF16", lo, hi))

    def bf16_bits(self, name):
        return int(self.raw(name, "BF16").view("<u2")[0])

    def packed(self, name):
        """A BitLinear weight as its raw U8 [out/4, in] tensor."""
        meta = self.meta(name)
        return self.raw(name, "U8").reshape(meta["shape"])


def bf16_to_f32(raw):
    return (raw.view("<u2").astype(np.uint32) << 16).view(np.float32)


def unpack_hf(packed):
    """transformers.integrations.bitnet.unpack_weights: U8 [out/4, in] -> uint8 [out, in] of w+1."""
    rows = packed.shape[0]
    out = np.empty((rows * WEIGHTS_PER_BYTE, packed.shape[1]), dtype=np.uint8)
    for i in range(WEIGHTS_PER_BYTE):
        out[i * rows:(i + 1) * rows] = (packed >> (2 * i)) & 3
    if out.max() > 2:
        raise SystemExit("a two-bit field of 3 appeared: not the transformers packing")
    return out


def pack_engine(values):
    """uint8 [out, in] of w+1 -> uint8 [out, in/4] in the fabric's stream order."""
    out, cols = values.shape
    if cols % WEIGHTS_PER_BYTE:
        raise SystemExit(f"{cols} inputs is not a multiple of four")
    v = values.reshape(out, cols // WEIGHTS_PER_BYTE, WEIGHTS_PER_BYTE)
    return np.ascontiguousarray(v[:, :, 0] | (v[:, :, 1] << 2) | (v[:, :, 2] << 4) | (v[:, :, 3] << 6))


def beats_for(k, enc):
    per_beat = ENCODINGS[enc]["per_beat"]
    return (k + per_beat - 1) // per_beat


def pack_engine3(values):
    """uint8 [out, in] of w+1 -> uint8 [out, 16 * ceil(in / 80)]: five weights a byte in base 3,
    each neuron padded to a whole beat with zero weights (a code of 1, so a padding byte is 121)."""
    out, cols = values.shape
    beats = beats_for(cols, "base3")
    padded = np.ones((out, beats * 80), dtype=np.uint8)
    padded[:, :cols] = values
    v = padded.reshape(out, beats * BEAT_BYTES, 5).astype(np.uint16)
    b = v[..., 0] + 3 * v[..., 1] + 9 * v[..., 2] + 27 * v[..., 3] + 81 * v[..., 4]
    return np.ascontiguousarray(b.astype(np.uint8))


def unpack_engine3(stream, cols):
    """The inverse: uint8 [out, 16 * beats] -> (int8 [out, cols] of the weights, int8 of the padding)."""
    v = stream.astype(np.int32)
    trits = np.empty((stream.shape[0], stream.shape[1], 5), dtype=np.int8)
    for t in range(5):
        trits[:, :, t] = (v % 3).astype(np.int8) - 1
        v //= 3
    flat = trits.reshape(stream.shape[0], -1)
    return flat[:, :cols], flat[:, cols:]


def pack_stream(values, enc):
    return pack_engine(values) if enc == "2bit" else pack_engine3(values)


def unpack_stream(stream, cols, enc):
    if enc == "2bit":
        return pack.unpack_weights(stream, cols), np.zeros((stream.shape[0], 0), dtype=np.int8)
    return unpack_engine3(stream, cols)


def hf_name(layer, matrix):
    kind = "self_attn" if matrix in ATTN else "mlp"
    return f"model.layers.{layer}.{kind}.{matrix}.weight"


def norm_name(layer, norm):
    if norm == "attn_sub_norm":
        return f"model.layers.{layer}.self_attn.attn_sub_norm.weight"
    if norm == "ffn_sub_norm":
        return f"model.layers.{layer}.mlp.ffn_sub_norm.weight"
    return f"model.layers.{layer}.{norm}.weight"


class Sink:
    """A file being written, hashed as it goes."""

    def __init__(self, path):
        self.path = path
        self.f = open(path, "wb")
        self.md5 = hashlib.md5()
        self.size = 0

    def write(self, data):
        self.f.write(data)
        self.md5.update(data)
        self.size += len(data)

    def pad_to(self, align):
        short = (-self.size) % align
        if short:
            self.write(b"\0" * short)

    def close(self):
        self.f.close()
        return {"bytes": self.size, "md5": self.md5.hexdigest()}


def md5_of(path, chunk=1 << 22):
    h = hashlib.md5()
    with open(path, "rb") as f:
        while True:
            block = f.read(chunk)
            if not block:
                return h.hexdigest()
            h.update(block)


def write_model_bin(store, out_dir, layers, geom, enc):
    """the model file plus the manifest's matrix table."""
    spec = ENCODINGS[enc]
    sink = Sink(os.path.join(out_dir, spec["file"]))
    hidden, inter, kv = geom["hidden_size"], geom["intermediate_size"], geom["kv_size"]
    shapes = {"q_proj": (hidden, hidden), "k_proj": (kv, hidden), "v_proj": (kv, hidden),
              "o_proj": (hidden, hidden), "gate_proj": (inter, hidden), "up_proj": (inter, hidden),
              "down_proj": (hidden, inter)}
    matrices = []
    t0 = time.time()
    for layer in range(layers):
        for matrix in MATRIX_ORDER:
            n, k = shapes[matrix]
            name = hf_name(layer, matrix)
            packed = store.packed(name)
            if packed.shape != (n // WEIGHTS_PER_BYTE, k):
                raise SystemExit(f"{name}: packed shape {packed.shape}, expected {(n // WEIGHTS_PER_BYTE, k)}")
            stream = pack_stream(unpack_hf(packed), enc)
            sink.pad_to(ALIGN)
            offset = sink.size
            sink.write(stream.tobytes(order="C"))
            beats = beats_for(k, enc)
            scale_name = f"{name}_scale"
            matrices.append({"layer": layer, "name": matrix, "offset": offset, "n": n, "k": k,
                             "beats_per_neuron": beats, "bytes": int(n * stream.shape[1]),
                             "padding_weights": beats * spec["per_beat"] - k,
                             "act_beats_per_vector": beats * (spec["per_beat"] // BEAT_BYTES),
                             "weight_scale": float(store.bf16(scale_name)[0]),
                             "weight_scale_bf16": f"0x{store.bf16_bits(scale_name):04x}"})
        print(f"  layer {layer}: {sink.size} bytes so far ({time.time() - t0:.1f} s)", flush=True)
    info = sink.close()
    return matrices, info, shapes


def write_norms_bin(store, out_dir, layers, geom):
    sink = Sink(os.path.join(out_dir, "norms.bin"))
    entries = []
    for layer in range(layers):
        for norm in NORM_ORDER:
            values = store.bf16(norm_name(layer, norm)).astype(np.float32)
            want = geom["intermediate_size"] if norm == "ffn_sub_norm" else geom["hidden_size"]
            if values.shape != (want,):
                raise SystemExit(f"{norm_name(layer, norm)}: shape {values.shape}, expected {(want,)}")
            entries.append({"layer": layer, "name": norm, "offset": sink.size, "len": int(values.size)})
            sink.write(values.tobytes())
    final = store.bf16(FINAL_NORM).astype(np.float32)
    entries.append({"layer": -1, "name": "model.norm", "offset": sink.size, "len": int(final.size)})
    sink.write(final.tobytes())
    return entries, sink.close()


def write_embed_bin(store, out_dir, geom):
    name = os.path.join(out_dir, "embed_bf16.bin")
    sink = Sink(name)
    rows = geom["vocab_size"]
    t0 = time.time()
    for lo in range(0, rows, BLOCK_ROWS):
        hi = min(lo + BLOCK_ROWS, rows)
        sink.write(store.raw(EMBED, "BF16", lo, hi).tobytes())
    info = sink.close()
    print(f"  embed_bf16.bin {info['bytes']} bytes in {time.time() - t0:.1f} s", flush=True)
    return info


class TopK:
    """The best k logits of every probe state, kept as the vocabulary streams past in blocks."""

    def __init__(self, states, k=5):
        self.k = k
        self.values = np.full((states, k), -np.inf)
        self.index = np.full((states, k), -1, dtype=np.int64)

    def update(self, block, base):
        """block: [rows, states] logits of vocabulary rows base .. base + rows."""
        rows = block.shape[0]
        take = min(self.k, rows)
        by_state = np.ascontiguousarray(block.T)
        part = np.argpartition(-by_state, take - 1, axis=1)[:, :take]
        part.sort(axis=1)
        values = np.concatenate([self.values, np.take_along_axis(by_state, part, axis=1)], axis=1)
        index = np.concatenate([self.index, part + base], axis=1)
        order = np.argsort(-values, axis=1, kind="stable")[:, :self.k]
        self.values = np.take_along_axis(values, order, axis=1)
        self.index = np.take_along_axis(index, order, axis=1)


VARIANTS = ("bf16 W, int8 h", "int8 W, float h", "int8 W, int8 h")


def quantize_head(store, out_dir, geom, probes, write=True):
    """Quantise the tied head per row and, in the same pass, follow four logit vectors of every
    probe hidden state: the bf16 reference and the three approximations of VARIANTS, so the runtime
    can see what the weight quantisation costs and what the activation quantisation costs.
    With write False the int8 head is read back out of out_dir instead of being made again."""
    rows, cols = geom["vocab_size"], geom["hidden_size"]
    weights = Sink(os.path.join(out_dir, "head_i8.bin")) if write else None
    scales_sink = Sink(os.path.join(out_dir, "head_scale.bin")) if write else None
    if not write:
        shipped_q = np.memmap(os.path.join(out_dir, "head_i8.bin"), dtype=np.int8, mode="r", shape=(rows, cols))
        shipped_scale = np.fromfile(os.path.join(out_dir, "head_scale.bin"), dtype="<f4")
    h_ref = np.asfortranarray(probes["h"].astype(np.float64).T)
    h_q = np.asfortranarray(probes["q"].astype(np.float64).T)
    states = h_ref.shape[1]
    top_ref = TopK(states)
    tops = {name: TopK(states) for name in VARIANTS}
    dot = {name: np.zeros(states) for name in VARIANTS}
    norm = {name: np.zeros(states) for name in VARIANTS}
    norm_ref = np.zeros(states)
    t0 = time.time()
    for lo in range(0, rows, BLOCK_ROWS):
        hi = min(lo + BLOCK_ROWS, rows)
        block = bf16_to_f32(store.raw(EMBED, "BF16", lo, hi)).reshape(hi - lo, cols).astype(np.float64)
        if write:
            scale = (np.abs(block).max(axis=1) / HEAD_QMAX).astype(np.float32)
            safe = np.where(scale > 0, scale, np.float32(1.0)).astype(np.float64)
            q = np.clip(np.rint(block / safe[:, None]), -HEAD_QMAX, HEAD_QMAX).astype(np.int8)
            weights.write(q.tobytes(order="C"))
            scales_sink.write(scale.tobytes())
        else:
            q, scale = np.asarray(shipped_q[lo:hi]), shipped_scale[lo:hi]
        qf = q.astype(np.float64)
        srow = scale.astype(np.float64)[:, None]
        logits_ref = block @ h_ref
        made = {"bf16 W, int8 h": block @ h_q, "int8 W, float h": (qf @ h_ref) * srow,
                "int8 W, int8 h": (qf @ h_q) * srow}
        top_ref.update(logits_ref, lo)
        norm_ref += (logits_ref ** 2).sum(axis=0)
        for name, logits in made.items():
            tops[name].update(logits, lo)
            dot[name] += (logits_ref * logits).sum(axis=0)
            norm[name] += (logits ** 2).sum(axis=0)
    info = weights.close() if write else {"bytes": os.path.getsize(os.path.join(out_dir, "head_i8.bin")),
                                          "md5": md5_of(os.path.join(out_dir, "head_i8.bin"))}
    scale_info = scales_sink.close() if write else {"bytes": os.path.getsize(os.path.join(out_dir, "head_scale.bin")),
                                                    "md5": md5_of(os.path.join(out_dir, "head_scale.bin"))}
    verb = "written" if write else "read back"
    print(f"  head_i8.bin {info['bytes']} bytes, head_scale.bin {scale_info['bytes']} bytes {verb} in {time.time() - t0:.1f} s", flush=True)
    stats = {"top_ref": top_ref, "tops": tops,
             "cosine": {name: dot[name] / np.sqrt(norm_ref * norm[name]) for name in VARIANTS}}
    return info, scale_info, stats


def rmsnorm(x, weight, eps):
    x = np.asarray(x, dtype=np.float32)
    return (x / np.sqrt((x.astype(np.float64) ** 2).mean(axis=-1, keepdims=True) + eps)).astype(np.float32) * weight


def quantize_int8(x):
    """BitNet's absmax int8 quantisation of a hidden state, as the runtime's head will do it."""
    x = np.asarray(x, dtype=np.float32)
    amax = np.maximum(np.abs(x).max(axis=-1, keepdims=True), 1e-5)
    return np.clip(np.rint(x * (HEAD_QMAX / amax)), -128, HEAD_QMAX).astype(np.int8), amax / HEAD_QMAX


def probe_states(store, geom, count, seed, extra, extra_label):
    """Hidden states to compare the two heads on, in three shapes plus any real dump."""
    rng = np.random.default_rng(seed)
    cols, eps = geom["hidden_size"], geom["rms_norm_eps"]
    final = store.bf16(FINAL_NORM).astype(np.float32)
    kinds, states = [], []
    unit = rng.standard_normal((count, cols)).astype(np.float32)
    states.append(unit / np.linalg.norm(unit, axis=1, keepdims=True))
    kinds.append(("gaussian unit", count))
    states.append(rmsnorm(rng.standard_normal((count, cols)).astype(np.float32), final, eps))
    kinds.append(("rmsnorm-shaped", count))
    embed_rows = rng.integers(0, geom["vocab_size"], size=(count, 8))
    mix = np.zeros((count, cols), dtype=np.float32)
    for t in range(count):
        coeff = rng.standard_normal(8).astype(np.float32)
        for j, row in enumerate(embed_rows[t]):
            mix[t] += coeff[j] * bf16_to_f32(store.raw(EMBED, "BF16", int(row), int(row) + 1))
    states.append(rmsnorm(mix, final, eps))
    kinds.append(("embedding mixture", count))
    if extra is not None:
        states.append(extra)
        kinds.append((f"real final hidden states ({extra_label})", extra.shape[0]))
    h = np.concatenate(states, axis=0)
    q, _ = quantize_int8(h)
    return {"h": h, "q": q, "kinds": kinds}


def one_agreement(ref_idx, other_idx, cos, lo, hi):
    top1_ref, top1_other = ref_idx[:, 0], other_idx[:, 0]
    same = top1_ref[lo:hi] == top1_other[lo:hi]
    in5 = np.array([top1_ref[t] in other_idx[t] for t in range(lo, hi)])
    overlap = np.array([len(set(ref_idx[t].tolist()) & set(other_idx[t].tolist())) for t in range(lo, hi)])
    return {"states": int(hi - lo), "top1_same": int(same.sum()), "top1_agreement": float(same.mean()),
            "top1_in_top5": float(in5.mean()), "top5_overlap": float(overlap.mean() / 5.0),
            "logit_cosine_min": float(cos[lo:hi].min()), "logit_cosine_mean": float(cos[lo:hi].mean())}


def head_agreement(probes, stats):
    """Per kind of hidden state and per variant: how often the approximate head's top-1 is the
    bf16 head's, whether the bf16 top-1 is in its top five, the top-5 overlap and the logit cosine."""
    ref_idx = stats["top_ref"].index
    out, lo = [], 0
    for kind, count in probes["kinds"]:
        hi = lo + count
        row = {"kind": kind, "states": int(count)}
        for name in VARIANTS:
            row[name] = one_agreement(ref_idx, stats["tops"][name].index, stats["cosine"][name], lo, hi)
        out.append(row)
        lo = hi
    whole = {"kind": "all kinds", "states": int(ref_idx.shape[0])}
    for name in VARIANTS:
        whole[name] = one_agreement(ref_idx, stats["tops"][name].index, stats["cosine"][name], 0, ref_idx.shape[0])
    return out, whole


def check_matrices(store, out_dir, matrices, sample, check_all, enc):
    """Read matrices back out of the model file and compare with the Hugging Face tensors: the
    trits, the padding, the first beat packed by hand, and, for base 3, the same trits the 2-bit
    packing of the same tensor gives."""
    spec = ENCODINGS[enc]
    path = os.path.join(out_dir, spec["file"])
    mm = np.memmap(path, dtype=np.uint8, mode="r")
    picked = matrices if check_all else [matrices[i] for i in sample]
    for entry in picked:
        n, k, off = entry["n"], entry["k"], entry["offset"]
        if off % ALIGN:
            raise SystemExit(f"{entry} is not 4096-aligned")
        stream = np.array(mm[off:off + entry["bytes"]]).reshape(n, entry["bytes"] // n)
        want = unpack_hf(store.packed(hf_name(entry["layer"], entry["name"]))).astype(np.int8) - 1
        got, padding = unpack_stream(stream, k, enc)
        if not np.array_equal(got, want):
            raise SystemExit(f"layer {entry['layer']} {entry['name']} does not unpack to its tensor")
        if padding.size and padding.any():
            raise SystemExit(f"layer {entry['layer']} {entry['name']}: a padding weight is not zero")
        if padding.shape[1] != entry["padding_weights"]:
            raise SystemExit(f"layer {entry['layer']} {entry['name']}: {padding.shape[1]} padding weights, "
                             f"the manifest says {entry['padding_weights']}")
        if enc == "base3":
            if not np.array_equal(got, pack.unpack_weights(pack.pack_weights(want), k)):
                raise SystemExit(f"layer {entry['layer']} {entry['name']}: base 3 and the 2-bit packing disagree")
            by_hand = bytes(sum((int(want[0, 5 * b + j]) + 1) * 3 ** j for j in range(5)) for b in range(BEAT_BYTES))
        else:
            if not np.array_equal(stream, pack.pack_weights(want)):
                raise SystemExit(f"layer {entry['layer']} {entry['name']}: the fast packer differs from pack.pack_weights")
            by_hand = bytes(sum((int(want[0, 4 * b + j]) + 1) << (2 * j) for j in range(4)) for b in range(BEAT_BYTES))
        if stream[0, :BEAT_BYTES].tobytes() != by_hand:
            raise SystemExit(f"layer {entry['layer']} {entry['name']}: first beat differs from the by-hand packing")
    return len(picked)


def check_matvec(store, out_dir, matrices, seed, enc):
    """One matvec both ways: from the int8 matrix, and straight out of the packed stream bytes,
    the fabric's way (the activations padded to a whole beat as the engine's store is)."""
    spec = ENCODINGS[enc]
    entry = next(e for e in matrices if e["layer"] == 0 and e["name"] == "gate_proj")
    n, k = entry["n"], entry["k"]
    mm = np.memmap(os.path.join(out_dir, spec["file"]), dtype=np.uint8, mode="r")
    stream = np.array(mm[entry["offset"]:entry["offset"] + entry["bytes"]]).reshape(n, entry["bytes"] // n)
    W = unpack_hf(store.packed(hf_name(0, "gate_proj"))).astype(np.int8) - 1
    x = np.random.default_rng(seed).integers(-128, 128, size=k, dtype=np.int64).astype(np.int8)
    reference = pack.matvec_int(W, x)
    per_byte = spec["per_byte"]
    padded = np.zeros(stream.shape[1] * per_byte, dtype=np.int8)
    padded[:k] = x
    xs = padded.astype(np.int32).reshape(stream.shape[1], per_byte)
    from_stream = np.zeros(n, dtype=np.int32)
    v = stream.astype(np.int32)
    for j in range(per_byte):
        w = (v % 3 if enc == "base3" else v & 3).astype(np.int32) - 1
        from_stream += w @ xs[:, j]
        v = v // 3 if enc == "base3" else v >> 2
    if not np.array_equal(reference, from_stream):
        raise SystemExit("the matvec off the packed stream differs from the int8 matvec")
    return entry, x, reference


def check_against_pack_ffn(out_dir, matrices, enc):
    """Layer 0's three MLP matrices must be byte for byte what pack_ffn.py already wrote (2 bits a
    weight), or, in base 3, must unpack to exactly the trits that 2-bit file holds."""
    mm = np.memmap(os.path.join(out_dir, ENCODINGS[enc]["file"]), dtype=np.uint8, mode="r")
    checked = []
    for matrix, name in (("gate_proj", "gate.bin"), ("up_proj", "up.bin"), ("down_proj", "down.bin")):
        path = os.path.join(HERE, name)
        if not os.path.exists(path):
            continue
        entry = next(e for e in matrices if e["layer"] == 0 and e["name"] == matrix)
        old = np.fromfile(path, dtype=np.uint8)
        new = np.array(mm[entry["offset"]:entry["offset"] + entry["bytes"]])
        if enc == "2bit":
            if old.size != new.size or not np.array_equal(old, new):
                raise SystemExit(f"layer 0 {matrix} differs from {name}")
        else:
            n, k = entry["n"], entry["k"]
            was = pack.unpack_weights(old.reshape(n, k // WEIGHTS_PER_BYTE), k)
            got, padding = unpack_engine3(new.reshape(n, entry["bytes"] // n), k)
            if not np.array_equal(got, was) or padding.any():
                raise SystemExit(f"layer 0 {matrix} in base 3 does not hold the trits of {name}")
        checked.append(name)
    return checked


def check_norms(store, out_dir, entries):
    values = np.fromfile(os.path.join(out_dir, "norms.bin"), dtype="<f4")
    for entry in entries:
        want = (store.bf16(FINAL_NORM) if entry["layer"] < 0 else store.bf16(norm_name(entry["layer"], entry["name"]))).astype(np.float32)
        got = values[entry["offset"] // 4:entry["offset"] // 4 + entry["len"]]
        if not np.array_equal(got, want):
            raise SystemExit(f"norms.bin: {entry} differs from the tensor")
    return len(entries)


def check_embed(store, out_dir, geom):
    meta = store.meta(EMBED)
    a, b = meta["data_offsets"]
    want = hashlib.md5()
    for lo in range(0, geom["vocab_size"], BLOCK_ROWS):
        hi = min(lo + BLOCK_ROWS, geom["vocab_size"])
        want.update(store.raw(EMBED, "BF16", lo, hi).tobytes())
    got = md5_of(os.path.join(out_dir, "embed_bf16.bin"))
    if got != want.hexdigest() or os.path.getsize(os.path.join(out_dir, "embed_bf16.bin")) != b - a:
        raise SystemExit("embed_bf16.bin is not the tensor's bytes")
    return b - a


def check_head(store, out_dir, geom, seed, rows_to_check=64):
    """A sample of head rows: the int8 row times its scale is the bf16 row to within half a step."""
    rows, cols = geom["vocab_size"], geom["hidden_size"]
    q_mm = np.memmap(os.path.join(out_dir, "head_i8.bin"), dtype=np.int8, mode="r", shape=(rows, cols))
    scales = np.fromfile(os.path.join(out_dir, "head_scale.bin"), dtype="<f4")
    if scales.shape != (rows,):
        raise SystemExit(f"head_scale.bin has {scales.shape}, expected {(rows,)}")
    worst = 0.0
    for row in np.random.default_rng(seed).integers(0, rows, size=rows_to_check):
        w = bf16_to_f32(store.raw(EMBED, "BF16", int(row), int(row) + 1)).astype(np.float64)
        s = float(scales[row])
        q = np.asarray(q_mm[row], dtype=np.float64)
        if s <= 0:
            if q.any():
                raise SystemExit(f"head row {row}: zero scale but nonzero int8")
            continue
        if np.abs(q).max() > HEAD_QMAX or not np.array_equal(q, np.clip(np.rint(w / s), -HEAD_QMAX, HEAD_QMAX)):
            raise SystemExit(f"head row {row} is not the quantisation of its bf16 row")
        worst = max(worst, float(np.abs(q * s - w).max() / s))
    if worst > 0.5 + 1e-6:
        raise SystemExit(f"a head row is off by {worst} steps")
    return rows_to_check, worst


def report_agreement(agreement, whole):
    print("\nhead agreement against the bf16 head (the int8 head the runtime uses is the last of each three):")
    for row in agreement + [whole]:
        print(f"  {row['kind']} ({row['states']} hidden states)")
        for name in VARIANTS:
            r = row[name]
            print(f"    {name:<16} top-1 {r['top1_same']}/{r['states']} = {r['top1_agreement']:.4f}, "
                  f"bf16 top-1 in its top-5 {r['top1_in_top5']:.4f}, top-5 overlap {r['top5_overlap']:.4f}, "
                  f"logit cosine mean {r['logit_cosine_mean']:.6f} min {r['logit_cosine_min']:.6f}")


def load_hidden(path, geom):
    hidden = np.atleast_2d(np.asarray(np.load(path), dtype=np.float32))
    if hidden.shape[1] != geom["hidden_size"]:
        raise SystemExit(f"{path} has {hidden.shape[1]} columns, expected {geom['hidden_size']}")
    return hidden, os.path.basename(path)


def measure_only(store, out_dir, geom, opts, seed):
    """Measure the agreement of the head already in out_dir and put the numbers in its manifest."""
    path = os.path.join(out_dir, "manifest.json")
    manifest = json.load(open(path, encoding="utf-8"))
    extra, extra_label = load_hidden(opts["--hidden"], geom) if opts["--hidden"] else (None, None)
    probes = probe_states(store, geom, opts["--probes"], seed, extra, extra_label)
    head_info, scale_info, stats = quantize_head(store, out_dir, geom, probes, write=False)
    agreement, whole = head_agreement(probes, stats)
    for name, info in (("head_i8.bin", head_info), ("head_scale.bin", scale_info)):
        if manifest["files"][name] != info:
            raise SystemExit(f"{name} in {out_dir} is not the one the manifest names: {info} against {manifest['files'][name]}")
    manifest["head"]["agreement"], manifest["head"]["agreement_all"] = agreement, whole
    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=1)
    report_agreement(agreement, whole)
    print(f"\n{path} rewritten with these numbers, {os.path.getsize(path)} bytes, md5 {md5_of(path)}")
    return 0


def main(argv):
    args = list(argv)
    opts = {"--layers": 30, "--probes": 128, "--seed": 5, "--hidden": None, "--encoding": "base3"}
    for flag in list(opts):
        if flag in args:
            i = args.index(flag)
            opts[flag] = args[i + 1] if flag in ("--hidden", "--encoding") else int(args[i + 1])
            del args[i:i + 2]
    enc = opts["--encoding"]
    if enc not in ENCODINGS:
        raise SystemExit(f"--encoding must be one of {sorted(ENCODINGS)}, not {enc!r}")
    do_embed = "--no-embed" not in args
    do_head = "--no-head" not in args
    check_all = "--check-all" in args
    measure = "--measure-head" in args
    for flag in ("--no-embed", "--no-head", "--check-all", "--measure-head"):
        if flag in args:
            args.remove(flag)
    out_dir = args[0] if args else os.environ.get("BITNET_PACK_DIR") or os.path.join(tempfile.gettempdir(), "bitnet-kria")
    os.makedirs(out_dir, exist_ok=True)
    layers = opts["--layers"]
    seed = opts["--seed"]

    snap, revision = snapshot_dir()
    store = Store(os.path.join(snap, SAFETENSORS))
    config = json.load(open(os.path.join(snap, "config.json"), encoding="utf-8"))
    heads, kv_heads = config["num_attention_heads"], config["num_key_value_heads"]
    head_dim = config["hidden_size"] // heads
    geom = {"hidden_size": config["hidden_size"], "intermediate_size": config["intermediate_size"],
            "num_hidden_layers": config["num_hidden_layers"], "num_attention_heads": heads,
            "num_key_value_heads": kv_heads, "head_dim": head_dim, "kv_size": kv_heads * head_dim,
            "rope_theta": config["rope_theta"], "rms_norm_eps": config["rms_norm_eps"],
            "vocab_size": config["vocab_size"], "tie_word_embeddings": config["tie_word_embeddings"],
            "hidden_act": config["hidden_act"], "max_position_embeddings": config["max_position_embeddings"],
            "bos_token_id": config["bos_token_id"], "eos_token_id": config["eos_token_id"]}
    print(f"snapshot {snap}\nrevision {revision}\nout_dir {out_dir}")
    print(f"geometry {geom}")
    if measure:
        return measure_only(store, out_dir, geom, opts, seed)

    spec = ENCODINGS[enc]
    model_file = spec["file"]
    files = {}
    t0 = time.time()
    print(f"{model_file} ({enc}):", flush=True)
    matrices, files[model_file], shapes = write_model_bin(store, out_dir, layers, geom, enc)
    norms, files["norms.bin"] = write_norms_bin(store, out_dir, layers, geom)
    if do_embed:
        files["embed_bf16.bin"] = write_embed_bin(store, out_dir, geom)
    agreement = whole = None
    if do_head:
        extra, extra_label = load_hidden(opts["--hidden"], geom) if opts["--hidden"] else (None, None)
        probes = probe_states(store, geom, opts["--probes"], seed, extra, extra_label)
        files["head_i8.bin"], files["head_scale.bin"], stats = quantize_head(store, out_dir, geom, probes)
        agreement, whole = head_agreement(probes, stats)
    tok_src = os.path.join(snap, "tokenizer.json")
    shutil.copyfile(tok_src, os.path.join(out_dir, "tokenizer.json"))
    files["tokenizer.json"] = {"bytes": os.path.getsize(tok_src), "md5": md5_of(tok_src)}
    pack_seconds = time.time() - t0

    def shape_entry(m):
        n, k = shapes[m]
        beats = beats_for(k, enc)
        return {"n": n, "k": k, "bytes": n * beats * BEAT_BYTES, "beats_per_neuron": beats,
                "padding_weights": beats * spec["per_beat"] - k,
                "act_beats_per_vector": beats * (spec["per_beat"] // BEAT_BYTES),
                "offset_in_layer": next(e["offset"] for e in matrices if e["name"] == m) if matrices else 0}

    layer_bytes = sum(shape_entry(m)["bytes"] for m in MATRIX_ORDER)
    manifest = {
        "model": "microsoft/bitnet-b1.58-2B-4T", "revision": revision, "layers_packed": layers,
        "partial": layers != geom["num_hidden_layers"], "geometry": geom,
        "stream_format": {"align": ALIGN, "name": enc, "file": model_file,
                          "weights_per_byte": spec["per_byte"], "weights_per_beat": spec["per_beat"],
                          "beat_bytes": BEAT_BYTES, "batch_max": 4,
                          "encoding": spec["text"],
                          "layer_stride": layer_bytes, "order": list(MATRIX_ORDER),
                          "shapes": {m: shape_entry(m) for m in MATRIX_ORDER}},
        "matrices": matrices,
        "norms": {"file": "norms.bin", "dtype": "float32", "order": list(NORM_ORDER), "entries": norms},
        "embed": {"file": "embed_bf16.bin", "dtype": "bf16", "rows": geom["vocab_size"], "cols": geom["hidden_size"]},
        "head": {"file": "head_i8.bin", "scales": "head_scale.bin", "dtype": "int8", "scale_dtype": "float32",
                 "rows": geom["vocab_size"], "cols": geom["hidden_size"], "tied": bool(geom["tie_word_embeddings"]),
                 "quantisation": "row r: scale_r = max|w_r| / 127, w_i8 = clip(round(w_r / scale_r), -127, 127)",
                 "agreement": agreement, "agreement_all": whole},
        "files": files,
    }
    with open(os.path.join(out_dir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=1)
    files["manifest.json"] = {"bytes": os.path.getsize(os.path.join(out_dir, "manifest.json")),
                             "md5": md5_of(os.path.join(out_dir, "manifest.json"))}

    print(f"\npacked in {pack_seconds:.1f} s")
    for name in (model_file, "manifest.json", "norms.bin", "embed_bf16.bin", "head_i8.bin", "head_scale.bin", "tokenizer.json"):
        if name in files:
            print(f"{os.path.join(out_dir, name)}  {files[name]['bytes']} bytes  md5 {files[name]['md5']}")
    print(f"{len(matrices)} matrices, {layers} layers, layer stride {layer_bytes} bytes, "
          f"{model_file} {files[model_file]['bytes']} bytes ({enc})")
    for m in MATRIX_ORDER:
        s = manifest["stream_format"]["shapes"][m]
        print(f"  {m}: N {s['n']} K {s['k']} {s['bytes']} bytes {s['beats_per_neuron']} beats a neuron, "
              f"{s['padding_weights']} padding weights, {s['act_beats_per_vector']} activation beats a vector, "
              f"offset in layer {s['offset_in_layer']}")
    print(f"first three matrices: " + "; ".join(f"L{e['layer']} {e['name']} @ {e['offset']} scale {e['weight_scale']!r}" for e in matrices[:3]))
    print(f"last matrix: L{matrices[-1]['layer']} {matrices[-1]['name']} @ {matrices[-1]['offset']} "
          f"+ {matrices[-1]['bytes']} = {matrices[-1]['offset'] + matrices[-1]['bytes']}")

    print("\nchecks:", flush=True)
    rng = np.random.default_rng(seed)
    sample = sorted(set(list(range(min(7, len(matrices)))) + rng.integers(0, len(matrices), size=min(13, len(matrices))).tolist()))
    print(f"  matrices unpacked back and compared with the tensors: {check_matrices(store, out_dir, matrices, sample, check_all, enc)}"
          f"{' (every one)' if check_all else ' of ' + str(len(matrices))}", flush=True)
    entry, x, product = check_matvec(store, out_dir, matrices, seed, enc)
    print(f"  matvec both ways on L0 {entry['name']}: {entry['n']} sums equal, range [{product.min()}, {product.max()}], "
          f"first four {product[:4].tolist()}")
    same = check_against_pack_ffn(out_dir, matrices, enc)
    print(f"  layer 0 trits identical to pack_ffn.py's {same if same else '(no gate.bin/up.bin/down.bin next to pack_ffn.py)'}")
    print(f"  norms.bin entries checked: {check_norms(store, out_dir, norms)}")
    if do_embed:
        print(f"  embed_bf16.bin is the tensor's {check_embed(store, out_dir, geom)} bytes")
    if do_head:
        rows_checked, worst = check_head(store, out_dir, geom, seed)
        print(f"  head rows requantised and compared: {rows_checked}, worst error {worst:.4f} steps (half a step is the bound)")
        report_agreement(agreement, whole)
    total = sum(v["bytes"] for v in files.values())
    print(f"\ntotal {total} bytes ({total / 2 ** 20:.1f} MiB) in {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
