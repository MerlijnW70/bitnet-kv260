"""Fetch layer 0's MLP ternary weights of BitNet b1.58 2B4T from Hugging Face, by HTTP byte range
out of the 1.18 GB model.safetensors, unpack them and save them as numpy arrays.

usage: python fetch.py [out_dir]

The checkpoint (microsoft/bitnet-b1.58-2B-4T, model.safetensors) stores every BitLinear weight
as U8 with four ternary weights a byte, the packing transformers documents in
integrations/bitnet.py: the packed tensor has out_features/4 rows, and bit pair i (bits 2i..2i+1)
of packed row r holds weight row i*rows + r as the value w+1, so 0, 1, 2 stand for -1, 0, +1 and
3 never appears. The weight scale is a single BF16 scalar a tensor, the reciprocal of the mean
absolute weight (dequantised weight = ternary / weight_scale).

Writes into out_dir:
  W_gate_proj.npy  int8 [6912, 2560] in {-1, 0, +1}      scale.npy   float32 [1], gate_proj's weight_scale
  W_up_proj.npy    int8 [6912, 2560]                     packed.npy  the raw U8 [1728, 2560] of gate_proj
  W_down_proj.npy  int8 [2560, 6912]                     header.json the safetensors index
  gamma.npy        float32 [6912], mlp.ffn_sub_norm.weight (the RMS norm before down_proj)
  scales.json      {"gate_proj": s, "up_proj": s, "down_proj": s} as floats, with the bf16 bits
The tensor names and packed shapes are read from the safetensors header (down_proj is packed as
U8 [640, 6912]).
"""
import json
import os
import struct
import subprocess
import sys

import numpy as np
import requests

try:
    import truststore

    truststore.inject_into_ssl()
except ImportError:
    pass

REPO = "microsoft/bitnet-b1.58-2B-4T"
FILE = "model.safetensors"
URL = f"https://huggingface.co/{REPO}/resolve/main/{FILE}"
CONFIG_URL = f"https://huggingface.co/{REPO}/resolve/main/config.json"
LAYER = "model.layers.0.mlp"
WEIGHT = f"{LAYER}.gate_proj.weight"
SCALE = f"{LAYER}.gate_proj.weight_scale"
PROJECTIONS = ("gate_proj", "up_proj", "down_proj")
NORM = f"{LAYER}.ffn_sub_norm.weight"
VALUES_PER_ITEM = 4


def fetch(url, start=None, end=None):
    """The body of url, or its bytes [start, end), following the redirect to the CDN; curl when
    Python's certificate store cannot verify the host."""
    headers = {} if start is None else {"Range": f"bytes={start}-{end - 1}"}
    try:
        r = requests.get(url, headers=headers, allow_redirects=True, timeout=300)
        r.raise_for_status()
        if start is not None and r.status_code != 206:
            raise SystemExit(f"expected 206 Partial Content, got {r.status_code}")
        data = r.content
    except requests.exceptions.SSLError:
        args = ["curl", "-sSL", "--max-time", "300"]
        if start is not None:
            args += ["-r", f"{start}-{end - 1}"]
        data = subprocess.run(args + [url], check=True, capture_output=True).stdout
    if start is not None and len(data) != end - start:
        raise SystemExit(f"wanted {end - start} bytes, got {len(data)}")
    return data


def fetch_range(url, start, end):
    return fetch(url, start, end)


def bf16_to_f32(raw):
    u16 = np.frombuffer(raw, dtype="<u2").astype(np.uint32)
    return (u16 << 16).view(np.float32)


def unpack_weights(packed):
    """transformers.integrations.bitnet.unpack_weights on numpy: U8 [rows/4, cols] -> int8 [rows, cols] in {-1, 0, +1}."""
    rows = packed.shape[0]
    out = np.zeros((rows * VALUES_PER_ITEM,) + packed.shape[1:], dtype=np.uint8)
    for i in range(VALUES_PER_ITEM):
        out[i * rows:(i + 1) * rows] = (packed >> (2 * i)) & 3
    if out.max() > 2:
        raise SystemExit("a two-bit field of 3 appeared: not the transformers packing")
    return out.astype(np.int8) - 1


def tensor_bytes(header, data_start, name, dtype):
    """The raw bytes of one tensor of the safetensors file, checked against the header's dtype."""
    meta = header[name]
    print(f"{name}: {meta}")
    if meta["dtype"] != dtype:
        raise SystemExit(f"{name}: expected {dtype}, the header says {meta['dtype']}")
    a, b = meta["data_offsets"]
    print(f"fetching bytes [{data_start + a}, {data_start + b}) = {b - a} bytes")
    return fetch_range(URL, data_start + a, data_start + b), meta


def describe(name, W):
    counts = {v: int((W == v).sum()) for v in (-1, 0, 1)}
    total = W.size
    print(f"{name}: shape {W.shape} dtype {W.dtype}")
    print(f"histogram: -1: {counts[-1]} ({counts[-1] / total:.4f})  0: {counts[0]} ({counts[0] / total:.4f})  "
          f"+1: {counts[1]} ({counts[1] / total:.4f})")
    dens = (W != 0).sum(axis=1) / W.shape[1]
    print(f"nonzero density per row: mean {dens.mean():.4f} min {dens.min():.4f} (row {dens.argmin()}) "
          f"max {dens.max():.4f} (row {dens.argmax()})")


def main():
    out_dir = sys.argv[1] if len(sys.argv) > 1 else os.path.dirname(os.path.abspath(__file__))
    os.makedirs(out_dir, exist_ok=True)

    config = json.loads(fetch(CONFIG_URL))
    print(f"config: hidden_size={config.get('hidden_size')} intermediate_size={config.get('intermediate_size')} "
          f"num_hidden_layers={config.get('num_hidden_layers')} hidden_act={config.get('hidden_act')} "
          f"rms_norm_eps={config.get('rms_norm_eps')} quant={config.get('quantization_config')}")
    hidden, inter = config["hidden_size"], config["intermediate_size"]

    (header_len,) = struct.unpack("<Q", fetch_range(URL, 0, 8))
    header_raw = fetch_range(URL, 8, 8 + header_len)
    header = json.loads(header_raw)
    open(os.path.join(out_dir, "header.json"), "wb").write(header_raw)
    data_start = 8 + header_len
    print(f"safetensors header {header_len} bytes, data starts at byte {data_start}")
    names = sorted(k for k in header if k.startswith(LAYER))
    print(f"{LAYER} tensors in the header: {names}")

    scales = {}
    weights = {}
    for proj in PROJECTIONS:
        raw, meta = tensor_bytes(header, data_start, f"{LAYER}.{proj}.weight", "U8")
        packed = np.frombuffer(raw, dtype=np.uint8).reshape(meta["shape"])
        raw_scale, _ = tensor_bytes(header, data_start, f"{LAYER}.{proj}.weight_scale", "BF16")
        scale = bf16_to_f32(raw_scale)
        bits = int(np.frombuffer(raw_scale, dtype="<u2")[0])
        W = unpack_weights(packed)
        want = (hidden, inter) if proj == "down_proj" else (inter, hidden)
        assert W.shape == want, (proj, W.shape, want)
        assert packed.shape == (want[0] // VALUES_PER_ITEM, want[1]), (proj, packed.shape)
        weights[proj] = W
        scales[proj] = {"weight_scale": float(scale[0]), "bf16_bits": f"0x{bits:04x}", "packed_shape": list(packed.shape),
                        "mean_abs_ternary": float(np.abs(W).mean())}
        if proj == "gate_proj":
            np.save(os.path.join(out_dir, "packed.npy"), packed)
            np.save(os.path.join(out_dir, "scale.npy"), scale.astype(np.float32))
        np.save(os.path.join(out_dir, f"W_{proj}.npy"), W)
        describe(f"W_{proj}", W)
        print(f"weight_scale: {scale[0]!r} (bf16 bits 0x{bits:04x}); 1/scale = {1.0 / scale[0]:.6f}; "
              f"mean |ternary| = {np.abs(W).mean():.4f}; mean |dequantised| = {np.abs(W).mean() / scale[0]:.6f}")

    raw, meta = tensor_bytes(header, data_start, NORM, "BF16")
    gamma = bf16_to_f32(raw).astype(np.float32)
    assert gamma.shape == (inter,), gamma.shape
    np.save(os.path.join(out_dir, "gamma.npy"), gamma)
    print(f"gamma (ffn_sub_norm.weight): {gamma.shape} float32, min {gamma.min():.6f} max {gamma.max():.6f} "
          f"mean {gamma.mean():.6f} max|gamma| {np.abs(gamma).max():.6f}, negatives {int((gamma < 0).sum())}, zeros {int((gamma == 0).sum())}")
    scales["ffn_sub_norm"] = {"shape": list(gamma.shape), "max_abs": float(np.abs(gamma).max()), "min": float(gamma.min()),
                              "max": float(gamma.max())}
    json.dump(scales, open(os.path.join(out_dir, "scales.json"), "w"), indent=1)

    for name in ("packed.npy", "W_gate_proj.npy", "W_up_proj.npy", "W_down_proj.npy", "gamma.npy", "scale.npy", "scales.json",
                 "header.json"):
        print(f"{name}: {os.path.getsize(os.path.join(out_dir, name))} bytes")


if __name__ == "__main__":
    main()
