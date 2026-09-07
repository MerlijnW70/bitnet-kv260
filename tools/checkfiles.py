"""Check every packed file against the size and md5 the manifest recorded when it was written.

usage: python3 checkfiles.py [DIR] [--encoding base3|2bit|any]
       (DIR defaults to /home/ubuntu/bitnet-kria, the encoding to base3)

pack_model.py writes manifest.json with a "files" section holding {bytes, md5} for every file it
produced, and pack_head_ternary.py writes head_t.json the same way for the ternary head.  This
reads both and reports one line a file.  A wrong file here is the difference between an answer and
a page of noise, and nothing else in the pipeline notices: the runtime checks the encoding name and
the geometry, not the bytes.

exit 0 every present file matches, 1 a file differs or a required file is missing.
"""
import hashlib
import json
import os
import sys

REQUIRED = ("model3.bin", "norms.bin", "embed_bf16.bin", "head_i8.bin", "head_scale.bin",
            "tokenizer.json")


def md5_of(path, chunk=1 << 22):
    h = hashlib.md5()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def wanted(dirname):
    """{name: (bytes, md5)} out of manifest.json and, if it is there, head_t.json."""
    want = {}
    mf = os.path.join(dirname, "manifest.json")
    if not os.path.exists(mf):
        raise SystemExit(f"checkfiles: no {mf}; pack the model first (tools/pack_model.py)")
    m = json.load(open(mf, encoding="utf-8"))
    for name, info in (m.get("files") or {}).items():
        want[name] = (int(info["bytes"]), info["md5"])
    ht = os.path.join(dirname, "head_t.json")
    if os.path.exists(ht):
        h = json.load(open(ht, encoding="utf-8"))
        if h.get("file") and h.get("md5"):
            want[h["file"]] = (int(h.get("bytes", -1)), h["md5"])
        if h.get("scale_file") and h.get("scale_md5"):
            want[h["scale_file"]] = (int(h.get("scale_bytes", -1)), h["scale_md5"])
    return want


def encoding(dirname, expect):
    """Print the stream format the manifest names, and say whether it is the one asked for."""
    m = json.load(open(os.path.join(dirname, "manifest.json"), encoding="utf-8"))
    sf = m.get("stream_format") or {}
    name, f = sf.get("name"), sf.get("file")
    print(f"stream_format {name} ({f}), {sf.get('weights_per_byte')} weights a byte, "
          f"{sf.get('weights_per_beat')} a beat")
    if expect != "any" and name != expect:
        print(f"ENCODING this bitstream streams {expect}; the manifest says {name}. "
              f"Repack: python3 tools/pack_model.py OUT --encoding {expect}")
        return 1
    return 0


def main(argv):
    expect = "base3"
    if "--encoding" in argv:
        i = argv.index("--encoding")
        expect = argv[i + 1]
        argv = argv[:i] + argv[i + 2:]
    dirname = argv[0] if argv else "/home/ubuntu/bitnet-kria"
    want = wanted(dirname)
    enc_bad = encoding(dirname, expect)
    bad = missing = checked = 0
    for name in sorted(want):
        size, md5 = want[name]
        path = os.path.join(dirname, name)
        if not os.path.exists(path):
            if name in REQUIRED:
                print(f"MISSING  {name}")
                missing += 1
            continue
        got_size = os.path.getsize(path)
        if size >= 0 and got_size != size:
            print(f"SIZE     {name}: {got_size} bytes, the manifest says {size}")
            bad += 1
            continue
        got = md5_of(path)
        if got != md5:
            print(f"MD5      {name}: {got}, the manifest says {md5}")
            bad += 1
            continue
        checked += 1
    for name in REQUIRED:
        if name not in want and not os.path.exists(os.path.join(dirname, name)):
            print(f"MISSING  {name} (not in the manifest either)")
            missing += 1
    if bad or missing:
        print(f"{checked} files match, {bad} differ, {missing} required files missing")
        return 1
    print(f"{checked} files match their manifest size and md5")
    return 1 if enc_bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
