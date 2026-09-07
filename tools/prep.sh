#!/bin/bash
set -eu

WEIGHTS_URL="${BITNET_WEIGHTS_URL:-https://huggingface.co/merlijn70w/bitnet-kv260-weights/resolve/main}"

HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
RECORDED="$ROOT/selftest/expected_ids.json"
CHECKPOINT="${BITNET_CHECKPOINT:-microsoft/bitnet-b1.58-2B-4T}"
OUT="${BITNET_PACK_DIR:-$HOME/bitnet-kria-packed}"
REMOTE_DIR="${BITNET_DIR:-/home/ubuntu/bitnet-kria}"
TARGET=""
LOCAL=0
DUMP=0
DL=""

usage() {
    cat <<EOF
tools/prep.sh -- fetch the packed weights, check every file, and put them where the runtime reads
them.  About 1.4 GB.

  tools/prep.sh ubuntu@kria                  fetch here, check, copy to $REMOTE_DIR
  tools/prep.sh ubuntu@kria --out DIR        fetch into DIR instead of $OUT
  tools/prep.sh ubuntu@kria --remote-dir DIR copy to DIR on the board instead
  tools/prep.sh --local DIR                  fetch and check into DIR, copy nothing
  tools/prep.sh --local DIR --dump           and build refdump/ from the original checkpoint
  tools/prep.sh --help

A file already in the directory whose size and md5 are the recorded ones is not fetched again, so
this is safe to run twice, and safe to run on a directory something else already filled.

The address to fetch from is the WEIGHTS_URL line at the top of this file.  docs/weights.md says
what to put there, what every file is checked against, and how to repack from the original
checkpoint instead.
EOF
}

say()  { printf '\n== %s\n' "$*"; }
note() { printf '   %s\n' "$*"; }
die()  { printf '\nprep.sh: %s\n' "$*" >&2; exit 1; }

url_set() {
    case "$WEIGHTS_URL" in
        http://*|https://*|file://*) return 0;;
    esac
    return 1
}

at_base() { printf '%s/%s\n' "${WEIGHTS_URL%/}" "$1"; }

url_or_die() {
    url_set && return 0
    cat >&2 <<EOF

prep.sh: the address to fetch the packed weights from is empty or has no scheme, so there is
nowhere to fetch them from.  Nothing was downloaded and nothing was changed.

The address it ships with is:

    https://huggingface.co/merlijn70w/bitnet-kv260-weights/resolve/main

It is set on the only line of $HERE/prep.sh that begins with WEIGHTS_URL=.  If you have changed
that line or set BITNET_WEIGHTS_URL, give an address that starts with http://, https:// or file://;
a trailing slash is stripped for you.

File names are appended to it, so the address has to serve model3.bin, manifest.json and the seven
files beside them.  To use another one for a single run:

    BITNET_WEIGHTS_URL=https://huggingface.co/OWNER/REPO/resolve/main $0 --local DIR

docs/weights.md lists every file and what each one is checked against.
EOF
    exit 2
}

md5_of() {
    if command -v md5sum >/dev/null 2>&1; then
        md5sum "$1" | cut -d' ' -f1
    elif command -v md5 >/dev/null 2>&1; then
        md5 -q "$1"
    else
        python3 - "$1" <<'PY'
import hashlib, sys
h = hashlib.md5()
with open(sys.argv[1], "rb") as f:
    for block in iter(lambda: f.read(1 << 22), b""):
        h.update(block)
print(h.hexdigest())
PY
    fi
}

size_of() { python3 -c 'import os, sys; print(os.path.getsize(sys.argv[1]))' "$1"; }

recorded() {
    python3 - "$RECORDED" "$1" <<'PY'
import json, sys
value = json.load(open(sys.argv[1], encoding="utf-8")).get(sys.argv[2])
print("" if value is None else value)
PY
}

manifest_table() {
    python3 - "$1" <<'PY'
import json, sys
manifest = json.load(open(sys.argv[1], encoding="utf-8"))
for name, info in sorted((manifest.get("files") or {}).items()):
    print(name, info["bytes"], info["md5"])
PY
}

manifest_model_file() {
    python3 - "$1" <<'PY'
import json, sys
manifest = json.load(open(sys.argv[1], encoding="utf-8"))
print((manifest.get("stream_format") or {}).get("file") or "")
PY
}

head_table() {
    python3 - "$1" <<'PY'
import json, sys
head = json.load(open(sys.argv[1], encoding="utf-8"))
if head.get("file") and head.get("md5"):
    print(head["file"], head.get("bytes", -1), head["md5"])
if head.get("scale_file") and head.get("scale_md5"):
    print(head["scale_file"], head.get("scale_bytes", -1), head["scale_md5"])
PY
}

lookup() {
    printf '%s\n' "$2" | awk -v want="$1" '$1 == want { print $2, $3; hit = 1 } END { exit !hit }'
}

download() {
    if [ "$DL" = curl ]; then
        if [ "$2" = resume ]; then
            curl -fL --retry 3 --retry-delay 2 --progress-bar -C - -o "$OUT/$1" "$3"
        else
            curl -fL --retry 3 --retry-delay 2 --progress-bar -o "$OUT/$1" "$3"
        fi
    elif [ "$DL" = wget ]; then
        if [ "$2" = resume ]; then
            wget -c -O "$OUT/$1" "$3"
        else
            wget -O "$OUT/$1" "$3"
        fi
    else
        return 1
    fi
}

matches() {
    local name="$1" want_bytes="$2" want_md5="$3" got_bytes got_md5
    [ -f "$OUT/$name" ] || return 1
    got_bytes="$(size_of "$OUT/$name")"
    if [ "$want_bytes" -ge 0 ] && [ "$got_bytes" != "$want_bytes" ]; then
        note "$name is $got_bytes bytes, the recorded size is $want_bytes"
        return 1
    fi
    got_md5="$(md5_of "$OUT/$name")"
    if [ "$got_md5" != "$want_md5" ]; then
        note "$name has md5 $got_md5, the recorded md5 is $want_md5"
        return 1
    fi
    return 0
}

get() {
    local name="$1" want_bytes="$2" want_md5="$3" url="$4" attempt
    if matches "$name" "$want_bytes" "$want_md5"; then
        note "$name is already here, $(size_of "$OUT/$name") bytes, md5 $want_md5"
        return 0
    fi
    url_or_die
    [ -n "$DL" ] || die "neither curl nor wget is on this machine, and one of them fetches the files"
    for attempt in resume fresh; do
        if [ "$attempt" = fresh ]; then
            note "fetching $name again from the start"
            rm -f "$OUT/$name"
        fi
        note "$url"
        if download "$name" "$attempt" "$url"; then
            if matches "$name" "$want_bytes" "$want_md5"; then
                note "$name $(size_of "$OUT/$name") bytes, md5 $want_md5"
                return 0
            fi
        fi
    done
    rm -f "$OUT/$name"
    die "$name did not arrive with the size and md5 this repository recorded, so it was deleted.
   Either $WEIGHTS_URL is not serving the files every number here was measured on, or the download
   is being corrupted.  Nothing after $name was fetched."
}

while [ $# -gt 0 ]; do
    case "$1" in
        --local)      LOCAL=1; OUT="${2:?--local needs a directory}"; shift 2;;
        --out)        OUT="${2:?--out needs a directory}"; shift 2;;
        --remote-dir) REMOTE_DIR="${2:?--remote-dir needs a directory}"; shift 2;;
        --dump)       DUMP=1; shift;;
        -h|--help)    usage; exit 0;;
        -*)           die "unknown option $1 (tools/prep.sh --help lists them)";;
        *)            TARGET="$1"; shift;;
    esac
done
if [ $LOCAL = 0 ] && [ -z "$TARGET" ]; then
    usage >&2
    exit 2
fi

[ -f "$RECORDED" ] || die "$RECORDED is missing, and it holds the md5s every fetched file is checked against"
mkdir -p "$OUT"
OUT="$(cd "$OUT" && pwd)"
if command -v curl >/dev/null 2>&1; then
    DL=curl
elif command -v wget >/dev/null 2>&1; then
    DL=wget
fi

say "the packed weights go into $OUT"

MANIFEST_MD5="$(recorded manifest_md5)"
[ -n "$MANIFEST_MD5" ] || die "$RECORDED has no manifest_md5, and without it nothing can be checked"

say "manifest.json, which every other file is checked against"
get manifest.json -1 "$MANIFEST_MD5" "$(at_base manifest.json)"

TABLE="$(manifest_table "$OUT/manifest.json")"
MODEL_FILE="$(manifest_model_file "$OUT/manifest.json")"
[ -n "$MODEL_FILE" ] || die "manifest.json names no stream_format file"

MODEL3_MD5="$(recorded model3_md5)"
if [ -n "$MODEL3_MD5" ]; then
    if [ "$(lookup "$MODEL_FILE" "$TABLE" | cut -d' ' -f2)" != "$MODEL3_MD5" ]; then
        die "manifest.json matched its own recorded md5, but its md5 for $MODEL_FILE is not the
   $MODEL3_MD5 that selftest/expected_ids.json records.  Refusing to fetch."
    fi
    note "manifest.json agrees with the recorded md5 for $MODEL_FILE"
fi

SNAPSHOT="$(recorded tokenizer_snapshot)"
if [ -z "$SNAPSHOT" ]; then
    SNAPSHOT=main
    note "no tokenizer_snapshot is recorded, so tokenizer.json comes from $CHECKPOINT at main"
fi
TOKENIZER_URL="https://huggingface.co/$CHECKPOINT/resolve/$SNAPSHOT/tokenizer.json"

for name in "$MODEL_FILE" norms.bin embed_bf16.bin head_i8.bin head_scale.bin tokenizer.json; do
    entry="$(lookup "$name" "$TABLE")" || die "manifest.json records no size and md5 for $name"
    say "$name"
    if [ "$name" = tokenizer.json ]; then
        get "$name" $entry "$TOKENIZER_URL"
    else
        get "$name" $entry "$(at_base "$name")"
    fi
done

say "head3_t.bin and head_t_scale.bin, which the fabric output head streams"
HEAD3_MD5="$(recorded head3_t_md5)"
if [ ! -f "$OUT/head_t.json" ]; then
    url_set && note "$(at_base head_t.json)"
    if url_set && download head_t.json fresh "$(at_base head_t.json)"; then
        note "head_t.json fetched"
    else
        rm -f "$OUT/head_t.json"
    fi
fi
if [ -f "$OUT/head_t.json" ]; then
    HTABLE="$(head_table "$OUT/head_t.json")"
    HEAD_FILE="$(printf '%s\n' "$HTABLE" | awk 'NR == 1 { print $1 }')"
    [ -n "$HEAD_FILE" ] || die "head_t.json names no head file and no md5"
    if [ -n "$HEAD3_MD5" ]; then
        if [ "$(lookup "$HEAD_FILE" "$HTABLE" | cut -d' ' -f2)" != "$HEAD3_MD5" ]; then
            die "head_t.json's md5 for $HEAD_FILE is not the $HEAD3_MD5 that
   selftest/expected_ids.json records.  Refusing to fetch a head this repository never measured."
        fi
        note "head_t.json agrees with the recorded md5 for $HEAD_FILE"
    fi
    HEAD_MISSING=0
    for name in $(printf '%s\n' "$HTABLE" | awk '{ print $1 }'); do
        [ -f "$OUT/$name" ] || HEAD_MISSING=1
    done
    if [ $HEAD_MISSING = 1 ] && ! url_set; then
        note "head3_t.bin or head_t_scale.bin is missing and there is no address to fetch it from."
        note "The runtime still answers with --head arm, at about 52 ms more a generated token."
    else
        for name in $(printf '%s\n' "$HTABLE" | awk '{ print $1 }'); do
            entry="$(lookup "$name" "$HTABLE")"
            get "$name" $entry "$(at_base "$name")"
        done
    fi
else
    note "head_t.json is not there, so head3_t.bin and head_t_scale.bin were not fetched."
    note "The runtime still answers with --head arm, at about 52 ms more a generated token."
fi

say "checking every file against the manifest it was written with"
python3 "$HERE/checkfiles.py" "$OUT" --encoding base3

du -sh "$OUT"

if [ $DUMP = 1 ]; then
    say "the numpy reference's stage dumps, which are built from the original checkpoint"
    if python3 -c 'import huggingface_hub' 2>/dev/null; then
        python3 - "$CHECKPOINT" <<'PY'
import sys
from huggingface_hub import snapshot_download
print("snapshot:", snapshot_download(sys.argv[1], allow_patterns=["model.safetensors", "config.json"]))
PY
        ( cd "$OUT" && python3 "$HERE/ref_model.py" --dump refdump ) || {
            echo
            echo "ref_model.py --dump failed.  Everything else is fetched and usable; selftest.sh"
            echo "reports the stage check as SKIPPED rather than failing.  It needs numpy and"
            echo "tokenizers and takes a few minutes; by hand:"
            echo "  cd $OUT && python3 $HERE/ref_model.py --dump refdump"
        }
    else
        echo "huggingface_hub is not installed here, and --dump reads the original checkpoint."
        echo "  pip install huggingface_hub numpy tokenizers   and run this again."
        exit 2
    fi
fi

if [ $LOCAL = 1 ]; then
    say "fetched into $OUT; copy it to the board yourself:"
    echo "  rsync -a --info=progress2 $OUT/ $REMOTE_DIR/"
    exit 0
fi

say "copying to $TARGET:$REMOTE_DIR (about 1.4 GB)"
ssh "$TARGET" "mkdir -p '$REMOTE_DIR'"
rsync -a --info=progress2 "$OUT/" "$TARGET:$REMOTE_DIR/"

cat <<EOF

Done.  On the board:

  git clone <this repository> && cd bitnet-kv260
  sudo ./setup.sh
  ./selftest.sh
  ./chat.sh "What is the capital of France?"

The first run after this copy reads the model from storage rather than the page cache: on the
reference board that was 8.13 s of setup at 51 MB/s against 0.87 s at 482 MB/s warm.  Throw the
first run's timing away.
EOF
