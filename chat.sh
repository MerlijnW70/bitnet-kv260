#!/bin/bash
set -u
DIR="${BITNET_DIR:-/home/ubuntu/bitnet-kria}"
NORMAL="${SUDO_USER:-$(id -un)}"

[ -x "$DIR/bitnet_kria" ]     || { echo "chat.sh: no $DIR/bitnet_kria -- run: sudo ./setup.sh" >&2; exit 1; }
[ -f "$DIR/tokenizer.json" ]  || { echo "chat.sh: no $DIR/tokenizer.json -- run tools/prep.sh on your PC" >&2; exit 1; }

SITE=$(sudo -u "$NORMAL" python3 -c 'import site; print(site.getusersitepackages())' 2>/dev/null)
[ -n "$SITE" ] || SITE="/home/$NORMAL/.local/lib/python3.10/site-packages"

args=()
if [ $# -eq 1 ] && [ "${1#-}" = "$1" ]; then
    args=(--prompt "$1")
else
    seen_sep=0
    for a in "$@"; do
        if [ $seen_sep = 0 ] && [ "$a" = "--" ]; then seen_sep=1; continue; fi
        if [ $seen_sep = 1 ]; then args+=(--prompt "$a"); seen_sep=2; continue; fi
        args+=("$a")
    done
fi

exec sudo env PYTHONPATH="$SITE" python3 "$DIR/bitnet_chat.py" --no-sudo --dir "$DIR" "${args[@]}"
