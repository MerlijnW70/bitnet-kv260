#!/bin/bash
set -eu
HERE="$(cd "$(dirname "$0")" && pwd)"
SELF="$HERE/$(basename "$0")"
APP=kv260-bitnet
DEST=/lib/firmware/xilinx/$APP

need_root() { [ "$(id -u)" = 0 ] || { echo "run this with sudo" >&2; exit 1; }; }

usage() {
    cat <<EOF
usage: sudo $SELF [--verify | --remove]

  (no argument)  compile $APP.dtso, then install $APP.bit.bin, $APP.dtbo and
                 shell.json into $DEST
  --verify       read fpga0's state and pl0_ref after the app has been loaded
  --remove       unload the app and delete $DEST
EOF
}

verify() {
    st=$(cat /sys/class/fpga_manager/fpga0/state 2>/dev/null || echo unreadable)
    clk=$(grep -E ' pl0_ref ' /sys/kernel/debug/clk/clk_summary 2>/dev/null \
          | awk '{for (i = 2; i <= NF; i++) if ($i ~ /^[0-9]{8,}$/) { print $i; exit }}' | head -1)
    printf 'fpga0 state  %s\n' "$st"
    printf 'pl0_ref      %s Hz\n' "${clk:-unreadable}"
    [ "$st" = operating ] || { echo "fpga0 is '$st', not 'operating': the bitstream is not in the fabric" >&2; return 1; }
    case "${clk:-}" in
        249999998) echo "the fabric is at 250 MHz"; return 0;;
        99999999)  echo "the fabric is at 100 MHz: the overlay did not take, so every answer is right and 2.5x slow" >&2; return 1;;
        "")        echo "could not read /sys/kernel/debug/clk/clk_summary" >&2; return 1;;
        *)         echo "pl0_ref is $clk Hz, expected 249999998" >&2; return 1;;
    esac
}

case "${1:-}" in
    -h|--help) usage; exit 0;;
    --verify)  need_root; if verify; then exit 0; else exit 1; fi;;
    --remove)
        need_root
        xmutil unloadapp >/dev/null 2>&1 || true
        rm -rf "$DEST"
        echo "removed $DEST"
        xmutil listapps || true
        exit 0;;
    "") ;;
    *) usage >&2; exit 1;;
esac

need_root
cd "$HERE"
for f in "$APP.bit.bin" "$APP.dtso" shell.json; do
    [ -f "$f" ] || { echo "$HERE/$f is missing" >&2; exit 1; }
done
command -v dtc >/dev/null || { echo "no dtc: sudo apt-get install device-tree-compiler" >&2; exit 1; }

FW=$(sed -n 's/.*firmware-name[[:space:]]*=[[:space:]]*"\([^"]*\)".*/\1/p' "$APP.dtso" | head -1)
[ "$FW" = "$APP.bit.bin" ] || {
    echo "firmware-name in $APP.dtso is '$FW', not '$APP.bit.bin'." >&2
    echo "The FPGA manager looks the bitstream up by that name, so loading would leave the previous" >&2
    echo "bitstream in the fabric with no error. Fix the .dtso before installing." >&2
    exit 1
}
echo "firmware-name in $APP.dtso is '$FW', which is the file name being installed"

dtc -@ -I dts -O dtb -o "$APP.dtbo" "$APP.dtso"
install -d -m 0755 "$DEST"
install -m 0644 "$APP.bit.bin" "$APP.dtbo" shell.json "$DEST/"
echo "installed into $DEST:"
ls -l "$DEST"
xmutil listapps || echo "xmutil listapps failed -- use the fpgautil path in setup.sh --fabric instead"

cat <<EOF

next:
  sudo xmutil unloadapp
  sudo xmutil loadapp $APP
  sudo $SELF --verify

--verify is this script checking the clock afterwards: fpga0 must read 'operating' and pl0_ref must
read 249999998 Hz. At 99999999 Hz the overlay did not take and the fabric is running at 100 MHz,
which gives right answers 2.5x slower with no error anywhere. ../doctor.sh refuses to pass on that
value too. If loading through xmutil misbehaves, ../setup.sh --fabric loads the same two files with
fpgautil instead; see ../docs/troubleshooting.md.
EOF
