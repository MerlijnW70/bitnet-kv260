#!/bin/bash
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
DIR="${BITNET_DIR:-/home/ubuntu/bitnet-kria}"
APP=kv260-bitnet
QUIET=0
CLOCK_ONLY=0
case "${1:-}" in
    --quiet) QUIET=1;;
    --clock) CLOCK_ONLY=1;;
    "") ;;
    *) echo "usage: $0 [--quiet | --clock]" >&2; exit 2;;
esac

pass=0; fail=0; skip=0
ok()   { pass=$((pass+1)); [ $QUIET = 1 ] || printf 'PASS  %-22s %s\n' "$1" "$2"; }
bad()  { fail=$((fail+1)); printf 'FAIL  %-22s %s\n' "$1" "$2"; [ -n "${3:-}" ] && printf '      %-22s fix: %s\n' "" "$3"; }
warn() { skip=$((skip+1)); [ $QUIET = 1 ] || printf 'SKIP  %-22s %s\n' "$1" "$2"; }
note() { [ $QUIET = 1 ] || printf '      %-22s %s\n' "" "$1"; }
loud() {
    fail=$((fail+1))
    printf '\n'
    printf '========================================================================\n'
    printf 'FAIL  fabric clock\n'
    printf '\n'
    for l in "$@"; do if [ -n "$l" ]; then printf '  %s\n' "$l"; else printf '\n'; fi; done
    printf '========================================================================\n'
    printf '\n'
}

read_pl0_ref() {
    line=$(grep -E ' pl0_ref ' /sys/kernel/debug/clk/clk_summary 2>/dev/null)
    [ -n "$line" ] || line=$(sudo grep -E ' pl0_ref ' /sys/kernel/debug/clk/clk_summary 2>/dev/null)
    printf '%s\n' "$line" \
      | awk '{for (i = 2; i <= NF; i++) if ($i ~ /^[0-9]{8,}$/) { print $i; exit }}' | head -1
}

clock_check() {
    CLK=$(read_pl0_ref)
    case "${CLK:-}" in
        249999998)
            ok "fabric clock" "pl0_ref $CLK Hz (250 MHz)"
            return 0;;
        99999999)
            loud "pl0_ref reads 99999999 Hz. The fabric is running at 100 MHz, not 250 MHz." \
                 "" \
                 "The bitstream was programmed WITHOUT its device-tree overlay, so the PL clock" \
                 "was left at the boot firmware's rate. Every answer this board gives will still" \
                 "be correct and every one will take 2.5 times as long. Nothing else on this" \
                 "board reports an error, which is why this check is the first one here." \
                 "" \
                 "Load the bitstream and the overlay together, then run ./doctor.sh again:" \
                 "" \
                 "    sudo ./setup.sh --fabric" \
                 "" \
                 "or, through the Kria app:" \
                 "" \
                 "    sudo xmutil unloadapp" \
                 "    sudo xmutil loadapp $APP" \
                 "    sudo firmware/install-firmware.sh --verify"
            return 1;;
        "")
            loud "pl0_ref could not be read from /sys/kernel/debug/clk/clk_summary." \
                 "" \
                 "Until it reads 249999998 there is no way to tell a 250 MHz fabric from a" \
                 "100 MHz one, and a 100 MHz fabric gives right answers 2.5 times slower with" \
                 "no error anywhere. debugfs is root-only, so run this where sudo works:" \
                 "" \
                 "    sudo grep pl0_ref /sys/kernel/debug/clk/clk_summary" \
                 "" \
                 "If the file itself is absent, this is not a Zynq UltraScale+ running the" \
                 "Xilinx kernel and nothing else in this repository will work either."
            return 1;;
        *)
            loud "pl0_ref reads $CLK Hz. It must read 249999998 (250 MHz)." \
                 "" \
                 "This is neither the 250 MHz the design was timed at nor the 100 MHz of a" \
                 "missing overlay, so something other than $APP is in the fabric, or the" \
                 "overlay in firmware/ is not the one this bitstream was built with. Every" \
                 "measurement in results/ was taken at 249999998 Hz." \
                 "" \
                 "Reload the bitstream WITH its overlay:" \
                 "" \
                 "    sudo ./setup.sh --fabric"
            return 1;;
    esac
}

if [ $CLOCK_ONLY = 1 ]; then
    clock_check
    exit $?
fi

[ $QUIET = 1 ] || cat <<'EOF'
bitnet-kv260 doctor -- the reference board is a KV260 Vision AI Starter Kit (xck26-sfvc784-2LV-c)
running Ubuntu 22.04.4 for Kria, kernel 5.15.0-1027-xilinx-zynqmp, with cma=1000M.
It reads and changes nothing. Every value it compares against is recorded in results/.
[this script itself has not been run on the reference board; the values it checks were measured there]

The fabric clock is checked first, because a fabric left at 100 MHz is the one failure that
answers correctly and reports nothing.
EOF

clock_check

if [ -r /sys/class/fpga_manager/fpga0/state ]; then
    ST=$(cat /sys/class/fpga_manager/fpga0/state)
    [ "$ST" = operating ] && ok fpga "fpga0 state $ST" \
        || bad fpga "fpga0 state $ST" "sudo ./setup.sh --fabric"
else
    bad fpga "/sys/class/fpga_manager/fpga0/state missing" "this is not a Zynq UltraScale+ running the Xilinx kernel"
fi
if [ -d /sys/kernel/config/device-tree/overlays/full ]; then ok overlay "/sys/kernel/config/device-tree/overlays/full present"
else warn overlay "no live overlay -- the fabric clock check above is the one that matters"; fi

KERN="$(uname -r)"
MACH="$(uname -m)"
if [ "$MACH" = aarch64 ]; then ok kernel "$KERN $MACH"; else bad kernel "$KERN $MACH -- not aarch64" "this runs on a Kria KV260 only"; fi
[ "$KERN" = "5.15.0-1027-xilinx-zynqmp" ] || note "the reference board runs 5.15.0-1027-xilinx-zynqmp; yours is $KERN, which is untested here"
if [ -r /proc/device-tree/model ]; then
    note "device tree model: $(tr -d '\0' < /proc/device-tree/model)"
fi
MEM_MB=$(awk '/MemTotal/ {printf "%d", $2/1024}' /proc/meminfo 2>/dev/null)
MEM_MB=${MEM_MB:-0}
if [ "$MEM_MB" -ge 3500 ]; then ok ram "${MEM_MB} MB total (reference board 3911 MB)"
else bad ram "${MEM_MB} MB total, the model needs about 1.5 GB of files and 520 MB contiguous" "a 4 GB KV260"; fi
FREE_G=$(df -BG --output=avail / 2>/dev/null | tail -1 | tr -dc '0-9')
if [ "${FREE_G:-0}" -ge 8 ]; then ok disk "${FREE_G} GB free on /"
else bad disk "${FREE_G:-?} GB free on /, the packed weights are about 1.4 GB" "free space or a larger card"; fi

CMA_T=$(awk '/CmaTotal/ {print $2}' /proc/meminfo)
CMA_F=$(awk '/CmaFree/  {print $2}' /proc/meminfo)
if [ -z "${CMA_T:-}" ]; then
    bad cma "/proc/meminfo has no CmaTotal -- this kernel has no CMA" "a kernel with CONFIG_CMA"
elif [ "$CMA_T" -ge 1024000 ]; then
    ok cma "CmaTotal ${CMA_T} kB, CmaFree ${CMA_F} kB"
    note "the three buffers take 637,534,208 B; on the reference board that leaves CmaFree 14852 kB"
else
    bad cma "CmaTotal ${CMA_T} kB, less than the 1024000 kB the buffers need" \
            "add cma=1000M to the kernel command line and reboot (setup.sh --bootargs prints how)"
fi

if lsmod | grep -q '^u_dma_buf'; then ok u-dma-buf "module loaded"
elif modinfo u-dma-buf >/dev/null 2>&1; then bad u-dma-buf "installed but not loaded" "sudo modprobe u-dma-buf"
else bad u-dma-buf "not installed" "sudo ./setup.sh (it builds it from github.com/ikwzm/udmabuf)"; fi

want_size() { case "$1" in udmabuf0) echo 545259520;; udmabuf1) echo 8388608;; udmabuf2) echo 83886080;; esac; }
for b in udmabuf0 udmabuf1 udmabuf2; do
    sysfs=/sys/class/u-dma-buf/$b
    if [ ! -d "$sysfs" ]; then
        [ "$b" = udmabuf2 ] \
          && warn "$b" "absent -- the fabric output head cannot run; --head arm still works, at about 52 ms a token more" \
          || bad "$b" "absent" "add it to /etc/modprobe.d/u-dma-buf.conf and reload the module"
        continue
    fi
    phys=$(tr -d ' \t\n' < "$sysfs/phys_addr"); size=$(tr -d ' \t\n' < "$sysfs/size"); want=$(want_size "$b")
    case "$phys" in
        0x*|0X*) physd=$((16#${phys#0[xX]}));;
        *)       physd=$((10#$phys));;
    esac
    size=$((10#$size))
    if [ "$size" -lt "$want" ]; then
        bad "$b" "size $size, wanted $want" "options u-dma-buf udmabuf0=545259520 udmabuf1=8388608 udmabuf2=83886080"
    elif [ "$physd" -ge 4294967296 ]; then
        bad "$b" "phys $phys is at or above 4 GiB; the runtime maps DDR_LOW" "reboot to re-place the CMA reservation"
    else
        ok "$b" "phys $phys size $size (below 4 GiB)"
    fi
done

BIT="$HERE/firmware/$APP.bit.bin"
if [ -f "$BIT" ]; then
    M=$(md5sum "$BIT" | cut -d' ' -f1)
    [ "$M" = 1f7e4fcc5ee4c905ebcfb57f60fe9675 ] \
      && ok bitstream "md5 $M (the one every number in results/ was measured on)" \
      || bad bitstream "md5 $M, expected 1f7e4fcc5ee4c905ebcfb57f60fe9675" "re-clone, or rebuild with hardware/build-ffn.tcl and expect different numbers"
    seen=0
    for L in /lib/firmware/$APP.bit.bin /lib/firmware/xilinx/$APP/$APP.bit.bin; do
        [ -f "$L" ] || continue
        seen=1
        LM=$(md5sum "$L" | cut -d' ' -f1)
        [ "$LM" = "$M" ] && ok firmware "$L is the same file" \
                         || bad firmware "$L differs from firmware/$APP.bit.bin" \
                                         "sudo ./setup.sh --fabric, or sudo firmware/install-firmware.sh -- the loader looks the file up BY NAME, so a stale copy loads silently"
    done
    [ $seen = 1 ] || warn firmware "neither /lib/firmware/$APP.bit.bin nor /lib/firmware/xilinx/$APP/$APP.bit.bin is installed yet"
else
    bad bitstream "$BIT missing" "git clone brings it; it is 7,797,692 B in the repository"
fi

if [ -f "$DIR/manifest.json" ]; then
    miss=""
    for f in model3.bin norms.bin embed_bf16.bin head_i8.bin head_scale.bin tokenizer.json; do
        [ -f "$DIR/$f" ] || miss="$miss $f"
    done
    [ -z "$miss" ] && ok weights "model3.bin norms.bin embed_bf16.bin head_i8.bin head_scale.bin tokenizer.json all present" \
                   || bad weights "missing:$miss" "sudo ./setup.sh --weights, or tools/prep.sh on a PC"
    if [ -f "$DIR/head3_t.bin" ] && [ -f "$DIR/head_t_scale.bin" ]; then
        ok head "head3_t.bin and head_t_scale.bin present -- the fabric head can run"
    else
        warn head "no head3_t.bin -- generation falls back to the ARM head, about 52 ms a token slower"
    fi
    if [ -x "$DIR/bitnet_kria" ]; then ok runtime "$DIR/bitnet_kria built"
    else bad runtime "$DIR/bitnet_kria missing" "sudo ./setup.sh --build"; fi
else
    bad manifest "$DIR/manifest.json missing -- no packed model in $DIR" "sudo ./setup.sh --weights"
fi

if [ -f "$DIR/manifest.json" ] && command -v python3 >/dev/null 2>&1; then
    OUT=$(python3 "$HERE/tools/checkfiles.py" "$DIR" --encoding base3 2>&1)
    ENCLINE=$(echo "$OUT" | grep -m1 '^stream_format ')
    if echo "$OUT" | grep -q '^ENCODING '; then
        bad encoding "${ENCLINE:-no stream_format in the manifest}" \
            "python3 tools/pack_model.py OUT --encoding base3 -- this bitstream has no two-bit path"
    else
        ok encoding "$ENCLINE"
    fi
    GOOD=$(echo "$OUT" | grep -m1 'files match their manifest size and md5')
    if [ -n "$GOOD" ]; then ok checksums "$GOOD"
    else bad checksums "$(echo "$OUT" | grep -E '^(MISSING|SIZE|MD5) ' | head -3)" \
                       "sudo ./setup.sh --weights re-fetches the file named above"; fi
else
    warn checksums "no python3 or no manifest -- the encoding and the file md5s were not checked"
fi

LOAD=$(cut -d' ' -f1 /proc/loadavg)
TOP=$(ps -eo pcpu=,comm= --sort=-pcpu | grep -vE ' (ps|awk|grep|sort|bash|sh|sudo|doctor\.sh|setup\.sh|selftest\.sh)$' | head -1)
BUSYP=$(echo "$TOP" | awk '{print $1+0}')
BUSY=$(echo "$TOP" | awk '{printf "%s %s%%", $2, $1}')
if awk "BEGIN{exit !(${BUSYP:-0} > 10)}"; then
    bad quiet "load average $LOAD, busiest process $BUSY" \
              "sudo systemctl restart polkit -- a quarter of a core costs about 1 tok/s, a whole core costs 15"
else
    ok quiet "load average $LOAD, busiest process $BUSY"
fi
GOV=$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor 2>/dev/null || echo "?")
KHZ=$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq 2>/dev/null || echo "?")
note "governor $GOV, cpu0 at $KHZ kHz (the reference board: userspace, all four at 1333333 kHz)"

NORMAL="${SUDO_USER:-$(id -un)}"
if [ "$NORMAL" = "$(id -un)" ]; then
    astok() { python3 -c 'import tokenizers'; }
else
    astok() { sudo -u "$NORMAL" python3 -c 'import tokenizers'; }
fi
if astok >/dev/null 2>&1; then
    ok tokenizers "importable as $NORMAL"
else
    bad tokenizers "not importable as $NORMAL" "as $NORMAL (not under sudo): pip3 install --user tokenizers"
fi

command -v dtc      >/dev/null 2>&1 && ok dtc      "$(dtc --version 2>&1 | head -1)" || bad dtc "missing" "sudo apt-get install device-tree-compiler"
command -v fpgautil >/dev/null 2>&1 && ok fpgautil "$(command -v fpgautil)"          || bad fpgautil "missing" "it ships with the Kria Ubuntu image"
command -v gcc      >/dev/null 2>&1 && ok gcc      "$(gcc --version | head -1)"      || bad gcc "missing" "sudo apt-get install build-essential"
command -v iverilog >/dev/null 2>&1 && ok iverilog "$(iverilog -V 2>&1 | head -1)"   || warn iverilog "absent -- the RTL self-test is skipped (it needs no board and no weights)"

echo
echo "$pass passed, $fail failed, $skip skipped."
[ $fail = 0 ] || echo "Fix the FAIL lines above before trusting any number this repository prints."
exit $([ $fail = 0 ] && echo 0 || echo 1)
