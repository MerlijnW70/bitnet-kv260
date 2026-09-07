#!/bin/bash
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
DIR="${BITNET_DIR:-/home/ubuntu/bitnet-kria}"
NORMAL="${SUDO_USER:-ubuntu}"
UDMABUF_REPO="${UDMABUF_REPO:-https://github.com/ikwzm/udmabuf}"
UDMABUF_SRC="${UDMABUF_SRC:-/home/$NORMAL/udmabuf}"
CONF=/etc/modprobe.d/u-dma-buf.conf
CONF_LINE="options u-dma-buf udmabuf0=545259520 udmabuf1=8388608 udmabuf2=83886080"
GCC_LINE="gcc -O3 -Wall -Wextra -std=c11 -march=armv8-a+crc -ffp-contract=off -o bitnet_kria bitnet_kria.c -lm -lpthread"
REBOOT_NEEDED=10

usage() {
    cat <<EOF
setup.sh -- everything on the board that a script can do, in one command.

  sudo ./setup.sh                 do as much as this boot allows, then say what is next
  sudo ./setup.sh --stage1        packages, u-dma-buf, the module config, the kernel command line
  sudo ./setup.sh --stage2        the fabric, the weights, the runtime, the checks
  sudo ./setup.sh --fabric        reload the bitstream with its overlay and verify the clock
  sudo ./setup.sh --weights       fetch the packed weights into $DIR and check every file
  sudo ./setup.sh --build         rebuild bitnet_kria only
  sudo ./setup.sh --bootargs      print how to add cma=1000M on this image, change nothing

Every step says what it is about to change before it changes it and skips work already done.
docs/weights.md says where the weights come from, docs/overview.md what the one reboot is for.
EOF
}

say()   { printf '\n== %s\n' "$*"; }
run()   { printf '   $ %s\n' "$*"; "$@"; }
unver() { printf '   [UNVERIFIED on the reference board] %s\n' "$*"; }
die()   { printf '\nsetup.sh: %s\n' "$*" >&2; exit 1; }
need_root() { [ "$(id -u)" = 0 ] || die "run this with sudo"; }

cma_kb() { awk '/CmaTotal/ {print $2}' /proc/meminfo; }

stage1() {
    need_root
    say "packages"
    unver "apt-get is not exercised in results/; the package names are the ones this build needs"
    run apt-get update -qq
    run apt-get install -y build-essential device-tree-compiler python3-pip curl "linux-headers-$(uname -r)" \
        || die "apt-get failed; without linux-headers-$(uname -r) u-dma-buf cannot be built"
    say "the tokenizer, as $NORMAL (the runtime runs as root, the tokenizer must not)"
    run sudo -u "$NORMAL" pip3 install --user --quiet tokenizers \
        || die "pip3 install tokenizers failed for $NORMAL"

    say "u-dma-buf (BSD 2-Clause, Ichiro Kawazome) -- built here, never redistributed by this repository"
    if modinfo u-dma-buf >/dev/null 2>&1; then
        printf '   already installed: %s\n' "$(modinfo -n u-dma-buf)"
    else
        unver "the clone/make/install below has not been run on the reference board"
        [ -d "$UDMABUF_SRC" ] || run sudo -u "$NORMAL" git clone --depth 1 "$UDMABUF_REPO" "$UDMABUF_SRC" \
            || die "could not clone $UDMABUF_REPO into $UDMABUF_SRC"
        ( cd "$UDMABUF_SRC" && run make ) || die "u-dma-buf did not build against the running kernel headers"
        run install -D -m 0644 "$UDMABUF_SRC/u-dma-buf.ko" "/lib/modules/$(uname -r)/extra/u-dma-buf.ko" \
            || die "could not install u-dma-buf.ko"
        run depmod -a
    fi

    say "the buffer sizes ($CONF)"
    if [ -f "$CONF" ] && grep -qF "$CONF_LINE" "$CONF"; then
        printf '   already: %s\n' "$CONF_LINE"
    else
        [ -f "$CONF" ] && { printf '   was: %s\n' "$(cat "$CONF")"; cp -a "$CONF" "$CONF.bak-bitnet-kv260"; }
        printf '   now: %s\n' "$CONF_LINE"
        echo "$CONF_LINE" > "$CONF"
    fi
    mkdir -p /etc/modules-load.d
    grep -q '^u-dma-buf$' /etc/modules-load.d/u-dma-buf.conf 2>/dev/null \
        || echo u-dma-buf > /etc/modules-load.d/u-dma-buf.conf

    say "the kernel command line"
    CT=$(cma_kb); CT=${CT:-0}
    if [ "$CT" -ge 1024000 ]; then
        printf '   CmaTotal is already %s kB; no reboot needed\n' "$CT"
        return 0
    fi
    bootargs
    return $REBOOT_NEEDED
}

bootargs() {
    cat <<EOF
   This board's CMA reservation is $(cma_kb) kB and the three buffers need 1024000 kB.
   Add   cma=1000M   to the kernel command line and reboot.  This script does NOT edit it:
   the mechanism differs between Kria images and getting it wrong makes a board that will not boot.
   [UNVERIFIED] the reference board was configured before these files existed; the current command
   line is printed below so you can see what to add to.

   current: $(cat /proc/cmdline)

   Look for whichever of these your image uses, add cma=1000M to the kernel arguments, and reboot:
EOF
    for f in /boot/firmware/cmdline.txt /etc/default/grub /boot/firmware/boot.scr /etc/kernel/cmdline \
             /boot/extlinux/extlinux.conf /etc/default/flash-kernel; do
        [ -e "$f" ] && printf '     present: %s\n' "$f"
    done
    cat <<'EOF'
   Then check with:  grep CmaTotal /proc/meminfo     (it must read 1024000 kB or more)
EOF
}

load_module() {
    say "u-dma-buf"
    lsmod | grep -q '^u_dma_buf' || run modprobe u-dma-buf || die "modprobe u-dma-buf failed"
    for b in udmabuf0 udmabuf1 udmabuf2; do
        s=/sys/class/u-dma-buf/$b
        [ -d "$s" ] || { printf '   %s ABSENT\n' "$b"; continue; }
        printf '   %s phys %s size %s\n' "$b" "$(cat "$s/phys_addr")" "$(cat "$s/size")"
    done
}

fabric() {
    need_root
    say "the bitstream and its overlay"
    cd "$HERE/firmware" || die "no firmware directory"
    [ -f kv260-bitnet.bit.bin ] || die "firmware/kv260-bitnet.bit.bin is missing"
    run dtc -@ -I dts -O dtb -o kv260-bitnet.dtbo kv260-bitnet.dtso || die "dtc failed"
    [ -d /sys/kernel/config/device-tree/overlays/full ] && run rmdir /sys/kernel/config/device-tree/overlays/full
    fpgautil -R >/dev/null 2>&1
    xmutil unloadapp >/dev/null 2>&1
    run fpgautil -b kv260-bitnet.bit.bin -o kv260-bitnet.dtbo || die "fpgautil could not load the bitstream"
    st=$(cat /sys/class/fpga_manager/fpga0/state)
    clk=$(grep -E ' pl0_ref ' /sys/kernel/debug/clk/clk_summary 2>/dev/null \
          | awk '{for (i = 2; i <= NF; i++) if ($i ~ /^[0-9]{8,}$/) { print $i; exit }}' | head -1)
    printf '   fpga0 state %s   pl0_ref %s Hz\n' "$st" "$clk"
    [ "$st" = operating ] || die "fpga0 is '$st', not 'operating'"
    [ "$clk" = 249999998 ] || die "pl0_ref is $clk Hz, not 249999998. The overlay did not take, so the
   fabric would run at 100 MHz: right answers, 2.5x slow, no error anywhere. Refusing to continue."
    printf '   the fabric is at 250 MHz\n'
}

weights() {
    need_root
    say "the packed weights, into $DIR"
    unver "fetching them over the network; the reference board's copy arrived over ssh from a PC"
    [ -d "$DIR" ] || run install -d -o "$NORMAL" -g "$NORMAL" "$DIR"
    if [ -n "${BITNET_WEIGHTS_URL:-}" ]; then
        run sudo -u "$NORMAL" env BITNET_WEIGHTS_URL="$BITNET_WEIGHTS_URL" \
            "$HERE/tools/prep.sh" --local "$DIR" \
            || die "tools/prep.sh stopped, and the lines above say why. Nothing after this was run."
    else
        run sudo -u "$NORMAL" "$HERE/tools/prep.sh" --local "$DIR" \
            || die "tools/prep.sh stopped, and the lines above say why. Nothing after this was run."
    fi
}

build() {
    need_root
    say "the runtime"
    [ -d "$DIR" ] || run install -d -o "$NORMAL" -g "$NORMAL" "$DIR"
    for f in bitnet_kria.c refstages.h bitnet_chat.py bitnet_chat.jinja edge_monitor.py \
             edge_alarms.json ina260-sample.py ids-power.txt power-bitnet.sh power-bitnetcpp.sh; do
        install -m 0644 "$HERE/runtime/$f" "$DIR/$f"
    done
    chmod +x "$DIR/power-bitnet.sh" "$DIR/power-bitnetcpp.sh"
    printf '   copied the runtime sources into %s\n' "$DIR"
    ( cd "$DIR" && printf '   $ %s\n' "$GCC_LINE" && eval "$GCC_LINE" ) || die "gcc failed"
    chown "$NORMAL:$NORMAL" "$DIR/bitnet_kria"
    printf '   built %s (md5 %s)\n' "$DIR/bitnet_kria" "$(md5sum "$DIR/bitnet_kria" | cut -d' ' -f1)"
}

stage2() {
    need_root
    CT=$(cma_kb); CT=${CT:-0}
    [ "$CT" -ge 1024000 ] || { bootargs; die "CMA is $CT kB; add cma=1000M to the kernel command line and reboot"; }
    printf '   [the module, fabric and build commands are ones recorded working on the reference board]\n'
    load_module
    fabric
    weights
    build
    say "checks"
    "$HERE/doctor.sh" || die "doctor.sh found something wrong; fix the FAIL lines above"
    cat <<EOF

Ready.  Next:

  ./selftest.sh                                  prove the board matches this repository's numbers
  ./chat.sh "What is the capital of France?"      one answer
  ./chat.sh                                       an interactive conversation

The first run after a file copy reads the model from storage rather than the page cache: on the
reference board that is 8.13 s of setup at 51 MB/s against 0.87 s at 482 MB/s warm, and its first
prompt is worthless.  Throw the first run away before believing any timing.
EOF
}

case "${1:---all}" in
    --stage1)   stage1; rc=$?; [ $rc = $REBOOT_NEEDED ] && { echo; echo "Now: sudo reboot, then  sudo ./setup.sh --stage2"; exit 0; }; exit $rc;;
    --stage2)   stage2;;
    --fabric)   fabric;;
    --weights)  weights;;
    --build)    build;;
    --bootargs) bootargs;;
    -h|--help)  usage;;
    --all)      stage1; rc=$?
                if [ $rc = $REBOOT_NEEDED ]; then
                    echo; echo "Now: sudo reboot, then  sudo ./setup.sh --stage2"; exit 0
                fi
                [ $rc = 0 ] || exit $rc
                stage2;;
    *) usage; exit 2;;
esac
