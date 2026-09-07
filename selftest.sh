#!/bin/bash
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
DIR="${BITNET_DIR:-/home/ubuntu/bitnet-kria}"
NORMAL="${SUDO_USER:-$(id -un)}"

usage() {
    cat <<EOF
selftest.sh -- prove this board is running what this repository measured.

  ./selftest.sh              the board tests: preconditions, weights, stage check, token ids, speed
  ./selftest.sh --rtl        add the two Icarus Verilog testbenches (no board and no weights needed;
                             the engine's takes seconds, the glue's took over ten minutes here)
  ./selftest.sh --rtl-only   the testbenches alone, on any machine with iverilog
  ./selftest.sh --quick      skip the stage check (it needs refdump/, see docs/weights.md)

The board tests refuse to start until doctor.sh --clock says the fabric is at 250 MHz.
docs/overview.md says what each step checks and how much of it has been run.
EOF
}

RTL=0; RTL_ONLY=0; QUICK=0
for a in "$@"; do
    case "$a" in
        --rtl) RTL=1;;
        --rtl-only) RTL=1; RTL_ONLY=1;;
        --quick) QUICK=1;;
        -h|--help) usage; exit 0;;
        *) usage >&2; exit 2;;
    esac
done
fails=0
step() { printf '\n=========== %s\n' "$*"; }
bad()  { fails=$((fails+1)); printf '\n*** FAILED: %s\n' "$*"; }

cat <<'EOF'
bitnet-kv260 self-test.
[step 2 was run while this repository was assembled and passed; step 4's expected ids come from
 four independent recorded runs of the reference board; steps 1 and 3 are the commands recorded
 working there, assembled into these scripts but not re-run on it]
EOF

run_rtl() {
    step "RTL testbenches (Icarus Verilog; no board, no weights)"
    command -v iverilog >/dev/null 2>&1 || { echo "no iverilog: sudo apt-get install iverilog"; return 1; }
    ( cd "$HERE/hardware" && ./run-testbenches.sh ) || bad "an RTL testbench"
}

[ $RTL_ONLY = 1 ] && { run_rtl; echo; [ $fails = 0 ] && echo "RTL self-test passed." || echo "$fails RTL failures."; exit $fails; }

step "the fabric clock, before anything else"
if ! "$HERE/doctor.sh" --clock; then
    cat <<'EOF'

selftest.sh will not run the board tests while the fabric clock is wrong.

At 99999999 Hz every answer below would still be correct and every rate would be about 2.5 times
too low, so the run would look like a slow board rather than an unloaded overlay. Fix the clock
first with the command doctor.sh named above, then run ./selftest.sh again.

  ./selftest.sh --rtl-only   still works: the Icarus Verilog testbenches need no board at all.
EOF
    exit 1
fi

step "preconditions (doctor.sh)"
"$HERE/doctor.sh" || bad "doctor.sh -- fix those lines first; everything below will be misleading"

[ $RTL = 1 ] && run_rtl

SITE=$(sudo -u "$NORMAL" python3 -c 'import site; print(site.getusersitepackages())' 2>/dev/null)
[ -n "$SITE" ] || SITE="/home/$NORMAL/.local/lib/python3.10/site-packages"

if [ $QUICK = 0 ]; then
    step "stage check against the numpy reference"
    if [ -f "$DIR/refdump/stages.bin" ]; then
        echo "  \$ sudo $DIR/bitnet_kria --dir $DIR --stage-check $DIR/refdump/stages.bin --head fabric"
        out=$(sudo "$DIR/bitnet_kria" --dir "$DIR" --stage-check "$DIR/refdump/stages.bin" --head fabric 2>&1)
        rc=$?
        echo "$out" | grep -E '^(  setup |stage check|position 7 |  top-1|  two-stage|  logits|  final_norm)' | tail -8
        n_bad=$(echo "$out" | grep -cE 'NOT EXACT|DIFFERS|BELOW THE FLOOR')
        if [ "$rc" != 0 ] || [ "$n_bad" != 0 ]; then
            bad "stage check: rc $rc, $n_bad lines said NOT EXACT / DIFFERS / BELOW THE FLOOR"
        else
            echo "  the reference board's own line for this test, verbatim from results/pv2-stagecheck-fabric.txt:"
            echo "    stage check: every integer stage exact, every float stage above its cosine floor, every top-1 the reference's"
        fi
    else
        echo "  SKIPPED: no $DIR/refdump/stages.bin."
        echo "  Make it on your PC with:  python3 tools/ref_model.py --dump refdump   (then copy refdump/ to $DIR)"
        echo "  tools/prep.sh --local $DIR --dump does this for you; see docs/weights.md."
    fi
fi

step "token ids and speed against the reference board"
sudo env PYTHONPATH="$SITE" python3 "$HERE/selftest/check_ids.py" --dir "$DIR" \
     --out "$HERE/selftest/ids-this-board.json" || bad "the recorded token ids"

echo
if [ $fails = 0 ]; then
    cat <<EOF
=========== PASSED

This board reproduces the reference board's token ids exactly and passes every stage of the
numeric check.  If the rates printed above are well below the reference board's, nothing is
broken -- but read results/kria-speed2-results.txt before quoting a number, and re-run on a board
with nothing logged in: an ssh login alone costs about a core-second of daemon churn, and a
quarter of a core costs about 1 generated token a second.
EOF
else
    cat <<EOF
=========== $fails FAILED

Do not quote any number from this board until these pass.  In order of likelihood:
  a file md5 differs        re-fetch that one file: sudo ./setup.sh --weights
  ids differ from id 0      a different bitstream, or the ARM head where fabric was recorded
  everything is just slow   something is on a core: sudo systemctl restart polkit; log out; retry
EOF
fi
exit $fails
