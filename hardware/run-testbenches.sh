#!/bin/sh
set -eu
cd "$(dirname "$0")"
which="${1:-both}"
rc=0

check() {
    if grep -q "$3" "$2"; then
        printf '%s: PASSED -- %s\n' "$1" "$(grep -m1 "$3" "$2")"
    else
        printf '*** %s FAILED: %s does not contain "%s"\n' "$1" "$2" "$3"
        tail -5 "$2" || true
        rc=1
    fi
}

if [ "$which" = both ] || [ "$which" = matvec ]; then
    echo "== tb_ternary_matvec (the ternary matvec engine, base 3, batch 1..4, K = 2560 and 6912)"
    iverilog -o matvec.vvp tb_ternary_matvec.v ternary_matvec.v trit_decode_lut.v
    vvp matvec.vvp > matvec.log 2>&1 || true
    cat matvec.log
    check tb_ternary_matvec matvec.log "0 wrong, 0 protocol faults"
fi

if [ "$which" = both ] || [ "$which" = glue ]; then
    echo
    echo "== tb_ternary_glue (the FFN glue around mul16.v -- this one is slow, over ten minutes)"
    iverilog -g2012 -o glue.vvp tb_ternary_glue.v ternary_glue_axi.v ternary_glue.v mul16.v
    vvp -n glue.vvp > glue.log 2>&1 || true
    cat glue.log
    check tb_ternary_glue glue.log "0 wrong, 0 protocol faults"
fi

if [ "$which" = both ] || [ "$which" = attention ]; then
    echo
    echo "== tb_attn_fx_axi (the attention engine against fxmodel.py, whole layers of twenty heads)"
    python3 genlayers.py attn_layers.txt "${ATTN_LAYERS:-12}"
    iverilog -g2012 -o attn.vvp tb_attn_fx_axi.v attn_fx_axi.v attn_fx_v3.v
    vvp -n attn.vvp > attn.log 2>&1 || true
    cat attn.log
    check tb_attn_fx_axi attn.log "top 0 wrong, sum 0 wrong, acc 0 wrong, tlast 0 wrong"
fi

echo
if [ $rc = 0 ]; then echo "RTL testbenches passed."
else echo "An RTL testbench FAILED; its whole output is in the .log file here."; fi
exit $rc
