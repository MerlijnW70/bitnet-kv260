#!/bin/sh
set -eu
cd "$(dirname "$0")"
device="${1:-85k}"
package="${2:-CABGA756}"
out=reports/ecp5-$device.txt

: > "$out"
{
    echo "== yosys $(yosys -V) / $(nextpnr-ecp5 --version 2>&1 | head -1)"
    echo "== device $device package $package"
    echo "== the engine is built with ULTRA 0, which maps its buffers to block RAM instead of UltraRAM"
    echo
} >> "$out"

yosys -q -p "read_verilog -defer ternary_matvec.v; chparam -set ULTRA 0 ternary_matvec; synth_ecp5 -top ternary_matvec -json ecp5_eng.json"
yosys -q -p "read_verilog ternary_glue.v mul16.v; synth_ecp5 -top ternary_glue -json ecp5_glue.json"

for part in eng glue; do
    echo "== $part" >> "$out"
    nextpnr-ecp5 "--$device" --json "ecp5_$part.json" --package "$package" 2>&1 |
        grep -E "TRELLIS_COMB:|TRELLIS_FF:|DP16KD:|MULT18X18D:|ALU54B:" >> "$out"
    echo >> "$out"
done
rm -f ecp5_eng.json ecp5_glue.json

cat "$out"
echo "Packing only: no routing and no timing closure. Four engines and one glue is four"
echo "times the engine row plus the glue row; the KV260 build uses that many."
