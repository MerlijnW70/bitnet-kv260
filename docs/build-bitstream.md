# Rebuilding the bitstream — optional

**You do not need to do this.** `firmware/kv260-bitnet.bit.bin` is in this repository (7,797,692 B,
md5 `1f7e4fcc5ee4c905ebcfb57f60fe9675`), it is the file every number in `results/` was measured on,
and the Kria Ubuntu image already carries everything needed to load it. Rebuilding costs a Vivado
install — tens of gigabytes and a licence file — and about sixteen minutes of compute, and gets you
the same design.

Do it if you want to change the design, verify it from source, or target a different part.

## What was used

| | |
|---|---|
| tool | AMD Vivado Design Suite 2026.1 |
| licence | Enterprise (this is what the build log records — see below) |
| part | `xck26-sfvc784-2LV-c` |
| board | `xilinx.com:kv260_som:part0:2.0` |
| host | Ryzen 9 9950X, 16 cores, 66 GB |
| wall clock | **15 min 57 s** |

## The build

```sh
cd hardware
vivado -mode batch -source build-ffn.tcl -notrace
cp ffn.bit.bin ../firmware/kv260-bitnet.bit.bin      # the shipped copy is only renamed
```

`build-ffn.tcl` reads the five Verilog files beside it and builds the whole block design from
scratch: four `ternary_matvec_axi` engines with their AXI DMAs and dual AXI GPIOs on `HP0..HP3`, the
`ternary_glue_axi` block with its own DMA on `HPC0`, five SmartConnects, fully-registered
AXI-Stream register slices in both directions between every DMA and its block, and the AXI-Lite
ports and GPIOs on `M_AXI_HPM0_FPD`. Everything runs on `pl_clk0` at 250 MHz.

It takes five optional name-value arguments, each defaulting to what the shipped bitstream has:

```sh
vivado -mode batch -source build-ffn.tcl -notrace -tclargs name try256 burst 256 ports spread
```

`name` (output base name and `work_<name>` directory), `burst` and `sburst` (each DMA's maximum
burst in beats on the MM2S and S2MM sides, 2..256), `swidth` (the S2MM memory-side data width, 32 or
128), and `ports` (`hp` = engine *i* on `HP<i>` with the glue on `HPC0`, the shipped arrangement;
`spread` = engines on `HP0 HP2 HPC0 HPC1` with the glue on `HP1`).

The default `name` is left as `ffn` rather than `kv260-bitnet` because it becomes a Vivado project
name, and because that is the exact configuration the shipped file was built with. The shipped copy
is a rename, nothing more — the overlay's `firmware-name`, the installed file and the app directory
all have to agree, and `kv260-bitnet` is the name the Kria convention wants.

**The register map is pinned** and the runtime depends on it: GPIO *i* at `0xA000_0000 + i*0x1_0000`,
DMA *i* at `0xA004_0000 + i*0x1_0000`, glue GPIO A at `0xA008_0000`, glue GPIO B at `0xA009_0000`,
glue DMA at `0xA00A_0000`. The script prints the map it actually built as lines beginning
`=== address:`; check them against those values.

## Before you build: check the RTL without Vivado

```sh
cd hardware
./run-testbenches.sh
```

Icarus Verilog, no licence, no board. The engine testbench takes seconds; the glue's took over ten
minutes here, because `mul16.v` is 370 kB of gate-level multiplier and the glue runs 6912 elements
twice. They must print:

```
checked 352 values, 0 wrong, 0 protocol faults
checked 58273 elements, 0 wrong, 0 protocol faults
```

The vectors in `hardware/vectors/` are in the repository; regenerate them with
`gen_matvec_vectors.py` and `gen_glue_vectors.py`, which need numpy and, for the two sets drawn from
the real model, the checkpoint in your Hugging Face cache.

## What to expect from the result

The shipped build closes at **WNS 0.108 ns** — 0.4% of a 4 ns period — with 0 failing endpoints of
178,007, TNS 0.000, WHS 0.010, THS 0.000, "All user specified timing constraints are met", and 0
critical warnings in any run. 42,271 CLB LUTs (36.09%), 53,342 registers (22.77%), 95 of 144 BRAM
tiles, 40 of 64 URAM, 0 DSP. The reports are in `hardware/reports/`.

**That is not a lot of slack.** On a different speed grade or SOM revision the timing would have to
be re-closed, and nothing here says it would close. That is expected, not a bug in the design.

## On Vivado editions

`build-ffn.log` line 1 on the machine that produced the shipped file reads *"A valid Vivado Design
Suite ENTERPRISE license has been detected."* Vivado checked out only the `Synthesis` and
`Implementation` features for device `xck26` and never a per-IP feature, which is what a fee-based
or evaluation core would have logged.

All six IP in the block design are no-charge LogiCORE IP — Zynq UltraScale+ MPSoC PS (PG201), AXI
DMA (PG021), AXI GPIO (PG144), AXI SmartConnect (PG247), AXI4-Stream Register Slice (PG085),
Processor System Reset (PG164) — and UG973 lists Kria as supported at every Vivado tier. So it
*should* build on a lower edition. **It was not tried, so it is not claimed.**

Note that the free tier changed in 2026.1: Basic is free but is still an annual subscription that
must find a licence file before Vivado will launch. That is the difference between an hour and an
afternoon, and it is exactly why the bitstream ships in this repository.

## What is deliberately not in this repository

No `.xci`, no `.dcp`, no `ip/` tree, no synthesised IP netlists, and no `build-ffn.log`. The first
four are AMD Distributable Components in modifiable or restricted form, for which redistribution
needs the recipient to hold their own Vivado licence; the log names a licensee's entitlement and
local paths. `build-ffn.tcl` regenerates all of them, so nothing is lost. See [../NOTICE](../NOTICE).

`ffn.bit.prm` is also not shipped: it is only needed for QSPI flash programming, which this setup
does not use, and it embeds the build machine's paths.
