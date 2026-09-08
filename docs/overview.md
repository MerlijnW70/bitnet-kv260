# Overview

Everything that used to be on the front page, so that the front page could become four steps.

`README.md` is the four commands. This file is what they do, what was measured, what was verified
and what was not, and what is in the tree.

---

## What this is

A 2-billion-parameter language model answering questions on a Kria KV260, with every matrix product
done by gates in the FPGA fabric.

`microsoft/bitnet-b1.58-2B-4T`, all 30 layers. Every ternary matrix-vector product — attention
projections, the FFN, and the output head — is streamed through four engines in the KV260's
programmable logic. Attention, the norms, RoPE, the softmax and the exact top-K rescore run on the
board's four Cortex-A53 cores. Nothing leaves the board.

[docs/how-it-works.md](how-it-works.md) is the engine, the base-3 stream, and what runs where.

## The measured numbers

On the reference board — one board, one image, one kernel, listed in
[ENVIRONMENT.md](../ENVIRONMENT.md):

| | measured |
|---|---|
| generation | **16.78 tok/s** at short context |
| prompt | **28.78 tok/s** |
| a generated token | **59.599 ms**, of which the fabric holds **77%** |
| weight bytes a token | **417,546,240**, streamed at **12.73 GB/s** (79.6% of the four engines' 16.0 GB/s beat rate) |
| power | **6.21 W** generating, **5.15 W** on the prompt, **3.81 W** idle |
| energy | **0.381 J** a generated token (0.147 J over idle) |
| resident memory | **412,652 kB** |
| against `bitnet.cpp` on the same board's own A53s | **5.88×** the generation rate at **5.15×** fewer joules a token; **3.36×** less resident |

Every one of those, with the file it came from and the command that produced it, is in
[docs/measurements.md](measurements.md). The raw records are in [results/](../results/).

**This is one board.** Everything here was measured on a single KV260 with one Ubuntu image and one
kernel. [ENVIRONMENT.md](../ENVIRONMENT.md) lists what was tested and, at more length, what was not.

## Correctness

Correctness is not asserted, it is checked. `bitnet_kria --stage-check` runs all 30 layers at 8
prompt positions and compares every stage against a numpy reference of the same arithmetic: **all 13
of the integer arrays are exact — bit for bit, every position** — every float stage is above its
cosine floor, the residual stream holds a cosine of 0.9998 or better at every layer, and the top-1
is the reference's at all 8. On a 26-prompt quality battery, **11 of 26 generations are
byte-identical** to the same weights running in bf16 on an RTX 4080, and four judging agents called
the board's answer worse than the GPU's on **0 of 26**.

## What you must own

- A **Kria KV260 Vision AI Starter Kit** (`xck26-sfvc784-2LV-c`) and its 12 V supply.
- A **microSD card** with about 20 GB free after the image, and the **Ubuntu 22.04 image for Kria**.
- **4 GB of board RAM** — this is what the KV260 has. The model needs about 1.5 GB of files on disk
  and 520 MB of physically contiguous memory.
- A **network path once**, to fetch the weights (about 1.4 GB over six files).

You do **not** need Vivado, a Vivado licence, or any AMD account. The bitstream is in this
repository (7,797,692 B) and loads with tools that ship on the Kria Ubuntu image.

You no longer need a PC. The packed weights are downloaded ready-made from
[`merlijn70w/bitnet-kv260-weights`](https://huggingface.co/merlijn70w/bitnet-kv260-weights) by
`tools/prep.sh`, which `setup.sh --weights` calls on the board; the packers in `tools/` still run
standalone if you want to repack from the original checkpoint yourself. Every file is checked against
the size and md5 recorded in `manifest.json` and `head_t.json` before anything uses it.
[docs/weights.md](weights.md) is the whole story, including how to serve them from somewhere else.

## The one reboot

Exactly one thing needs it: **`cma=1000M` on the kernel command line.** The three u-dma-buf buffers
total 637,534,208 B, the default Kria CMA reservation cannot hold them, and the kernel command line
is read only at boot. `setup.sh --bootargs` prints your current command line and the boot-config
files it can find, and **does not edit them** — the mechanism differs between Kria images and
getting it wrong makes a board that will not boot.

Everything else is live: the module is a `modprobe`, the bitstream and its overlay are one
`fpgautil` call (measured at 173 ms), and the runtime is a `gcc`.

About **45 minutes** from a board in a box to a first token, of which roughly 20 minutes is a human
with an SD card and about 8 is script time. That total is an estimate and is marked as one; the last
two seconds of it are measured to three figures: setup 1.3 s, prompt 0.56 s, first token 59.6 ms.

## Asking it something

`README.md` stops at the checks. The front end is `chat.sh`:

```sh
./chat.sh "What is the capital of France?"
./chat.sh
./chat.sh --timing -- "What is an FPGA?"
```

It exists because the invocation is awkward and the awkwardness is not optional: `bitnet_kria` opens
`/dev/mem` so it must be root, while `tokenizers` is a `--user` install belonging to the normal user.
A single bare argument is the question; anything after a `--` is the question; every other argument
goes to `runtime/bitnet_chat.py` unchanged (`--dir --exe --max-new --context --threads
--cache-dtype --head --temp --top-p --seed --timing --system --no-stream --quiet`).

**Throw the first run away.** After any file copy the first run reads the model from storage rather
than the page cache: 8.13 s of setup at 51 MB/s against 0.87 s at 482 MB/s, and its first prompt is
worthless (6.52 tok/s, a 967 ms shift scan). The same process's second and third repeats gave
29.11 / 16.81 tok/s.

## How to know it worked

```sh
./doctor.sh          # every precondition, one PASS/FAIL line each, changes nothing
./doctor.sh --clock  # the fabric clock alone, exit 0 or 1
./selftest.sh        # the above, plus the numeric checks below
./selftest.sh --rtl-only   # just the Verilog testbenches — no board, no weights, no bitstream
```

`doctor.sh` reports the **fabric clock** before anything else, because a fabric left at 100 MHz is
the one failure that answers correctly and reports nothing. It reads `pl0_ref` from
`/sys/kernel/debug/clk/clk_summary`, requires `249999998`, and on `99999999` prints a block naming
both fixes. `selftest.sh` runs `doctor.sh --clock` first and refuses to run any board test until it
passes, so a 100 MHz fabric can never be mistaken for a slow board in a self-test result.

`selftest.sh` then checks four things, cheapest failure first:

1. **`doctor.sh`** — CMA size, the three u-dma-buf buffers and that all three sit below 4 GiB,
   `fpga0` operating, the bitstream's md5 against the installed copy on both loader paths, every
   packed file against the md5 its manifest recorded, a quiet board, and that `tokenizers` imports
   as the normal user. It **refuses** rather than warns on the silent failures.
2. **The RTL** (`--rtl`), with Icarus Verilog, needing no board and no weights:
   `tb_ternary_matvec` must print `checked 352 values, 0 wrong, 0 protocol faults` and
   `tb_ternary_glue` must print `checked 58273 elements, 0 wrong, 0 protocol faults`. The engine's
   testbench takes seconds; the glue's took over ten minutes here, because `mul16.v` is 370 kB of
   gate-level multiplier simulated gate by gate.
3. **The stage check** — `bitnet_kria --stage-check` against `refdump/stages.bin` from the numpy
   reference: all 30 layers at 8 prompt positions, with the 13 integer arrays required to be exact
   at every one. On a correct board it ends:
   `stage check: every integer stage exact, every float stage above its cosine floor, every top-1
   the reference's`, and `rc 0`. `refdump/` is opt-in: `tools/prep.sh --local DIR --dump` builds it,
   and without it this step reports SKIPPED rather than failing.
4. **The recorded token ids** — six short prompts whose exact generated ids were recorded on the
   reference board and agree across four independent runs of it. Decoding is greedy, so a correct
   board reproduces them exactly; one different id means the fabric, the weights or the runtime is
   not what it should be. The prompt and generation rates are printed beside the reference board's
   in the same table, so a *slow* board and a *wrong* board are told apart in one screen.

## The forty alarms

`runtime/edge_monitor.py` is the demo the quality numbers come from: forty lines of the kind a PLC
or a sensor gateway puts on its diagnostic bus, each answered with one line of strict JSON, with one
`bitnet_kria` held open for the whole run and the INA260 sampled across every answer.

```sh
cd /home/ubuntu/bitnet-kria
sudo env PYTHONPATH=/home/ubuntu/.local/lib/python3.10/site-packages \
     python3 edge_monitor.py --job triage --no-sudo --json-out edge.json
```

`--job both` adds the board's self-reports from its own live sensors; `--job selftest` runs those
alone. `--saving N` measures what holding one process open is worth, and `--idle N` takes an idle
power baseline first. The scored results are in [docs/quality.md](quality.md) and
[results/edge-scores.json](../results/edge-scores.json).

## What was verified while this repository was assembled, and what was not

The reference board was deliberately **not disturbed** while these files were put together, so it is
worth saying plainly which parts have been executed in this form and which have only been assembled
from commands recorded working.

**Run, here, in this layout, and passing:**

- both Icarus Verilog testbenches, against the RTL and the vectors exactly as they are in
  `hardware/` — `tb_ternary_matvec` gave `checked 352 values, 0 wrong, 0 protocol faults` through
  `run-testbenches.sh matvec`, and `tb_ternary_glue` gave
  `checked 58273 elements, 0 wrong, 0 protocol faults`. The RTL here is the research repository's,
  apart from the vector paths and the comments stripped for publication;
- `hardware/gen_matvec_vectors.py` and `hardware/gen_glue_vectors.py`, which regenerate
  `hardware/vectors/` **byte for byte identically** to what is committed;
- `hardware/grown/verify_pieces.py`, which re-derives each exam's own 65,536 confirm rows and runs
  the sixteen grids `hardware/mul16.v`'s partial products were built from against them:
  `16 of 16 pieces whole on every confirm row`, 1,048,576 cases and 33,554,432 answer bits, in
  under two seconds and with nothing but the Python standard library. Its negative tests were run
  too: one flipped cell is caught, and erasing any one of the 494 kept gates in turn is caught all
  494 times;
- the self-test's golden data: the six prompts' recorded ids agree across four independent recorded
  runs of the reference board, and their prompt ids were re-derived here from the prompt text
  through `runtime/bitnet_chat.py`'s own template and the checkpoint's `tokenizer.json`;
- `tools/checkfiles.py` against synthetic packs — a corrupted file, a wrong size, a missing file
  and a two-bit manifest each produce the intended failure;
- `tools/prep.sh` against a local server — an empty directory, a full one, no address set, a
  corrupted file, a server handing back the wrong bytes, and a doctored manifest;
- every shell script parses and every Python file compiles;
- the bitstream's md5 is `1f7e4fcc5ee4c905ebcfb57f60fe9675`, the value recorded beside every
  measurement in `results/`.

**Assembled but not re-run on the board:** `setup.sh`, `doctor.sh`, `chat.sh` and `selftest.sh`
themselves. The commands inside them are copied from ones recorded working on the reference board —
`dtc`, `fpgautil -R`, `xmutil unloadapp`, `fpgautil -b … -o …`, the `fpga0/state` and `pl0_ref`
reads, the `u-dma-buf` sysfs reads, `modprobe`, the `/etc/modprobe.d` contents, the exact `gcc`
line, and the `sudo env PYTHONPATH=… python3 … --no-sudo` invocation — but the scripts around them
are new. Each says so in its own output.

**Genuinely untested anywhere**, and marked `[UNVERIFIED]` where they run:

- the u-dma-buf clone, build and install in `setup.sh --stage1` (the reference board's module was
  installed before any of this existed);
- the `apt-get` line;
- the kernel-command-line edit — which is why `setup.sh --bootargs` **prints** what to change and
  refuses to edit it;
- fetching any weight file from any address: the reference board's copy arrived over `rsync` from a
  PC that had packed it.

**The Kria app path is no longer untested.** `README.md` step 3 was run on the reference board:
`firmware/install-firmware.sh` installed the three files, `xmutil listapps` listed `kv260-bitnet`,
`xmutil loadapp kv260-bitnet` reported `loaded to slot 0`, `--verify` read `operating` and
`pl0_ref 249999998`, a generation answered correctly, and `bitnet_kria --stage-check` returned every
integer stage exact and every top-1 the reference's. Two things came out of that run and are written
up in [troubleshooting.md](troubleshooting.md): `xmutil unloadapp` cannot remove an overlay that
`fpgautil` created, so `sudo fpgautil -R` first if the fabric was loaded the other way; and this
image has no `dfx-mgrd.service`, yet `loadapp` works. Every speed figure in `results/` was still
measured over the `fpgautil` path; both load the same two files and set the same clock.

**Two changes were made to the code that came from the research repository**, both because the
friction of shipping exposed them: `bitnet_chat.py` and `edge_monitor.py` gained a `--head` option
defaulting to `auto`. Before that neither front end could reach the fabric output head, so the
documented first-token command ran the ARM head at roughly half the advertised generation rate with
nothing to say why. The runtime already accepted `--head fabric`; only the front ends could not pass
it. That change has now been run on the board: the repository's own `bitnet_chat.py`, at its default
`--head auto`, answered at 20.02 generated tokens a second where the board's older copy without the
option gave 10.29 on the same prompt through the same fabric. Those two figures are one short prompt
each, not a benchmark; `results/` holds the measured rates.

## What is known to be fiddly

All three of the ways this goes wrong on a fresh board are **silent**. `doctor.sh` exists for them.

**1. The fabric runs at 100 MHz and nothing says so.**
`fpgautil -b kv260-bitnet.bit.bin` *without* `-o kv260-bitnet.dtbo` programs the fabric and leaves
`pl0_ref` at the boot firmware's 99999999 Hz. Every answer is still correct; it is simply 2.5×
slower, with no error anywhere. Two nearby versions of the same trap: the overlay names the
firmware **by file name**, so pointing `fpgautil` at a differently-named file leaves the *previous*
bitstream in the fabric; and a live overlay must be removed with
`sudo rmdir /sys/kernel/config/device-tree/overlays/full` first. `setup.sh --fabric` does all of
this and refuses to continue on 99999999, and `doctor.sh` checks it before anything else.

**2. CMA too small, or a buffer above 4 GiB.**
With `cma=1000M` the three buffers leave `CmaFree` at 14,852 kB of 1,024,000 — there is no
headroom, which is why `cma=1000M` is not optional. The engines carry a 40-bit address but the
runtime maps `DDR_LOW`, so every buffer must land below 4 GiB. `doctor.sh` prints each buffer's
`phys_addr` and `size` with that assertion.

**3. Something else is using a core.**
`polkitd` spinning at 88.9% of one A53 charged a busy core to two entire passes of measurement in
this project's history. With a quarter of an A53 taken, generation went from 16.33 to 12.03 tok/s;
with a whole core, to 0.82. An ssh login alone costs about a core-second of daemon churn.
`doctor.sh` refuses above 10% and suggests `sudo systemctl restart polkit`.

**The encoding guard is real and you should let it work for you.** Feed a 2-bit manifest to this
base-3 build and it stops with: *"this build streams base 3, five weights a byte and eighty a beat;
the manifest says (none) with 64 a beat and 4 a byte (model.bin). Repack with pack_model.py
--encoding base3."*

More in [docs/troubleshooting.md](troubleshooting.md).

## What the model is, and is not, good at

This matters more than the speed, and it is measured rather than asserted:
[docs/quality.md](quality.md), with the raw answers in
[results/quality-answers.json](../results/quality-answers.json) and
[results/edge-scores.json](../results/edge-scores.json).

**It is good at format.** In the 40-line alarm-triage demo the board produced **strict JSON on the
first try, 40 of 40**, with exactly the right keys and both fields inside their vocabularies, under
greedy decoding. A gateway could parse and route these with no human in the loop.

**It is poor at judgement.** In the same 40 lines it got the subsystem right 19 of 30 clear cases
and the **severity right only 16 of 30** — twelve of thirteen genuinely critical faults were
downgraded to warning or info. It missed instrument faults almost entirely (sensor category: 0 of
the 4 clear ones). Three of fourteen transcribed `action` fields were wrong or dangerous.

**And that is the model, not the fabric.** The same weights in bf16 on an RTX 4080 downgrade the
same twelve severities and score identically on the clear lines, and 26 of the 40 answers are
identical token for token between the board and the GPU. On the separate 26-prompt battery: the
wrong syllogism, the circular CRC-32 explanation, the haiku that is not 5-7-5, the false claim that
a `for` loop takes a condition, the wrong carry-out formula — the RTX 4080 makes every one of them
the same way. What the fabric is responsible for is that 11 of 26 generations are byte-identical to
the GPU's, 98.7% of the board's tokens are the GPU's own top-1 at that position and 100% are inside
its top-5, and the integer stages are exact.

Do not put this in a paging or interlock path. As an offline structured-output front end at 5.5 W
with nothing on the wire, it does the job.

## What is here

```
firmware/     the bitstream, its device-tree overlay, shell.json, and the Kria app installer
hardware/     the Verilog, the two testbenches with their vectors, the Vivado build script,
              and the utilisation and timing reports the resource claims come from
     grown/   the grids mul16.v's sixteen partial-product pieces were built from, the exam files
              that state what each was asked for, the attempt that failed, and a standalone
              verifier that runs all sixteen against 65,536 rows each of their truth tables
runtime/      bitnet_kria.c (the token loop), the chat front end, the edge demo, the power scripts
tools/        the packers, the numpy reference, the quality and benchmark harnesses, prep.sh
selftest/     the recorded token ids and the script that checks them
results/      the measured artefacts every number here comes from, verbatim
docs/         this file, how it works, the measurements, quality, the weights, troubleshooting,
              rebuilding
setup.sh doctor.sh selftest.sh chat.sh
```

- [docs/how-it-works.md](how-it-works.md) — the engine, the base-3 stream, what runs where.
- [docs/measurements.md](measurements.md) — every number above, with the file it came from and the
  command that produced it.
- [docs/quality.md](quality.md) — what the model gets right and wrong, scored.
- [docs/weights.md](weights.md) — where the weights come from, how every file is verified, and how
  to repack.
- [docs/troubleshooting.md](troubleshooting.md) — one section per way this goes wrong.
- [docs/build-bitstream.md](build-bitstream.md) — rebuilding in Vivado. **Optional.**
- [hardware/grown/README.md](../hardware/grown/README.md) — where `mul16.v`'s multipliers came
  from, what was planned and what was searched for, and how to check it yourself.
- [ENVIRONMENT.md](../ENVIRONMENT.md) — the one configuration this was measured on.

## What this is not

- Not a PYNQ overlay and not a Vitis-AI model. It is a plain Linux program plus a device-tree
  overlay.
- Not an AMD App Store app. It installs into the standard Kria app layout, but no `.deb` in AMD's
  `xlnx-firmware-*` namespace is built and the `xlnx-config` snap is not required.
- Not tested on KR260, KD240, Ubuntu 24.04, PetaLinux, or any other kernel.
- Not a general FPGA LLM framework. One model, one board, one bitstream.

## Licence and attribution

The original work here — the Verilog, the testbenches, the C, the Python, the Tcl, the shell and
the documentation — is under the **Apache License, Version 2.0**, Copyright 2026 Merlijn W. See
[LICENSE](../LICENSE). The source files carry no licence header because this project's rule forbids
comments in code, and Apache 2.0 does not require one.

Two things are **not** under that licence, and [NOTICE](../NOTICE) sets out both in full:

- **`firmware/kv260-bitnet.bit.bin`** contains AMD LogiCORE IP in machine-executable form and is
  redistributed under section 3.1(3)C of the AMD/Xilinx End User License Agreement. It is supplied
  **solely for programming an AMD/Xilinx device** — the XCK26 on a Kria KV260 — and for no other
  purpose. AMD, Xilinx, Vivado, Kria, LogiCORE and Zynq are trademarks of Advanced Micro Devices,
  Inc.; this project is not affiliated with or endorsed by AMD.
- **The model weights are not redistributed in this repository.** They are derived from
  [`microsoft/bitnet-b1.58-2B-4T`](https://huggingface.co/microsoft/bitnet-b1.58-2B-4T) (MIT,
  Copyright (c) Microsoft Corporation) and fetched by `tools/prep.sh`. If you host or redistribute
  the packed files, carry
  [THIRD_PARTY_LICENSES/LICENSE.microsoft-bitnet](../THIRD_PARTY_LICENSES/LICENSE.microsoft-bitnet)
  with them: `embed_bf16.bin` and `norms.bin` are byte-for-byte Microsoft tensors and the rest are
  derivatives.

`u-dma-buf` (BSD 2-Clause, Copyright (c) 2015-2026 Ichiro Kawazome) is built by `setup.sh` from
upstream and is not redistributed here. `bitnet.cpp` (MIT, Microsoft) appears only as the CPU
baseline and is built by you.

Microsoft's own note on the model, worth repeating: *"We do not recommend using BitNet b1.58 in
commercial or real-world applications without further testing and development."*

Please cite the model: BitNet b1.58 2B4T Technical Report,
[arXiv:2504.12285](https://arxiv.org/abs/2504.12285).
