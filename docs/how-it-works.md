# How it works

## The split

BitNet b1.58 has ternary weights: every weight is -1, 0 or +1. A matrix-vector product with int8
activations is then pure integer addition and subtraction — no multiplier at all — which is exactly
what an FPGA's LUT fabric is good at and exactly what a CPU is not.

So the split is: **every ternary matrix product runs in the fabric; everything else runs on the four
A53s.**

| in the fabric (250 MHz) | on the A53s |
|---|---|
| q, k, v projections | RMS norms, and the int8 activation quantisation |
| o projection | RoPE |
| gate and up projections | grouped-query attention and the softmax |
| down projection | the residual adds and the output scales |
| the FFN glue's four multiplies | the search for the FFN's five shift amounts |
| the output head's stage-1 ternary scoring | the exact top-K rescore of the shortlist |

That is 77% of a generated token in the fabric, and 61.9% of the token is weight bytes moving.

## Five weights a byte, in base 3

A trit carries log2(3) = 1.585 bits. Two bits a weight wastes a fifth of the bandwidth. So the
packing is base 3:

```
byte = sum over j of (w_j + 1) * 3^j,  j = 0..4,  lowest power first
```

243 of the 256 codes are used; a byte of five zero trits is 121, not 0. A 16-byte AXI beat carries
**80 weights**. A neuron of K inputs is `ceil(K / 80)` beats — 32 for K = 2560, 87 for K = 6912 —
with the last beat's spare weights padded to zero.

The whole model is **417,546,240 bytes**, 80.14% of the two-bit packing, of which 737,280 bytes is
`down_proj`'s padding. That is the end of the road for packing: the weights carry about 1.55 bits of
entropy each and base 3 spends 1.585, so under 2% is left, and taking it would need a
variable-length code the fabric would have to decode serially.

The manifest records the encoding by name. Feed a two-bit file to this bitstream and the runtime
refuses, naming the fix, rather than streaming nonsense.

## The engine

`hardware/ternary_matvec.v`. One 128-bit weight beat a clock, four engines, so 16.0 GB/s if the
memory can keep up. Each engine sits between an AXI DMA's MM2S and S2MM streams with a dual AXI GPIO
for its control and status words.

The thing worth knowing about it: **an engine holds up to four activation vectors and reads the
weight stream once for all of them.** A run of batch B answers B positions for one pass of the
weights. Prompt evaluation uses that — four positions a group, two passes of the stream instead of
four — which is why the prompt runs at 28.78 tok/s against generation's 16.78. Generation cannot,
within one conversation, because token *t+1* needs token *t*; four *separate* conversations have no
such dependency, and `ctrl[10:9]` already carries the batch, so that speed-up needs no change to the
fabric at all. It has not been built.

A layer is 13,918,208 bytes in four engine phases, not seven, because adjacent matrices that share
their input width are streamed as one run:

| phase | matrices | neurons | beats a neuron |
|---|---|---|---|
| 1 | q, k, v | 3840 (960 an engine) | 32 |
| 2 | o | 2560 (640 an engine) | 32 |
| 3 | gate, up | 13824 (3456 an engine) | 32 |
| 4 | down | 2560 (640 an engine) | 87 |

## The FFN glue, and where the multipliers came from

The FFN's integer path needs four real multiplies per element — squaring the gated value, then two
scalings, then the output quantisation — with five shift amounts the A53 searches for each token.
`hardware/ternary_glue.v` does them in the fabric, one element a beat in pass A and four a beat in
pass B, keeping the running maximum the output scale needs.

The 16×16 multipliers those four multiplies use, `hardware/mul16.v`, were **not written by hand**:
370 kB of gate-level Verilog — sixteen partial-product pieces summed by a four-level tree of fifteen
serial adders — with no DSP and no vendor primitive. In its 12,284 lines there is no `*` at all, and
the only seventeen `+` are the step counters that say which clock each module is on; every bit of
the arithmetic itself is gates. They came out of an evolutionary circuit compiler, and what that
means here is narrower than the phrase usually suggests. The grids are published in
[hardware/grown/](../hardware/grown/) so it can be read rather than believed. Four things happened,
in this order:

1. **A deterministic planner laid every piece out.** `hardware/grown/ppplan.py` places each gate of
   a partial-product piece by construction — a one-hot step chain, a delay chain for `x`, a latch
   for the one bit `y_i`, one `maj` on the lane — moving operands to the rows a gate can read by
   shortest paths. Its output is `pp00.grid`..`pp15.grid`. That is a plan. Nothing is searched.
2. **That plan was the search's starting population.** Every kept piece's exam file carries
   `start expeditions/mul/pp<NN>.grid 4096`: 4,096 copies of the planned grid were the colony the
   compiler began from.
3. **A candidate was kept only if it was exactly right, and was then pruned.** The exams say
   `confirm 65536`: 65,536 rows drawn from across the piece's 2^32-row truth table, and every one of
   the 32 answer bits must be right on every one of those rows or the candidate is not kept.
   `lean wires` then prunes any gate that does not carry the answer. Three of the sixteen pieces
   came back leaner than the plan — 499 planned gates against 494 kept; the other thirteen came back
   as they went in.
4. **Unseeded, it did not converge.** `hardware/grown/tiny.txt` asks for a 2-bit serial multiplier —
   16 cases, 64 answer bits, the whole table — with no `start` line. On that seed the search reached
   62 of the 64 bits and never closed the gap. The nine-gate serial adder the tree is built from was
   not grown either: it was laid out by hand and seeded the same way, because from nothing the
   search reached about half of its table.

   **The evidence for this step is weaker than for the other three, and that should be said.** The
   64-bit and half-a-table figures are recorded in the header of `tiny.txt` by whoever ran it. There
   is no run log here to check them against, and the serial adder's own exam and grid are not
   published at all. Everything above this point can be re-verified from the files in
   `hardware/grown/`; this cannot.

So the honest summary is: **the layout was planned by construction, an evolutionary search checked
it on 65,536 rows of each piece's table and pruned it, and unseeded search did not solve even a
2-bit multiplier.** The checking and the pruning are real, and unusual. The layout is not the
search's. And 65,536 rows is not the whole 2^32-row table: it is 1 row in 65,536, drawn from across
it, which is a strong check and not a proof.

The check is standalone and takes two seconds:

```sh
python3 hardware/grown/verify_pieces.py
```

It re-derives each exam's own 65,536 confirm rows — drawn by splitmix64 from across the whole table,
not a low corner of it — and runs all of them at once, needing nothing but the Python standard
library: `16 of 16 pieces whole on every confirm row`, 1,048,576 cases and 33,554,432 answer bits.
It fails when a grid is wrong. Erase one of the 494 kept gates, one at a time, and every one of the
494 is caught; that is what `lean wires` bought, and it means there is no gate in these grids to
spare. [hardware/grown/README.md](../hardware/grown/README.md) has the grid format, an annotated
exam, both negative tests and a per-piece table.

`tb_ternary_glue.v` then checks 58,273 elements against the numpy reference through the composed
multipliers, gate by gate, which is why that testbench takes over ten minutes rather than seconds.

The DSP count is a curiosity, not a claim: the design uses 0 of the KV260's 1248 DSPs, but that is
because the multiplies are small and serial, not because these multipliers beat a DSP.

## The output head

The tied head is 128,256 × 2560 — bigger than any layer. Two stages:

1. **In the fabric**: the head quantised to trits by an absolute-mean rule, streamed through the same
   four engines (65,667,072 bytes, one long sequential read, measured at 15.93-16.01 GB/s — the
   engines' full beat rate), giving an approximate score for every one of the 128,256 rows. Take the
   top 256.
2. **On the A53s**: rescore those 256 exactly against the int8 head and take the argmax.

Cost: **5.40 ms a call** against the ARM head's ~52 ms. That is the difference between about 16.8
and about 8 generated tokens a second, so `chat.sh` and `bitnet_chat.py --head auto` choose the
fabric head whenever it is packed.

The shortlist is exact **by measurement, not by proof**. A Cauchy-Schwarz bound was tried and is
about sqrt(2560) too loose to certify anything. What is measured: over 2115 real head states the
worst rank of the true argmax was 56 out of a K of 256, and 49-60 on states perturbed by the board's
own measured drift — the shortlist has never been within 4.5× of failing — and stage 2 re-checks the
top 256 exactly on every call. One case where K = 256 is *not* enough is scoring every position
rather than only generated ones (teacher forcing, perplexity): there the same rule needs K in the
hundreds to thousands. Anyone adding such a mode must raise `--head-k` or use `--head arm`.

## Hiding the A53 behind the fabric

The four pool threads take a slice each of the q/k/v scaling, RoPE and cache write, of the glue's
pass-A beats, and of the shift scan; the beats and the scan's first pass are done a chunk of the
gate+up stream at a time, and the head's shortlist a chunk of the head stream at a time, each chunk
being one engine run whose DMA completion says its sums are in DDR. The output scale, whose sum of
doubles has a chain order that must not be cut, runs on a fifth thread while the fabric runs the
glue and the `down_proj` stream.

Every split is over an operation that is elementwise or a maximum, and `--threads 1` runs them all
inline instead — and produces **identical generated tokens**. That is the check that says the
overlap changed no arithmetic.

## Memory

`u-dma-buf` gives physically contiguous, user-mappable memory that the DMAs can address directly.
Three buffers, 637,534,208 B in all, which is why `cma=1000M` is required:

| | size | holds |
|---|---|---|
| `udmabuf0` | 545,259,520 | `model3.bin`, read once at start-up and streamed from there for ever after |
| `udmabuf1` | 8,388,608 | activations, sums, the glue's beats, the head's sums |
| `udmabuf2` | 83,886,080 | `head3_t.bin`, the ternary head |

All three must land below 4 GiB, because the engines carry a 40-bit address but the runtime maps
`DDR_LOW`. `doctor.sh` asserts it.

## Where it could go faster

See [measurements.md](measurements.md). The short version: the model stream runs at 12.73 GB/s
against the head stream's 16.0, not because the memory cannot keep up — it demonstrably can, in the
same token — but because the model is streamed in 120 short runs a token, each paying its activation
reload, register writes, DMA descriptor setup, completion poll and drain. That is 6.70 ms a token.
The floor under everything, with every ARM millisecond hidden and every byte at the beat rate, is
30.2 ms a token: **33 tokens a second**.
