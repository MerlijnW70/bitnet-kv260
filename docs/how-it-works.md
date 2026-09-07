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

## The FFN glue, and the multipliers that were grown

The FFN's integer path needs four real multiplies per element — squaring the gated value, then two
scalings, then the output quantisation — with five shift amounts the A53 searches for each token.
`hardware/ternary_glue.v` does them in the fabric, one element a beat in pass A and four a beat in
pass B, keeping the running maximum the output scale needs.

The 16×16 multipliers those four multiplies use, `hardware/mul16.v`, were **not written by hand**.
They were grown by an evolutionary circuit compiler — sixteen partial-product pieces and fifteen
serial adders that it kept, composed into a bit-serial multiplier. 375 kB of gate-level Verilog with
no DSP, no vendor primitive, and no hand-written arithmetic. `tb_ternary_glue.v` checks 58,273
elements against the numpy reference through them.

That is a curiosity, not a claim: the design uses 0 of the KV260's 1248 DSPs, but that is because
the multiplies are small and serial, not because the grown multiplier beats a DSP.

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
