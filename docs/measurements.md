# Every number, and where it came from

Nothing here was re-measured for this document. Each row names the file in `results/` it was copied
out of and the command that produced it. All of it is one KV260 — see [../ENVIRONMENT.md](../ENVIRONMENT.md).

Reproduce any of it with `../selftest.sh`, or with the command in the right-hand column.

## Speed

Five bench prompts, three repeats each, 267 prompt and 876 generated tokens, context 512, one
long-lived process; `results/kria-speed2-results.txt`.

| | measured | how |
|---|---|---|
| prompt | **28.78 tok/s** (34.747 ms a token) | `tools/provebench.py --exe ./bitnet_kria --head fabric --repeats 3 --prompts <yours>` † |
| generation | **16.78 tok/s** (59.599 ms a token) | same |
| per repeat | 17.10, 16.55, 16.70 tok/s | same |
| whole 31-prompt battery, context 1024 | 28.71 prompt / 15.44 generation | `tools/headcheck.py out.json --head fabric` |
| the same battery, `--prompt-batch 1` | 19.74 prompt / 16.03 generation | as above with `-- --prompt-batch 1` |

† **The five bench prompts are not in this repository.** They lived in a `prompts.json` on the
reference board and the file was never checked in, so the exact texts are gone; only their token
counts survive (13, 16, 14, 16 and 30 prompt ids, generating 8, 58, 21, 128 and 77). `provebench.py`
takes any `--prompts` file of the same shape and says so if you have none. The 26-prompt battery
**is** here, in `tools/quality_prompts.json`, and `headcheck.py` runs it; its recorded ids are in
`results/b3v2-ids-fabric.json` and six of them are the self-test's.
| 360-position runs | 27.5-28.2 prompt / 14.0 generation | `results/b3v2-360.txt` |
| setup, warm | **1.29-1.56 s**, of which `model3.bin` 417,546,240 B in 0.86-0.90 s (465-483 MB/s) | any run's banner |
| setup, cold page cache | **8.13 s** at 51 MB/s, and up to 55.30 s fully cold | `results/kria-speed2-results.txt` |
| bitstream + overlay load | **173 ms** | `sudo fpgautil -b kv260-bitnet.bit.bin -o kv260-bitnet.dtbo` |
| max resident | **412,652 kB** | `/usr/bin/time -v`, `results/b3v2-rss.txt` |

The battery's generation figure (15.44) sits below the bench figure (16.78) because of an
undiagnosed intermittent stall in the runtime's shift scan: on three of the 31 battery prompts the
scan cost 7.8-20.3 ms a forward against its 2.7 ms norm. Those three carry 5,305 ms of the 4,980 ms
difference between the two batteries. It is not the batch's and not base 3's — the same stall
appears in the 2-bit runs — and it is recorded here rather than smoothed away.

## Where a generated token goes

59.6 ms at the bench prompts' context; `results/kria-speed2-results.txt`.

| | ms | share |
|---|---|---|
| the engines' weight stream | 32.80 | 55.0% |
| the two glue passes | 6.85 | 11.5% |
| the output head | 5.43 | 9.1% (4.12 of it its own ternary stream) |
| attention | 4.78 | 8.0% (14.6 ms at 360 positions) |
| norms, quantisation, RoPE, residuals | 3.50 | 5.9% |
| the shift scan | 2.50 | 4.2% |
| the activation loads | 0.88 | 1.5% |
| the sums out of the buffer | 0.28 | 0.5% |

**The fabric holds 77% of the token; 61.9% of it is weight bytes moving.**

## Bandwidth

Four engines take one 16-byte beat a clock each at 250 MHz: 16.0 GB/s if the memory can feed them.

| | measured |
|---|---|
| the model stream | 417,546,240 B in 32.799 ms = **12.73 GB/s**, 79.6% of the beat rate |
| the head stream | 65,667,072 B in 4.103-4.123 ms = **15.93-16.01 GB/s**, 99.5-100.0% |

The gap in the model stream is not the memory — the head stream proves the DDR feeds the fabric at
its full beat rate in the same token. Nor is it the 120 short runs a token: every step outside the
weight bursts together is 5.4% of the engine time, about 1.8 ms a token
(`results/phase-remeasure.log`). It is inside the bursts. Engines 1 and 2, on HP1 and HP2, finish
9-33% after engines 0 and 3 in every phase, together, while 0 and 3 wait (`results/probe4.log`).
Other port arrangements are slower: 6.4% with the engines on HP0 HP2 HPC0 HPC1
(`results/kria-ddr-results.txt`), and 7.5% more burst time on HP0 HP1 HP3 HPC0
(`results/lone-test.log`).

## Power and energy

`runtime/power-bitnet.sh` with `runtime/ina260-sample.py` sampling the INA260 at
`/sys/class/hwmon/hwmon2` every 50 ms; `results/pv2-power-lfl.txt`, `results/pv2-power-long.txt`.

| phase | power |
|---|---|
| idle, this bitstream loaded | **3.808 W** |
| setup | 4.416 W |
| prompt | 5.148 W |
| generation | **6.206 W** (6.159 W over a 256-token run) |

Energy: **0.381 J a generated token**, 0.147 J over idle. Against the earlier 2-bit bitstream the
rate rose 13.0% and the energy a token fell only 4.4%, because this bitstream idles 0.274 W higher:
the batched engine's 40 UltraRAMs cost about a quarter of a watt at rest whether or not a token is
being made. That is a real cost and it is why the energy improves less than the speed.

## Against bitnet.cpp on the same board

Same model, same question, same chat template, 4 threads, `--temp 0`, `-n 80`, run back to back on
the same KV260; `results/pv2-cpp.txt`, `results/kria-speed2-results.txt`.

| | bitnet.cpp (A53) | bitnet_kria (fabric) | ratio |
|---|---|---|---|
| idle | 3.790 W | 3.808 W | |
| load | 32.924 s | 1.327 s of setup | |
| generation | 2.77 tok/s, 1.963 J a token | **16.29 tok/s, 0.381 J a token** | **5.88× rate, 5.15× fewer joules** |
| prompt | 4.49 tok/s, 1.176 J a token | **25.48 tok/s, 0.202 J a token** | **5.68× rate, 5.82× fewer joules** |
| max resident | 1,386,672 kB | **412,652 kB** | **3.36× less** |

Counting only the increment over idle, generation is 4.04× fewer joules. bitnet.cpp is untouched by
any of this work and reproduces its own numbers across three nights: 4.49 / 2.77 tok/s, against
4.49 / 2.64 and 4.48 / 2.72.

At the level of a whole answer, over the 40-line edge demo (`results/edge-demo-results.txt`): the
board answers in 14,549.9 ms and 79.547 J a mean answer against bitnet.cpp's 77,703.1 ms and
400.552 J — **5.34× faster at 5.04× fewer joules an answer**. The fabric draws *more* watts (5.467
against 5.155 mean) and uses five times fewer joules, because it is finished 5.34× sooner.

## Long contexts and the int8 cache

Attention runs on the A53s and reads the whole key/value cache for every token, so it waits on DDR:
about 0.08 ms a token for every cached position. Generation, 22 new tokens after a repeated prompt,
`results/kria-int8-results.txt`:

| positions | float32 cache (default) | bf16 cache | **int8 cache** |
|---|---|---|---|
| 250 | 13.79 tok/s | 14.68 tok/s | **15.72 tok/s** |
| 1000 | 7.66 tok/s | 8.54 tok/s | **9.31 tok/s** |
| 1800 | 5.18 tok/s | 5.90 tok/s | **6.57 tok/s** |

`--cache-dtype i8` quantises each query head once a token and scores it against the int8 keys with
integer dot products, reading a quarter of the bytes float32 reads. Whole-board energy at a
1000-token prompt, three runs each, page cache dropped before every run:

| | float32 | int8 | |
|---|---|---|---|
| prompt | 12.88 tok/s at 5.340 W, **0.415 J a token** | 15.68 tok/s at 5.117 W, **0.326 J a token** | 21% fewer joules |
| generation | 7.29 tok/s at 5.750 W, **0.789 J a token** | 9.00 tok/s at 5.633 W, **0.626 J a token** | 21% fewer joules |

Of every run at this prompt in `results/kria-int8-results.txt`, 3 of 11 int8 runs generated at 5.60
to 6.54 tok/s and 1 of 9 float32 runs at 5.33. The cause was the measurement, not the runtime: every
dip began in the second a new SSH login reached the board, and on the Ubuntu desktop image a login
starts a user session (PulseAudio, snapd-desktop-integration, portals) that takes about 15,000
context switches and 18% system time for a few seconds. Traced token by token, the same eight runs
with the waiting side polling over fresh logins had tokens of up to 382 ms; with one SSH connection
held open and no new logins, no token took over 165 ms, float32 averaged 136.5 to 141.4 ms a token and
int8 109.5 to 114.0 ms (`results/kria-dip-traces.txt`). Measure with nothing logging in.

int8 is not bit-exact. Over the 31 prompts of `tools/quality_prompts.json` and the bench, 12 answers
are identical and 19 differ; on reading them the differences are wording (144 / 12 + 7 is 19, the
train arrives at 12:05 and 91 is composite in both). float32 stays the default and gives the
recorded ids exactly.

With `--context 2048` the float32 cache locks about 1.6 GB of the board's 3.9 GB and pushes the page
cache out; the same run then varies by up to 40%. The int8 cache is a quarter of that size.

## Attention in the fabric

`--cache-dtype fab` hands attention to two engines in the fabric, beside the matvec engines they share
their DMA with. A layer sends each port a header (the pass, its share of the positions, the twenty
query factors and query rows, and the maxima) and then its range of the key/value blocks, and reads
back the maxima, then the weight sums and accumulators; the ARM only divides. The arithmetic is
`--cache-dtype fx`'s, in integers: Q12 scores from int8 keys and a once-a-token int8 query, softmax
weights from a 4096-entry exp table times a 21-entry one in 20 bits cut off 20 logits under the head's
best score, and values summed as `((w * vscale16) >> 16) * v8`. `results/kria-attention-results.txt`.

| a 1000-token prompt | prompt | generation | energy a generated token |
|---|---|---|---|
| float32 on the A53s | 9.57 tok/s | 5.21 tok/s | |
| int8 on the A53s | 16.4 tok/s | 9.09 tok/s | 5.97 W, **0.657 J** |
| **the fabric** | **25.8 tok/s** | **15.4 tok/s** | 6.84 W, **0.444 J** |

| an 1800-token prompt | prompt | generation | attention a forward |
|---|---|---|---|
| float32 on the A53s | 8.98 tok/s | 5.18 tok/s | 74.7 ms |
| int8 on the A53s | 12.01 tok/s | 6.50 tok/s | 50.7 ms |
| **the fabric** | **22.95 tok/s** | **13.87 tok/s** | **13.8 ms** |

Three runs of each at the 1000-token prompt with the page cache dropped: prompt energy 0.331 -> 0.228 J
a token, generation 0.657 -> 0.444 J. The two engines cost 0.27 W at rest (idle 3.83 -> 4.10 W) and
more while they run; a token still ends up a third cheaper because it is over sooner.

The fabric answers what `fx` answers, every token: over the 31 battery prompts the ids are identical,
2198 of them, and against float32 they differ on the same 19 prompts int8 differs on. The float32 path
is untouched by this bitstream: the stage check passes and the battery gives the recorded 2097 ids.

The engine is `hardware/attn_fx_v3.v` (five key/value groups of four query heads, one set of
multipliers shared across the groups, accumulators in distributed RAM), wrapped for the stream by
`hardware/attn_fx_axi.v` and put beside the matvec engine by `hardware/ternary_port_axi.v`, which
switches the DMA between them on bit 15 of the port's GPIO. `hardware/tb_attn_fx_axi.v` checks it
against `hardware/fxmodel.py` layer by layer, and CI runs that on every push. Two of the four ports
carry one: three do not fit, 89.9% of the LUTs and 0.9 ns short of the clock.

## The FFN glue at sixteen chains

With attention off the ARM, the largest thing left in a forward that was not the weight stream was the
glue: 6.85 ms a token, 114 us a call over thirty layers and two passes, out of 2467 LUTs. It was
compute-bound and not stream-bound -- a 32-clock frame shared by eight chains at offset 4c takes an
element every four clocks, while a pass moves only 110 KB. Sixteen chains at offset 2c take one every
two, which is what the input pipeline and the 128-bit and 32-bit streams already sustained.

`hardware/run-testbenches.sh glue` on both widths: 58273 elements, 0 wrong, 0 protocol faults, every
line of output identical apart from the clock counts. 6912 elements take 27817 clocks at eight chains
and 13995 at sixteen, 4.03 -> 2.02 clocks an element.

| per forward (ms), a 1000-token prompt, `fab` | 8 chains | 16 chains |
|---|---|---|
| glue A | 3.434 | **1.778** |
| glue B | 3.411 | **1.754** |
| the weight engines | 13.761 | 13.763 |
| attention | 9.573 | 9.588 |
| **a forward** | **42.562** | **39.276** |

Nothing but the glue moves. A 1000-token prompt goes 23.73 -> 25.75 tok/s over three runs and a prompt
token 0.2409 -> 0.2275 J; an 1800-token prompt 21.72 -> 22.95 tok/s and its generation 13.03 -> 13.87.
A generated token at the 1000-token prompt is 0.4455 -> 0.4438 J, which is inside the noise of runs
that generate 22 tokens. The battery on float32 still gives the recorded 2097 ids, all identical.

It costs 1367 LUTs of 117120 (the glue 2467 -> 4241, no block RAM and no DSP) and the build closes
wider than before: WNS +0.008 -> +0.049 ns, hold met, `hardware/reports/attn-timing.txt`.

## Several sequences at once

A generated token reads 417.5 MB of weights and another 65.7 MB for the head. At the 11.9 GB/s this
board's DDR gives, that is about 40 of its 58 ms, and base-3 packing is already within 1.4% of what
entropy coding the trits would save, so the bytes cannot come down. They can be shared. `--gen-batch N`
puts N sequences in the engines' slots and the weight stream is read once for the group; each sequence
keeps its own cache, its own attention, its own glue passes and its own head, so only the stream is
shared. A slot is a (position, sequence) pair, which is all a forward ever assumed. The sequences'
prompts go on the input line with a semicolon between them, or one prompt serves all of them, and the
ids come back as `sequence:id`.

| 250-token prompt, context 1024, three runs each | generation | power | energy a token |
|---|---|---|---|
| one sequence | 18.57 tok/s | 6.828 W | 0.3677 J |
| **two sequences** | **23.75 tok/s** | 6.449 W | **0.2716 J** |

Generation is 27.9% higher and a token costs 26.1% less, 2.72 -> 3.68 tokens a joule. The power falls
with the batch because more of a token is the DMA streaming and less of it is the A53s.

Context is the price: a sequence's cache is layers x context x 1312 bytes, 80.6 MB at 2048 positions,
and udmabuf0 has about 107 MB free once the weights are in it. Two sequences therefore want 1024
positions each; `--gen-batch` says so and refuses rather than overrunning the buffer.

`--gen-batch 1` gives the shipped runtime's ids exactly, both sequences of a `--gen-batch 2` run give
what each prompt gives on its own, and the float32 battery still gives the recorded 2097 ids.

One line of ids is one prompt and a semicolon starts the next, so a run holds exactly as many
sequences as there are prompts: a lone caller is never slowed by a copy of itself riding the other
slot. `--gen-batch` is the most a run may hold. A line may open with `!N`, the tokens that run may
generate, which only ever lowers `--max-new`. The table below is taken with `--gen-batch-fill`,
which restores the older reading where a single prompt fills every slot.

How far it pays, 48 tokens a sequence at context 512, two runs each:

| sequences | generation | power | energy a token | tokens a joule |
|---|---|---|---|---|
| 1 | 17.17 tok/s | 6.759 W | 0.3937 J | 2.54 |
| **2** | **22.70 tok/s** | 6.491 W | 0.2860 J | 3.50 |
| 3 | 21.93 tok/s | 5.980 W | 0.2727 J | 3.67 |
| **4** | 23.97 tok/s | 5.950 W | **0.2482 J** | **4.03** |

Three sequences are *slower* than two. That is the engine rather than noise: it answers `SLICES = 2`
activation vectors a clock, so a batch of b costs `ceil(b / SLICES)` clocks a weight beat. Two ride
one clock, three and four both cost two, so past two the weight stream stops being shared and only the
power falls. Two is the speed sweet spot on this part and four the energy one; three is never worth it.

`SLICES = 4` would put four sequences back to one clock a beat, around 31 tok/s, and does not fit
xck26. Out of context an engine is 3805 LUTs with 10 BRAM + 10 URAM at two slices and 7204 LUTs with
10 BRAM + 30 URAM at four: forty blocks an engine, 160 over the four, against 90 BRAM + 40 URAM
already spent of the part's 144 + 64. Pairing two slices onto one memory's two ports does not rescue
it either, because a true-dual-port block RAM is limited to 36-bit ports where the simple dual port
the store uses gives 72, so the 128-bit word costs twice the blocks a copy and the total is unchanged.

## An endpoint on the board

`runtime/bitnet_serve.py` puts an OpenAI-compatible endpoint in front of the runtime, so anything that
speaks that API can point at the board. It holds one bitnet_kria open and groups requests that arrive
close together into one `--gen-batch` run, which is where the throughput comes from: the whole model's
weight stream is read once for the group instead of once a request.

    python3 bitnet_serve.py --gen-batch 2 --context 1024

    curl http://kria:8080/v1/chat/completions -H 'content-type: application/json' \
      -d '{"messages":[{"role":"user","content":"144 / 12 + 7?"}]}'

Measured through the endpoint, 64 tokens a request, the same prompt:

| | end to end | its generation phase |
|---|---|---|
| one request | 15.41 tok/s | 18.22 tok/s |
| two at once | **19.41 tok/s** | **23.84 tok/s** |

The generation figure matches what the runtime gives on its own, so nothing is lost in the grouping;
the end-to-end number is lower than it because each request also pays for its prompt.

Routes: `POST /v1/chat/completions` and `/v1/completions`, both with `stream`, `GET /v1/models` and
`GET /healthz`. The board samples greedily, so temperature and top_p are accepted and ignored.

## Correctness

| | measured | how |
|---|---|---|
| every integer stage | **exact** — the 13 integer arrays the reference dumps (`l0_xq`, `l0_qi`, `l0_ki`, `l0_vi`, `l0_aq`, `l0_oi`, `l0_x2q`, `l0_g`, `l0_u`, `l0_shifts`, `l0_h`, `l0_q8`, `l0_d`) at each of 8 prompt positions, with all 30 layers run; not one line in the file says NOT EXACT | `bitnet_kria --stage-check refdump/stages.bin --head fabric`; `results/pv2-stagecheck-fabric.txt` |
| the residual stream | cosine against the reference at every fifth layer, worst 0.999812 at layer 13 | same |
| final norm | cosine 0.999440 | same |
| logits | cosine 0.999891, max abs difference 0.5261 | same |
| top-1 | the reference's, at all 8 positions | same |
| the fabric two-stage head vs the exact head | same argmax at all 8 positions | same |
| teacher-forced agreement with an RTX 4080 | **98.7%** of board tokens are the GPU's own top-1, **100%** inside its top-5 | `results/kria-quality.txt` |
| generations byte-identical to the GPU | **11 of 26** | `results/kria-quality.txt` |
| determinism | the same prompt run twice gave byte-identical ids | `q24_determinism` |
| the RTL, engine | `checked 352 values, 0 wrong, 0 protocol faults` | `hardware/run-testbenches.sh matvec` |
| the RTL, glue | `checked 58273 elements, 0 wrong, 0 protocol faults` | `hardware/run-testbenches.sh glue` |
| recorded token ids | six prompts, identical across four independent runs of the reference board | `selftest/check_ids.py` |

## The fabric

`hardware/reports/`, from the Vivado run that produced `firmware/kv260-bitnet.bit.bin`.

| | |
|---|---|
| CLB LUTs | 42,271 of 117,120 (36.09%) |
| registers | 53,342 of 234,240 (22.77%) |
| Block RAM tiles | 95 of 144 (65.97%) |
| URAM | 40 of 64 (62.50%) |
| DSPs | **0** of 1248 |
| timing, post-route slow corner | WNS 0.108 ns, TNS 0.000, WHS 0.010, THS 0.000 |
| failing endpoints | 0 of 178,007 |
| clock | 250 MHz |
| one engine | 4,179 LUT / 1,829 FF / 10 RAMB36 / 10 URAM |

Against the earlier 2-bit engine (1,353 LUT / 787 FF / 8 RAMB36 / 0 URAM): 3.09× the LUTs and 2.32×
the flip-flops, for a fifth fewer bytes on the wire and four vectors answered a pass.

On a Lattice ECP5, packed by yosys and nextpnr-ecp5 with the engine's UltraRAM mapped to block RAM
(`hardware/ecp5-fit.sh`, `hardware/reports/ecp5-85k.txt`, `ecp5-45k.txt`): one engine is 8,111 LUT,
2,236 FF and 40 DP16KD, the glue 4,663 LUT, 6,116 FF and 1 DP16KD. Four engines and the glue need
161 block RAMs, which fits an LFE5UM-85F (208) and not a 45F (108), at 37,107 of 83,640 LUTs on the
85F. That is packing only: nothing was routed, no timing was closed and nothing ran on an ECP5.

## What is left on the table

Each of these is a projection, marked as one, and none is measured.

1. **An even split across the four engines** — engines 1 and 2 hold the other two up in every
   phase. Shares matched to each engine's measured rate would save about 84 µs a layer
   (`results/probe5.log`): 2.5 ms a token, 57.1 ms, 17.5 tok/s, at the price of repacking the 417 MB
   weight file into uneven shares. Fewer, longer engine runs are worth at most the 1.8 ms the runs
   cost: 57.8 ms, 17.3 tok/s.
2. **Batched generation across independent conversations.** The engines already answer one pass of
   the weight stream for four vectors and the prompt already uses it; four separate conversations
   have no token-to-token dependency and `ctrl[10:9]` already carries the batch. Nothing in the
   fabric would change. Taken in full: about 43.2 ms a token for each of four, 23.1 tok/s each.
3. **The glue**, 6.85 ms a token, the one part of the pipeline not revisited since it was built.

They do not add — lever 2 halves the very stream lever 1 shortens — and the floor under both, with
every ARM millisecond hidden and every byte moving at the beat rate, is the **30.2 ms the fabric
needs: 33 tokens a second**.

**The packing is finished.** These weights carry about 1.55 bits of entropy each and base 3 spends
1.585 (8 bits over 5 weights). Under 2% is left between this packing and the entropy floor, and
taking it would need a variable-length code the fabric would have to decode serially.
