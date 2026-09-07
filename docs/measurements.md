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
its full beat rate in the same token. It is that the model is streamed in 120 short runs a token
(four engine phases over thirty layers), each with its activation reload, register writes, DMA
descriptor setup, completion poll and drain.

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

## What is left on the table

Each of these is a projection, marked as one, and none is measured.

1. **Fewer, longer engine runs** — the largest measured gap, 6.70 ms a token. Taken in full: 52.9 ms
   a token, 18.9 tok/s.
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
