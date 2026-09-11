# The measured artefacts

These are the raw records every number in this repository comes from, **copied verbatim** from the
research repository the work was done in. Nothing here was edited, reformatted or re-run for
publication, and that is the point: if a number in the README disagrees with a number here, the file
here is right.

One consequence of copying them verbatim: the file paths written *inside* them name the research
repository's layout (`board/…`, `board/research/ternary/bitnet/…`), not this repository's. The
mapping is:

| in these files | here |
|---|---|
| `board/ffn.bit.bin`, `board/ffn.dtso` | `firmware/kv260-bitnet.bit.bin`, `firmware/kv260-bitnet.dtso` |
| `board/ternary_matvec.v`, `board/ternary_glue.v`, `board/mul16.v`, `board/build-ffn.tcl` | `hardware/` |
| `board/bitnet_kria.c`, `board/bitnet_chat.py`, `board/edge_monitor.py`, `board/power-*.sh` | `runtime/` |
| `board/research/ternary/bitnet/*.py` | `tools/` |
| `board/research/ternary/vectors/` | `hardware/vectors/` |
| `board/ternary_matvec.c`, `board/ternary_ffn.c` | not shipped — they are the earlier **two-bit** engine's drivers and do not work with this bitstream |
| `~/bitnet-kria`, `~/matvec`, `~/ffn` | directories on the reference board, not in any repository |

Some files also mention the binaries `bitnet_kria_b3`, `bitnet_kria_before` and `baseline/…`. Those
are build-to-build comparisons made during the work; the shipped source is the one whose md5 is
recorded as *"the fixed runtime"*.

## What each file is

| file | what |
|---|---|
| `kria-speed2-results.txt` | the main speed and power pass on the shipped base-3 bitstream: tokens a second, where a token goes, bandwidth, power, the bitnet.cpp comparison, and the honest open item (the shift-scan stall) |
| `kria-spec-results.txt` | the speculative-decoding study and the barrier fix; also 13,113 id-against-id comparisons proving the shipped runtime emits the same tokens as before the fix, disturbed and undisturbed |
| `phase-remeasure.log` | the engine pipeline split step by step on the shipped bitstream, twice, and one engine-0 burst sampled in 24 slices, from a profiling build of the runtime that is not shipped |
| `probe4.log` | when each of the four engines finishes, in each of the four phases, on the shipped bitstream |
| `probe5.log` | how far each engine has got when the first one finishes, and the time an even split would save |
| `lone-test.log` | the same four-engine timing on a bitstream, not shipped, with the engines on HP0 HP1 HP3 HPC0, and its 2097 ids against the shipped bitstream's |
| `kria-ddr-results.txt` | the DMA burst pass, and the `spread` port arrangement measured 6.4% slower |
| `kria-quality.txt` | the 26-prompt battery, three ways, with all 78 answers and four judges' verdicts |
| `quality-answers.json` | the raw answers, ids, timings and repetition records those verdicts were formed from |
| `edge-demo-results.txt` | the 40-alarm edge demo: strict JSON 40/40, severity 16/30, energy an answer, and what the board could and could not be used for |
| `edge-scores.json` | the per-line scores behind that table |
| `pv2-stagecheck-fabric.txt` | `bitnet_kria --stage-check` output: every integer stage exact at 8 positions through all 30 layers |
| `model-agreement.txt` | the numpy reference against Hugging Face's own tensors, layer by layer |
| `pv2-short.txt`, `b3v2-360.txt` | the bench prompts at short and long context, three and four repeats |
| `pv2-power-lfl.txt`, `pv2-power-long.txt` | INA260 power traces, phase by phase, at 50 ms |
| `pv2-cpp.txt` | the same measurement on bitnet.cpp, segmented by llama.cpp's own milliseconds |
| `b3v2-rss.txt` | `/usr/bin/time -v` over a generation run |
| `b3v2-ids-fabric.json` | the generated token ids for all 31 battery and bench prompts. `selftest/expected_ids.json` is a six-prompt subset of exactly these, after checking they agree with three other recorded runs |
| `kria.txt` | the earliest pass: bitnet.cpp brought up on the board's own CPUs, which is where the baseline's flags come from |
| `kria-model-results.txt` | the first whole-model pass on the fabric, and the source of the weighting the analysis tools use |
| `kria-speed-results.txt` | the speed pass before base 3, including the fabric's own reference-vector runs and the `polkitd` diagnosis |

The last three are here because tools in `tools/` cite them by name (`breakdown.py`,
`head_shortlist.py`, `llamaseg.py`, `quality_bitnetcpp.py`); they describe **earlier** states of the
work, including the two-bit bitstream, so read `kria-speed2-results.txt` for anything shipped.

## Reproducing them

`../selftest.sh` reproduces the two that matter for correctness (the stage check and the token ids).
For the rest:

```sh
python3 tools/headcheck.py out.json --head fabric        # the 26-prompt battery (its prompts ARE here)
python3 tools/provebench.py --exe ./bitnet_kria --head fabric --repeats 3 \
        --prompts your-bench.json --out bench.json        # the five bench prompts are NOT here
python3 tools/headcheck.py --compare a.json b.json       # id by id
python3 tools/headcheck.py --numbers out.json            # its rates
sudo env EXE=./bitnet_kria FLAGS='--head fabric' NEW=256 CTX=512 ./power-bitnet.sh
sudo env NEW=80 THREADS=4 ./power-bitnetcpp.sh && python3 tools/llamaseg.py <log> <run.err>
python3 tools/quality_board.py ; python3 tools/quality_hf.py ; python3 tools/quality_merge.py
python3 runtime/edge_monitor.py --job both --selftests 3 --saving 5 --idle 20
python3 tools/score_edge.py
```

Everything must be run on a board with nothing logged in and nothing polling: an ssh login while a
run is in flight takes a core off four spinning pool threads. Both measurement scripts in the
recorded pass were launched detached and waited for by a single connection that slept on the board
until a marker file appeared.
