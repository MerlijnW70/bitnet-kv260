# The self-test's recorded answers

`expected_ids.json` holds six short prompts from the 26-prompt battery
(`tools/quality_prompts.json`) together with the **exact token ids** the reference KV260 produced
for them, running the shipped base-3 bitstream and runtime with `--head fabric`, greedy, at
`--context 1024 --threads 4 --cache-dtype f32`.

Two things make those ids worth testing against:

- **They agree across four independent recorded runs** of the reference board
  (`b3v2-ids-fabric.json`, `pv2-ids-fabric.json`, `ids-b3-fabric.json`, `spec-ids-after.json` — all
  31 prompts, all identical). They are not one lucky sample.
- **The prompt ids reproduce from the prompt text**, checked with the checkpoint's own
  `tokenizer.json` (snapshot `04c3b9ad9361b824064a1f25ea60a8be9599b127`) through
  `runtime/bitnet_chat.py`'s own `chat_text`. So a mismatch in the *prompt* ids means a different
  tokenizer, and `check_ids.py` reports that case separately from a mismatch in the generated ids.

Decoding is greedy. A correct board reproduces every id.

## What the answers are

| prompt | what the reference board answers |
|---|---|
| `q01_france` | `The capital of France is Paris.` |
| `q06_mul` | `17 times 23 equals 391.` |
| `q09_seq` | `The next number in the sequence is 32. This is because each number is obtained by multiplying the previous number by 2, which is a characteristic of a geometric sequence.` |
| `q15_yesno` | `No.` |
| `q17_haiku` | `Silicon dreams take flight, / Wires weave through circuits bright, / Tech's heartbeat, pure light.` |
| `q24_determinism` | `A shift register is used for storing binary data, performing arithmetic operations, and generating serial numbers.` |

162 prompt ids and 97 generated ids in all — about fifteen seconds of board time.

`q24_determinism` is the battery's repeated prompt: run twice, the reference board gave
byte-identical ids both times.

## Running it

```sh
./selftest.sh                    # this, plus doctor.sh and the stage check
python3 selftest/check_ids.py    # this alone, as root
```

`check_ids.py --out ids.json` writes what your board produced, so two boards can be compared with
`tools/headcheck.py --compare a.json b.json`.

## Regenerating it

The whole 31-prompt battery on your own board:

```sh
python3 tools/headcheck.py mine.json --head fabric
python3 tools/headcheck.py --compare results/b3v2-ids-fabric.json mine.json
python3 tools/headcheck.py --numbers mine.json
```
