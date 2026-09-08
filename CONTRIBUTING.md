# Contributing

Most of this can be worked on without a KV260. Read the first section before anything else; it is
the rule the whole project is built on.

## The rule: the tokens must not change

Every optimisation in `results/` was accepted only after proving that the model's output did not
move. Three passes took generation from 8.31 to 16.78 tok/s, **2.0 times faster**, and not one
generated token id changed. Two changes that were faster were **rejected** for breaking it: an int8
KV cache and a bf16 one, each of which altered the output on more than a dozen of the 31 battery
prompts. Both numbers come from `results/kria-speed-results.txt` and
`results/kria-speed2-results.txt`, which record each pass measured back to back on the board; the
per-pass figures are 1.74x and 1.17x.

So: **if your change touches the arithmetic, greedy output must stay byte-identical**, and the
repository ships the means to prove it.

```sh
./selftest.sh
```

on a board runs the stage check, which compares all 30 layers at 8 positions against a numpy
reference and requires every integer stage to be exact, and then compares generated token ids with
the ones recorded in `selftest/expected_ids.json`.

A change that deliberately changes the output is not forbidden, it is a different thing: say so in
the pull request, say what moved and by how much, and do not describe it as an optimisation.

## Working without a board

You do not need hardware to change the engine, the glue, the packers or the reference.

```sh
sudo apt-get install iverilog python3-numpy
./selftest.sh --rtl-only
```

That elaborates the RTL and runs both testbenches against recorded vectors: the matvec engine at
batch 1 to 4 with K of 2560 and 6912, and the FFN glue around the `mul16.v` multipliers. The matvec
bench takes seconds. The glue bench takes over ten minutes, because it simulates 16x16 bit-serial
multipliers gate by gate.

Regenerating the test vectors needs the checkpoint weights; the recorded ones in `hardware/vectors/`
do not, which is why they are committed.

The cheapest check here needs no board, no iverilog and not even numpy:

```sh
python3 hardware/grown/verify_pieces.py
```

Two seconds, the standard library only. It runs the sixteen grids `mul16.v`'s partial products were
built from against 65,536 rows each of their truth tables. `hardware/grown/README.md` says where
those grids came from — planned by construction, then seeded into an evolutionary search that kept
them only when exact and pruned what carried nothing — and what happened when the search was given
no seed at all.

## Working with a board

```sh
./doctor.sh
```

first, always. It reports the fabric clock before anything else, and half the confusing behaviour
anyone will ever report to you is a fabric silently running at 100 MHz instead of 250. Then
`./selftest.sh` for the full check.

If you change speed, quote the measurement, not the impression. `tools/provebench.py` and
`tools/headcheck.py` are what produced every number in `results/`, and both take a `--compare` of two
recorded runs. State which board, which image and which kernel: see `ENVIRONMENT.md`.

## House rules

**No comments in code.** Not `//`, not `///`, not `/* */`, not `#` in shell or Python, none in Tcl or
the device-tree source. What the code does belongs in the code; what it is for belongs in `docs/`;
what it measured belongs in `results/`. Shebangs, preprocessor directives, Python docstrings and
Verilog `(* ... *)` attributes are not comments and are fine.

```sh
python3 tools/no_comments.py
```

enforces it, and CI runs it on every push.

**No prose in the repository that is not documentation.** Results files hold measured numbers, taken
from a real run, with the command that produced them. Mark a projection as a projection.

**Say what is untested.** Several scripts print `[UNVERIFIED]` where they run a step no one has run
on a real board. If you verify one, delete the marker and say in the pull request what you ran.

## Where to say it

There is no contact email, on purpose: an address in a public repository is scraped within days,
and everything one would be used for has a better home here.

- Something does not work: an [issue](https://github.com/MerlijnW70/bitnet-kv260/issues). The
  template asks for the `doctor.sh` output because most reports are settled by it.
- A question, an idea, or something you built with it: a
  [discussion](https://github.com/MerlijnW70/bitnet-kv260/discussions).
- Something sensitive: **Security → Report a vulnerability**, which is private until there is a fix.
  [SECURITY.md](SECURITY.md) says what is in scope; note that the runtime maps physical memory and
  runs as root.

## Pull requests

CI runs the RTL testbenches, compiles every Python and shell file, parses every JSON, and enforces
the comment rule. None of it needs hardware.

Tell us in the pull request:

- what you changed and why;
- whether the generated tokens are unchanged, and how you know;
- what you ran, verbatim, and on what;
- what you did not test.

Contributions are under Apache 2.0, per section 5 of the licence. There is no separate agreement to
sign.

## What is worth doing

Named in `results/` as measured dead ends or open ends:

- **The glue is not batched like the engines.** Batching it is worth about 6.85 ms a position and
  would drop the break-even for speculative decoding from 1.705 accepted tokens to 1.245, which is
  the difference between it paying and not. `results/kria-spec-results.txt` has the arithmetic.
- **Attention is the only stage that grows with context**, from 4.7 ms at short context to 12.7 at
  309 positions, and it cannot hide under a weight stream for a reason explained in
  `results/kria-speed2-results.txt`.
- **Packing is finished.** Base 3 is 1.585 bits a weight against these weights' 1.55 bits of
  entropy, so there is no prize left there. Do not spend time on it.
- **The A53 side reads at 6.5 GB/s while the fabric reads at 13.3.** Anything that moves work off
  the cores and into the fabric is likely to pay; the output head is the worked example.
