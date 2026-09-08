# The grids behind `hardware/mul16.v`

`hardware/mul16.v` holds sixteen modules `pp00`..`pp15` and the 16x16 bit-serial multiplier built
from them. Those sixteen modules were not written by hand. This directory holds what they came
from, so that claim can be checked rather than believed: the grids, the exam files that state what
each grid was asked for, a standalone verifier, the planner that laid the grids out, the script
that composed the Verilog, and the record of the attempt that failed.

## What actually happened

Four things, in order, and the artefacts here show each of them.

1. **The layout was planned, not found.** `ppplan.py` places every gate of a partial-product piece
   by construction: a one-hot step chain, a delay chain for `x`, a latch for the single bit `y_i`,
   and one `maj` on the lane; operands are moved to the rows a gate can read by shortest paths over
   the reader graph. Its output is `pp00.grid`..`pp15.grid`. That is a plan. No search.

2. **The plan was then handed to an evolutionary circuit compiler as its starting population.**
   Every `pp<NN>fed.txt` carries the line

   ```
   start expeditions/mul/pp00.grid 4096
   ```

   meaning 4,096 copies of the planned grid were the colony the search began from. The compiler
   itself is not published, and nothing here says how it mutates or selects.

3. **What it kept, it kept because the answer was exact.** The exams say `confirm 65536`: a
   candidate survives only if every one of 65,536 rows drawn from the truth table comes out right,
   on every one of the 32 output bits. `lean wires` then prunes gates that do not carry the answer. For
   three of the sixteen pieces the kept grid is leaner than the planned one — `pp04` by one gate,
   `pp14` by two, `pp15` by two — 499 planned gates against 494 kept. For the other thirteen the
   search kept the plan unchanged.

4. **From nothing it did not work.** `tiny.txt` is a 2-bit serial multiplier asked for with no
   `start` line: 16 cases, 64 answer bits, the whole table, and no seed grid. The search reached 62
   of the 64 bits on that seed and never closed the gap. That file is here because it is part of
   the record. Nothing in this directory was grown from nothing.

   The serial adder the sixteen pieces are summed by went the same way, and further: it was not
   planned by a script but **laid out by hand**, nine gates on thirty-two rows, then seeded as 4,096
   copies and kept unchanged — its exam records that from nothing the search reached about half of
   its table. That grid is not published here; `mul16.v`'s fifteen adders are fifteen instances of
   the one module it produced.

So: **the layout was planned by construction, an evolutionary search checked it on all 65,536
confirm rows and pruned it, and unseeded search did not converge.**

Say that number honestly: a piece's case is 32 bits wide, so its full table is 2^32 rows and 65,536
of them is one row in 65,536. They are drawn from across the whole table rather than from a corner
of it (see below), and every kept gate is load-bearing under them, but this is a strong check and
not a proof.

## The files

| | |
|---|---|
| `pp00.grid` .. `pp15.grid` | the planned grids, written by `ppplan.py` |
| `pp00.txt` .. `pp15.txt` | the exam each planned grid answers |
| `pp00fed.grid` .. `pp15fed.grid` | the kept grids — what the search returned, and what `hardware/mul16.v` is built from |
| `pp00fed.txt` .. `pp15fed.txt` | the exams carrying the `start` and `keep` lines |
| `tiny.txt` | the unseeded 2-bit attempt that failed |
| `verify_pieces.py` | the verifier |
| `ppplan.py`, `dotplan.py`, `dotsim.py` | the constructive planner and the two modules it is built on |
| `mulcompose.py` | the script that wrote `hardware/mul16.v` |

The exam files' own header comments name the paths those files had in the repository they were made
in. `expeditions/mul/pp<NN>.grid` is this directory's `pp<NN>.grid`;
`board/research/ternary/ppplan.py` and `board/research/ternary/mulcompose.py` are this directory's
`ppplan.py` and `mulcompose.py`; `board/mul16.v` is `hardware/mul16.v`. The exam text is left
exactly as it was so that re-running the planner reproduces these files byte for byte.

## The grid format

A `.grid` file is whitespace-separated integers and nothing else. A grid is `rows` by `columns`;
column 0 is fed and never written, so the file holds `(columns - 1) * rows` cells, **written column
by column**, and within a column the rows top to bottom. Cell `(at, row)`, for `at` in
`1..columns-1`, is at index `(at - 1) * rows + row`.

`rows` and `columns` come from the exam file beside it — the numbers are not in the grid file.

### What a row is

Reading a grid top to bottom, with `state` from the exam:

| rows | what they are |
|---|---|
| `0` | input `x`: bit `t` of the multiplicand at step `t`, nought once the word is over |
| `1` | input `y`: bit `t` of the multiplier at step `t`, nought once the word is over |
| `2 .. state+1` | **state**: whatever a state row holds in the last column is carried into the next step's column 0. This is the only memory the machine has |
| `state+2` | the **lane**: the answer, read at the last column of every step, one bit a step |
| `state+3 .. rows-2` | **fresh** rows: nought at column 0 of every step, never carried. Scratch space, and a source of the constant nought a `maj` or `sum` needs |
| `rows-1` | the **constant one** |

For the 32-row pieces `state` is 27, so the lane is row 29, row 30 is fresh and row 31 is the one.
For the 64-row pieces `state` is 59, the lane is row 61, row 62 is fresh, row 63 is the one.

### What a column is

A step runs left to right across the columns. Column 0 is set up as above. For each column after
it, every cell reads the **previous** column and writes its own:

| kind | name | what it writes |
|---:|---|---|
| `0` | wire | the previous column's value on this row, unchanged |
| `3` | xor | `up ^ down` |
| `4` | not | `1 - straight` |
| `5` | maj | `maj(up, down, straight)` — the majority of the three |
| `7` | sum | `up ^ down ^ straight` |
| `10` | up | the value on row `row - 1` |
| `11` | down | the value on row `row + 1` |
| `13` | leap | the value on row `row + rows/2` |
| `14` | hop | the value on row `row + rows/4` |

`up` is the previous column on row `(row - 1) mod rows`, `down` on `(row + 1) mod rows`, `straight`
on `row` itself; every neighbour is taken modulo `rows`, so the top and bottom rows are neighbours.
`sum` and `maj` together are a full adder: the same three inputs give the sum bit and the carry.

Of the nine kinds, these grids use eight: `0`, `4`, `5`, `7`, `10`, `11`, `13`, `14`. No `xor`.

### What the exam says

```
inputs 32          the case is 32 bits wide
outputs 32         32 answer bits are asked for
rows 32            the grid
columns 8
steps 16           16 steps of word
state 27           27 state rows, so the lane is row 29
hold 16            then 16 steps with both inputs fed nought
fold 1 below       one lane, read below the state rows, a bit a step
gates sum maj not up down leap hop     the kinds the search was allowed
colony 4096        how the compiler was run
renew 32
sample 256
confirm 65536      65,536 rows drawn from the truth table must be exactly right
packed
lean wires         prune any gate that does not carry the answer
seed 4343
serial pp 0        the filler: which function is being asked for
start expeditions/mul/pp00.grid 4096   4,096 copies of the planned grid as the starting colony
keep expeditions/mul/pp00fed.grid      where the kept grid was written
```

`serial pp i` asks for one partial product. A case is 32 bits: bit `2t` is `x` bit `t`, bit `2t+1`
is `y` bit `t`. At step `k` the lane must show bit `k` of `(x * y_i) << i`, over all 32 steps.

The 65,536 confirm rows are not the first 65,536 cases and not a low corner of the table — they are
drawn from the whole 2^32 table by splitmix64 on the exam's seed: case `at` is
`splitmix64(seed + (at + 1) * 0x9E3779B97F4A7C15)` masked to `inputs` bits. With `seed 4343` those
65,536 cases are all distinct, and each of the 32 case bits is set in between 32,397 and 33,043 of
them. `verify_pieces.py` reproduces that draw exactly.

## Running the verifier

```
python3 verify_pieces.py                 # every kept grid, pp00fed .. pp15fed
python3 verify_pieces.py --planned       # the planner's grids instead
python3 verify_pieces.py GRID EXAM       # one grid under one exam
python3 verify_pieces.py --cases N ...   # fewer or more confirm rows than the exam asks
```

It is standalone: the Python standard library and nothing else — no numpy (which this repository
does use elsewhere), and no import from the repository the grids were made in. It runs all 65,536
cases at once by holding one integer per grid row with a bit per case, so a piece costs one pass.
It prints a line per piece and exits non-zero if any case is wrong.

```
$ python3 verify_pieces.py
pp00fed.grid: 15 gates, 32 rows, 8 columns, lane 29, serial pp 0: 65536 of 65536 confirm rows whole (0.1s)
...
pp15fed.grid: 43 gates, 64 rows, 12 columns, lane 61, serial pp 15: 65536 of 65536 confirm rows whole (0.1s)
16 of 16 pieces whole on every confirm row (1.8s)
```

Under 2 seconds for all sixteen pieces, 1,048,576 cases and 33,554,432 answer bits in total
(Python 3.13, one core).

### It can fail

A verifier that cannot fail proves nothing, so:

* Change `pp08fed.grid` column 7 row 45 from kind `5` (maj) to kind `7` (sum) — one cell of 704 —
  and it says
  `pp08bad.grid: ... WRONG on confirm row 0, case 0x4fc5801e: x=47878 y=14467 y_8=0 lane 0x00bb0000 wants 0x00000000`,
  and exits 1. `y` is 14467, whose bit 8 is 0, so that piece must show nothing at all; the broken
  grid puts a burst on the lane.
* Erase any one of the 494 kept gates — set its cell to `0` — and the piece fails. All 494 of them,
  one at a time, across all sixteen pieces: **494 of 494 caught**. That is what `lean wires` buys:
  there is no gate in these grids to spare.

The converse is worth stating plainly. The verifier checks the *answer*, not the gate count, so
adding a gate on a dead row (for example `pp08fed.grid` column 3 row 20, kind `10`) still passes —
it is a 41-gate grid computing the same function. Leanness is a property the compiler enforced, not
something this verifier can confirm.

## Re-laying the plan

```
python3 ppplan.py all --out=/some/empty/dir
```

lays all sixteen pieces again from nothing but the construction, checks each on the host model, and
writes the grids and both exam files. Fourteen of the sixteen come out **byte for byte identical**
to the `pp<NN>.grid`, `pp<NN>.txt` and `pp<NN>fed.txt` published here.

Two do not, and this is worth saying rather than hiding: `pp08` comes out at 39 gates in 16 columns
and `pp10` at 42 gates in 12 columns, where the published grids are 40 gates in 12 columns and 40 in
10. The planner was edited after those two were written, and it now settles on a different geometry
for them. Both re-laid grids are exact — `verify_pieces.py` passes them on all 65,536 confirm rows —
they are simply not the same grids. The other fourteen reproduce exactly.

`ppplan.py` needs `dotplan.py` and `dotsim.py`, which sit beside it. `dotsim.py` is the host model
of the machine described above, the scalar cousin of what `verify_pieces.py` does in parallel;
`dotplan.py` is the layout planner `ppplan.py` builds on. Neither searches: every gate is placed by
a shortest path, and the beam in both is over placements, never over populations.

## Composing the Verilog

`mulcompose.py` is the script that wrote `hardware/mul16.v`. It **cannot be run end to end from
this repository**: turning a grid into a Verilog module is done by `oracle netlist`, a subcommand of
the circuit compiler, and that compiler is not published; neither is the nine-gate serial adder grid
the tree of adders comes from — the hand-laid one of point 4 above. It is here because the
composition is the part a reader most needs to check, and it can be checked:

* Its `top()` function generates the `mul16` module. That generated module is **identical, line for
  line, to the `mul16` module in `hardware/mul16.v`** — 66 lines, sixteen leaf pieces, a four-level
  balanced tree of fifteen serial adders, a register on every node, latency 5, 32 clocks a product.
* Its `piece_testbench()` and `testbench()` functions write the testbenches, in full, here. Run them
  against the published `hardware/mul16.v` in iverilog and the modules agree with the grids: each
  `pp<NN>` module is whole on 300 operand pairs, and `mul16` is whole on 2,000 products back to back
  and 64 one at a time.

The research copy of this script wrote a header comment into every file it generated. This
repository allows no comments in a `.v` file, and `hardware/mul16.v` has none; those lines are gone
here and this README says what they said instead. The paths it uses were also pointed at this
directory and at `hardware/mul16.v`.

## What is not here, and why

The compiler. No search engine, no GPU kernels, no mutation or selection, no expedition machinery
beyond the exam files above. Publishing the grid format and a scorer says nothing about how a grid
is found, which is the point: everything needed to check the claim is here, and nothing needed to
reproduce the tool is.

## The sixteen pieces

| piece | rows | columns | lane | planned gates | kept gates | confirm rows |
|---|---:|---:|---:|---:|---:|---:|
| pp00 | 32 | 8 | 29 | 15 | 15 | 65,536 |
| pp01 | 32 | 8 | 29 | 18 | 18 | 65,536 |
| pp02 | 32 | 8 | 29 | 17 | 17 | 65,536 |
| pp03 | 32 | 8 | 29 | 17 | 17 | 65,536 |
| pp04 | 32 | 8 | 29 | 21 | **20** | 65,536 |
| pp05 | 32 | 8 | 29 | 24 | 24 | 65,536 |
| pp06 | 32 | 10 | 29 | 26 | 26 | 65,536 |
| pp07 | 32 | 12 | 29 | 29 | 29 | 65,536 |
| pp08 | 64 | 12 | 61 | 40 | 40 | 65,536 |
| pp09 | 64 | 12 | 61 | 42 | 42 | 65,536 |
| pp10 | 64 | 10 | 61 | 40 | 40 | 65,536 |
| pp11 | 64 | 10 | 61 | 42 | 42 | 65,536 |
| pp12 | 64 | 10 | 61 | 40 | 40 | 65,536 |
| pp13 | 64 | 10 | 61 | 40 | 40 | 65,536 |
| pp14 | 64 | 10 | 61 | 43 | **41** | 65,536 |
| pp15 | 64 | 12 | 61 | 45 | **43** | 65,536 |
| | | | | **499** | **494** | **1,048,576** |
