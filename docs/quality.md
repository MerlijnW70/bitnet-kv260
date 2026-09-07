# What the model is and is not good at

Two batteries were run, both with greedy decoding so that everything here is reproducible, and both
against the **same weights on other hardware** so that a mistake can be attributed to the model
rather than to the fabric.

- **26 prompts, 8 categories**, three ways: this board, `bitnet.cpp` on the same board's A53s, and
  `microsoft/bitnet-b1.58-2B-4T` in bf16 on an RTX 4080 through `transformers`. Judged by four
  agents, each given two categories and all three answers.
  `results/kria-quality.txt`, raw answers in `results/quality-answers.json`.
  Reproduce: `tools/quality_board.py`, `tools/quality_hf.py`, `tools/quality_bitnetcpp.py`,
  `tools/quality_merge.py`.
- **40 equipment alarms**, the same three ways, scored automatically against an answer key.
  `results/edge-demo-results.txt`, scores in `results/edge-scores.json`.
  Reproduce: `runtime/edge_monitor.py`, `tools/score_edge.py`.

## The good part: format discipline is total

| | board (fabric) | bitnet.cpp (A53) | RTX 4080 (bf16) |
|---|---|---|---|
| strict JSON, first try | **40/40** | 40/40 | 40/40 |
| keys exactly `{category, severity, action}` | **40/40** | | |
| `category` inside its seven-word vocabulary | **40/40** | | |
| `severity` inside its three-word vocabulary | **40/40** | | |

`json.loads` on the stripped answer, with no first-`{`-to-last-`}` rescuing. A gateway could parse
and route these with no human reading them.

Also good, and measured: no board answer was gibberish, none was in the wrong language, none drifted
further from the reference as it got longer, and the same prompt run twice gave byte-identical ids.
On the 256-token degeneration test the board did **not** fall into repetition and the RTX 4080 did.
Code answers came back with intact fences, 4-space indentation and an unmangled `[::-1]`.

## The bad part: judgement is poor, and it is the model's fault

| | board | bitnet.cpp | RTX 4080 |
|---|---|---|---|
| category right (30 clear lines) | 19/30 | 20/30 | 19/30 |
| **severity right** (30 clear lines) | **16/30** | 13/30 | **16/30** |
| both right | 12/30 | 10/30 | 12/30 |

**One "critical" in forty lines, and twelve of thirteen genuinely critical faults downgraded to
warning or info.** Anything downstream that acts on severity — paging, shutdown interlocks,
escalation — would sleep through an earth fault that had already opened a breaker.

**Instrument faults are missed almost entirely.** Sensor category: 1 of 6 over all forty lines, and
0 of the 4 clear ones. A transmitter frozen at the 4.00 mA rail for 47 minutes while the process
moved 12 K came back as `none`/`info` from all three ways.

**The `action` field is a hint, not an instruction.** Three of fourteen transcribed answers advise
something wrong or dangerous: reopen a breaker on an unresolved earth fault, reduce a load to the
value it already has, run the brake test that just failed.

**And it is slow for a control loop.** 14.5 s a mean answer, three quarters of it prompt evaluation,
is about four alarms a minute. A burst from one plant trip would queue.

**Every one of the eighteen mistakes the board makes on the clear lines, the RTX 4080 makes too, the
same way.** On the thirty clear lines the board and the GPU score identically, and 26 of the 40
answers are identical token for token. The demo's weakness is the model, not the board.

The same pattern in the 26-prompt battery — these are all wrong, and all wrong identically on the
RTX 4080 running the same weights: the invalid syllogism, the circular CRC-32 explanation, the haiku
that is not 5-7-5, the "four lines" that is three sentences, the false claim that a `for` loop takes
a condition, the wrong carry-out formula, the circled-plus read as addition and the micro sign read
as the Greek letter. BitNet b1.58 2B4T is trained mostly on English, and the Dutch quality measured
here — a correct answer followed by "populestat", "een very bekend stad" and filler — is likewise a
fact about the model, demonstrated on the GPU.

## What the fabric is responsible for, and how it scores

| | measured |
|---|---|
| generations byte-identical to the RTX 4080 | **11 of 26** (ids and text), including 90 ids on one and 77 off a 124-id prompt |
| teacher-forced agreement | **98.7%** of board tokens are the GPU's own top-1 at that position; **100%** inside its top-5 |
| every disagreement | on an HF logit gap of 0 to 0.5 |
| the other 15 generations | share a prefix and then take one different token; 12 of those forks are at generated position 4 or later (54, 42, 41, 41, 24, 11, 11, 11, 10, 9, 4, 102), each a synonym- or punctuation-level substitution, after which the board stays fluent |
| the judges' verdict | board worse than the GPU on **0 of 26**; equivalent or identical on 16; different but fine on 10; **better on 4** |
| integer stages against the numpy reference | **every one exact**, at 8 positions, through all 30 layers |

That is the shape of an integer, shift-quantised datapath meeting bf16 at a near-tied logit. It is
not drift and it does not grow with length: the longest identical generations are among the longest
generations, and on the 256-token test the board held the GPU's ids for 102 tokens and then stayed
coherent for the remaining 154 while the GPU's own continuation was the one that repeated.

## The honest summary

As an **offline structured-output front end** — take a line of telemetry, return a parseable object,
name the subsystem, keep the data on the device — this board does the job at 5.5 W with nothing on
the wire, 5.34× faster and on 5.04× fewer joules than the same board's own CPUs.

As a **judge of how urgent something is**, it does not, and no amount of fabric fixes that: the same
weights on a 4080 are exactly as wrong. What would fix it is a bigger model, or severity rules in
the prompt — and the second was deliberately not done here, because it would have been tuning the
prompt to the answer key.

Microsoft say the same thing about their own model, and it is worth repeating: *"We do not recommend
using BitNet b1.58 in commercial or real-world applications without further testing and
development."*
