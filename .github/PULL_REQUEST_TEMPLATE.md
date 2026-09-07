## What this changes, and why

## Do the generated tokens still match?

- [ ] This change cannot affect the arithmetic (documentation, packaging, tooling).
- [ ] It can, and greedy output is byte-identical: `./selftest.sh` passes and the ids match.
- [ ] It can, and the output **does** change. What moved, and by how much:

## What you ran, verbatim

```
```

## On what

Board, image and kernel if you ran anything on hardware; simulator version if you ran the
testbenches. `ENVIRONMENT.md` describes the reference board these numbers came from.

## What you did not test

## Checklist

- [ ] `python3 tools/no_comments.py` passes: no comments in any code file.
- [ ] `./selftest.sh --rtl-only` passes, or the change touches no RTL.
- [ ] Numbers quoted are measured, and projections are marked as projections.
- [ ] Any `[UNVERIFIED]` marker I made true is deleted, and I said what I ran.
