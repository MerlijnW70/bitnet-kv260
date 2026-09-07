# Security

## Reporting something sensitive

Use GitHub's private vulnerability reporting: **Security → Report a vulnerability** on
[this repository](https://github.com/MerlijnW70/bitnet-kv260/security/advisories/new). It reaches the
maintainer privately and nothing is published until there is a fix. There is no email address to
write to, deliberately: an address in a public repository is scraped within days.

For anything that is not sensitive, open an [issue](https://github.com/MerlijnW70/bitnet-kv260/issues)
or a [discussion](https://github.com/MerlijnW70/bitnet-kv260/discussions) instead.

## What this project is, in security terms

It is a research demonstration that runs a language model on a board you own. It is not a service,
it has no network listener, and it authenticates nothing. Two facts matter more than any of that.

**The runtime needs root and maps physical memory.** `bitnet_kria` opens `/dev/mem` to reach the
AXI registers and the contiguous buffers, which is why every documented command runs it under
`sudo`. A bug in it can write anywhere in physical memory. Read it before you run it on a board that
matters, and do not run it on a machine you share with anyone you do not trust.

**Nothing is verified cryptographically.** The weights are checked against the sizes and MD5 sums in
`manifest.json` and `head_t.json`, which catches a truncated download or a corrupt file. It is not a
signature and it is not meant to be: MD5 is broken against a deliberate attacker, and the manifest
travels with the files it describes. If you need provenance, fetch from
[the model repository](https://huggingface.co/merlijn70w/bitnet-kv260-weights) over HTTPS and check
the commit there, or repack from `microsoft/bitnet-b1.58-2B-4T` yourself with `tools/pack_model.py`.

**The bitstream is a binary you are asked to load into your own FPGA.** It was built from the Tcl and
Verilog in `hardware/` by Vivado 2026.1, and `docs/build-bitstream.md` says how to rebuild it and
compare. If you would rather not load a binary from a stranger, rebuild it.

## What is in scope

A report is worth making about: a memory error in the runtime, a path that writes outside the
directories it should, a check that can be made to pass on wrong data, or anything in the setup
scripts that damages a board.

Not in scope, because they are known and documented: the model says wrong things, which
`results/kria-quality.txt` and `results/edge-demo-results.txt` measure and the same weights on a GPU
reproduce; and the fabric runs at 100 MHz with no error when the overlay does not load, which
`doctor.sh` exists to catch.
