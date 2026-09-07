# The weights: where they come from and what is checked

The 1.4 GB of packed model files are not in this repository and are no longer packed from the
original checkpoint as the default path. They are **downloaded ready-made** from an address you
set once, and every byte of every file is checked against a value recorded in this repository
before anything is allowed to use it.

`tools/prep.sh` owns the download. `setup.sh --weights` calls it on the board; nothing else in the
tree fetches these files.

---

## Where they come from

The files live at
[`merlijn70w/bitnet-kv260-weights`](https://huggingface.co/merlijn70w/bitnet-kv260-weights), a public
Hugging Face model repository, and `tools/prep.sh` already points at it on line 4:

```sh
WEIGHTS_URL="${BITNET_WEIGHTS_URL:-https://huggingface.co/merlijn70w/bitnet-kv260-weights/resolve/main}"
```

That is the only definition of the address in the repository. File names are appended to it, so the
address has to serve `.../model3.bin`, `.../manifest.json` and the rest. A trailing slash is stripped
for you, and the scheme is required: `prep.sh` treats anything that does not start with `http://`,
`https://` or `file://` as not set and stops with status 2 before downloading anything, rather than
failing later with an HTTP error.

To serve them from somewhere else — your own mirror, a local file:// directory, an internal server —
set `BITNET_WEIGHTS_URL` in the environment for one run, which `setup.sh --weights` passes through,
or edit that one line.

If every file is already in the directory and already matches its recorded size and md5 — after an
`rsync` from another machine, say — `prep.sh` verifies them and exits 0 without needing an address at
all.

## What is fetched, and from where

From `$WEIGHTS_URL`:

| file | about | what it is |
|---|---|---|
| `manifest.json` | small | geometry, the stream format, and the size and md5 of every other file |
| `model3.bin` | 417,546,240 B | every ternary matrix of all 30 layers, five weights a byte in base 3 |
| `norms.bin` | 1,761,280 B | the RMS-norm weights |
| `embed_bf16.bin` | 656,670,720 B | Microsoft's bf16 embedding table, byte for byte |
| `head_i8.bin` | 328,335,360 B | the tied output head quantised per row |
| `head_scale.bin` | 513,024 B | that head's per-row scales |
| `head3_t.bin` | 65,667,072 B | the ternary stage-1 head the fabric streams |
| `head_t_scale.bin` | 513,024 B | that head's per-row scale |
| `head_t.json` | small | the size and md5 of the two files above |

`head_t.json`, `head3_t.bin` and `head_t_scale.bin` are one group and are the only optional part.
If they cannot be fetched, `prep.sh` says so and carries on: the runtime still answers with
`--head arm`, at about 52 ms more a generated token. The six above them — `manifest.json`,
`model3.bin`, `norms.bin`, `embed_bf16.bin`, `head_i8.bin`, `head_scale.bin` — are required, as is
`tokenizer.json` below, and a failure on any of those stops the run.

The sizes above are for orientation. Nothing checks against *this* table — the script checks
against `manifest.json`, which is itself checked first.

### The tokenizer is fetched separately, and stays that way

`tokenizer.json` is **not** taken from `$WEIGHTS_URL`. It comes from the Hugging Face repository of
the checkpoint itself:

```
https://huggingface.co/microsoft/bitnet-b1.58-2B-4T/resolve/<revision>/tokenizer.json
```

where `<revision>` is the `tokenizer_snapshot` recorded in `selftest/expected_ids.json`
(`04c3b9ad9361b824064a1f25ea60a8be9599b127`), so the file is pinned to the exact commit whose
tokenizer produced this repository's recorded prompt ids. The reason it is fetched rather than
redistributed: Microsoft ships `tokenizer.json` under MIT, but whether a re-derived Llama 3 BPE
table counts as "Llama Material" under Meta's community licence is untested, and one 17 MB download
deletes the question.

`BITNET_CHECKPOINT` overrides the repository if you need a mirror; the revision still comes from
`selftest/expected_ids.json`, and the md5 check below still applies.

## What every file is checked against

The chain has one root in this repository and everything else hangs off it.

1. **`manifest.json` against `selftest/expected_ids.json`.** That file records
   `manifest_md5 = 45cabaf7c91635df93069195d11202d0`, the md5 of the manifest that was on the
   reference board when every number in `results/` was measured. `prep.sh` fetches
   `manifest.json` first and refuses to go on unless it hashes to exactly that. The same value is
   in `ENVIRONMENT.md` and in `results/kria-speed2-results.txt`.
2. **A cross-check before anything large is downloaded.** `selftest/expected_ids.json` also records
   `model3_md5` and `head3_t_md5` independently of the manifest. `prep.sh` compares the manifest's
   own entry for `model3.bin` against `model3_md5`, and `head_t.json`'s entry for `head3_t.bin`
   against `head3_t_md5`, and refuses if either disagrees. A manifest that matched its md5 but
   named a different model would be caught here.
3. **Every other file against the verified `manifest.json`.** `pack_model.py` wrote a `files`
   section holding `{bytes, md5}` for `model3.bin`, `norms.bin`, `embed_bf16.bin`, `head_i8.bin`,
   `head_scale.bin` and `tokenizer.json`. `prep.sh` checks the **size first** — which catches a
   truncated download or an HTML error page immediately, before spending time on the next 650 MB
   file — and then the md5.
4. **`head3_t.bin` and `head_t_scale.bin` against `head_t.json`**, which `pack_head_ternary.py`
   wrote the same way. `head3_t.bin` additionally has the recorded `head3_t_md5` behind it, so it
   is rooted in the repository. `head_t_scale.bin` is not: its only recorded value lives in
   `head_t.json`, which arrives from the same place it does. It is 513,024 B of per-row scales and
   the fabric head is wrong without it, so this is worth knowing, and it is the one file whose
   integrity rests on the server rather than on this repository.
5. **`tools/checkfiles.py` again at the end**, over the whole directory, as an independent pass that
   also asserts the stream format is `base3` — the shipped bitstream has no two-bit path.
   `doctor.sh` runs the same check on every invocation, so a file that rots later is caught too.

A file already in the directory whose size and md5 are right is not fetched again. A file that is
there but wrong is downloaded once with `curl -C -` (resume) and, if it is still wrong, once more
from the start; if it is still wrong after that, `prep.sh` deletes it and stops, naming the file. It
never leaves a file it could not verify.

## Running it

```sh
tools/prep.sh ubuntu@kria              # on a PC: fetch, check, rsync to the board
tools/prep.sh --local /some/dir        # fetch and check into a directory, copy nothing
sudo ./setup.sh --weights              # on the board: fetch and check straight into the model dir
```

`setup.sh --weights` runs `prep.sh --local "$DIR"` as the normal user, so the files are not left
owned by root, and `setup.sh --stage2` runs it between the fabric and the runtime build. It is
idempotent: run it twice and the second run only hashes what is already there.

## Repacking from the original checkpoint instead

The packers are still in the tree and still work standalone. Nothing in the setup flow calls them
any more, so use them directly:

```sh
pip install numpy huggingface_hub
python3 -c "from huggingface_hub import snapshot_download; snapshot_download('microsoft/bitnet-b1.58-2B-4T')"
python3 tools/pack_model.py OUT --encoding base3     # model3.bin, norms.bin, embed_bf16.bin,
                                                     # head_i8.bin, head_scale.bin, tokenizer.json,
                                                     # manifest.json
python3 tools/pack_head_ternary.py OUT               # head3_t.bin, head_t_scale.bin, head_t.json
python3 tools/checkfiles.py OUT --encoding base3
```

`tools/pack.py` holds the stream-format primitives `pack_model.py` imports and can be run on its own
to write the older two-bit single-matrix test vectors. `tools/fetch.py` pulls layer 0's MLP tensors
straight out of `model.safetensors` by HTTP byte range, which is how the packing was worked out.
Neither is on the path to a running board.

A repacked directory should reproduce the recorded md5s exactly — `pack_model.py` writes nothing
time-dependent — so `tools/checkfiles.py` and `doctor.sh` are the test of whether it did.

## The stage-check dumps

`refdump/` is the numpy reference's stage dumps, about 25 MB, which `selftest.sh` compares the board
against stage by stage. It is derived from the original checkpoint, not from the packed files, so it
is **not** part of the download and is no longer built by default:

```sh
tools/prep.sh --local /some/dir --dump        # downloads the checkpoint too and builds refdump/
python3 tools/ref_model.py --dump refdump     # or by hand, in the model directory
```

Without it `selftest.sh` reports the stage check as SKIPPED rather than failing. `--dump` needs
`numpy`, `tokenizers` and `huggingface_hub`, and downloads the 1.18 GB `model.safetensors`.

## Licensing

The packed files are derived from `microsoft/bitnet-b1.58-2B-4T`, which is MIT.
`embed_bf16.bin` and `norms.bin` are byte-for-byte Microsoft tensors; the rest are derivatives.
MIT's one condition is that the notice travels with them, so
`THIRD_PARTY_LICENSES/LICENSE.microsoft-bitnet` belongs **inside** whatever archive you serve from
`$WEIGHTS_URL`, not only in this repository. See `NOTICE`.
