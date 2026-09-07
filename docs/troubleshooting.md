# Troubleshooting

Start with `./doctor.sh`. It checks everything below and prints the fix on the same line. It changes
nothing, so it is always safe to run.

---

## It answers correctly but at about 6-7 tokens a second

**The fabric is at 100 MHz.** `fpgautil -b kv260-bitnet.bit.bin` *without* `-o kv260-bitnet.dtbo`
programs the fabric and leaves `pl0_ref` at the boot firmware's 99999999 Hz. The design still works
and every answer is still right; it is 2.5× slower and there is no error message anywhere.

```sh
sudo grep ' pl0_ref ' /sys/kernel/debug/clk/clk_summary    # must read 249999998
sudo ./setup.sh --fabric                                   # reload with the overlay, refuse on 99999999
```

## It answers at about 8 tokens a second and the numbers otherwise look right

**The ARM output head is in use.** It costs about 52 ms a call against the fabric head's 5.40 ms.
Either `head3_t.bin`/`head_t_scale.bin` are missing, or `/dev/udmabuf2` is missing.
`bitnet_chat.py --head auto` says which on stderr when it falls back. Those two files are the one
optional part of the download, so `prep.sh` will have said on its own output that it could not get
them:

```sh
sudo ./setup.sh --weights                        # fetch them, and check them
python3 tools/pack_head_ternary.py /path/to/dir  # or pack them from the checkpoint instead
```

`--temp` above 0 also forces the ARM head, because sampling needs every logit and the fabric head
does not compute them. That is by design and the runtime says so.

## `modprobe u-dma-buf` says "Invalid module format" after a kernel upgrade

Expected, and the fix is one command. `setup.sh` installs the module under
`/lib/modules/$(uname -r)/extra` and runs `depmod` — it does **not** use DKMS, so an out-of-tree
module built against the old kernel's headers will not load into the new one.

```sh
sudo ./setup.sh --stage1        # rebuilds it against the running headers
```

If you would rather it were managed for you, `github.com/ikwzm/u-dma-buf-kmod-dpkg` packages it
with DKMS.

## `modprobe u-dma-buf` fails, or a buffer is smaller than asked for

**CMA.** With `cma=1000M` the three buffers leave `CmaFree` at 14,852 kB of 1,024,000 — there is no
headroom at all, which is why it is not optional.

```sh
grep -E 'CmaTotal|CmaFree' /proc/meminfo        # CmaTotal must be >= 1024000 kB
sudo ./setup.sh --bootargs                       # prints your command line and the config files it finds
```

`setup.sh` deliberately does **not** edit the boot configuration: the mechanism differs between Kria
images and getting it wrong makes a board that will not boot.

## A buffer's `phys_addr` is at or above 0x100000000

The engines carry a 40-bit address but the runtime maps `DDR_LOW`, so all three buffers must be
below 4 GiB. A reboot re-places the CMA reservation. This was never seen on the reference board and
would present as a DMA to the wrong place rather than an error, which is why `doctor.sh` asserts it.

## Loading a different bitstream appears to do nothing

**The overlay names the firmware by file name.** `firmware/kv260-bitnet.dtso` says
`firmware-name = "kv260-bitnet.bit.bin"`, and `fpgautil` copies the file you give it into
`/lib/firmware` under that name. Pointing `fpgautil` at a differently-named file **leaves the old
bitstream in the fabric**. Swapping means copying over that name. And an already-live overlay must be
removed first:

```sh
sudo rmdir /sys/kernel/config/device-tree/overlays/full
```

`setup.sh --fabric` does both. `doctor.sh` compares `/lib/firmware/kv260-bitnet.bit.bin` with the
one in the repository and fails if they differ.

## `prep.sh` says the address to fetch the weights from has not been set

The packed weights are downloaded from an address the owner of a copy of this repository sets by
hand, and it has not been set here. `tools/prep.sh` stops before downloading anything rather than
trying an address that is not one and failing with an HTTP error. Edit the `WEIGHTS_URL` line at
the top of `tools/prep.sh`; [weights.md](weights.md) says exactly what to put there.

If every file is already in the directory and already correct, `prep.sh` never asks for an address
at all — it verifies what is there and exits 0. Seeing this message therefore also means at least
one file is missing or wrong, and the lines above it name which.

## A fetched file's size or md5 does not match

`prep.sh` refuses, deletes the file it could not verify, and stops without fetching anything after
it. It tries twice first: once resuming, then once from the start, because a truncated download and
a wrong file look the same until you have the whole thing.

Take the refusal seriously. Every file is checked against a value recorded in this repository —
`manifest.json` against `selftest/expected_ids.json`, everything else against `manifest.json` — so a
mismatch means the address is serving files this repository was not measured on, or the download is
being corrupted. Neither is something to work around. [weights.md](weights.md) sets out the whole
chain.

```sh
python3 tools/checkfiles.py /path/to/dir --encoding base3   # the same check, on its own
```

## `bitnet_kria` refuses to start, naming the encoding

```
this build streams base 3, five weights a byte and eighty a beat; the manifest says (none)
with 64 a beat and 4 a byte (model.bin). Repack with pack_model.py --encoding base3.
```

That is the guard working. This bitstream's engines have no two-bit path.
`python3 tools/pack_model.py OUT --encoding base3`.

## `bitnet_kria stopped before it was ready`

It opens `/dev/mem`, so it must run as root. Use `./chat.sh`, which handles the whole invocation —
`sudo env PYTHONPATH=<the normal user's site-packages> python3 bitnet_chat.py --no-sudo ...`. The
awkwardness is not avoidable: the runtime must be root while `tokenizers` is a `--user` install of
the normal user, and sudo's credential cache is per-session with no tty, so a python parent that
spawns `sudo ./bitnet_kria` must itself be started under sudo.

## `no tokenizers module`

`tokenizers` must be installed for the **normal** user, not for root:

```sh
pip3 install --user tokenizers        # as ubuntu, not under sudo
```

## Everything is just slow, and the shift scan looks large

**Something else is using a core.** `polkitd` spinning at 88.9% of one A53 charged a busy core to
two entire passes of measurement in this project's history. With a quarter of an A53 taken,
generation went from 16.33 to 12.03 tok/s; with a whole core, to 0.82. An ssh login alone costs
about a core-second of daemon churn.

```sh
top -bn1 | head -15
sudo systemctl restart polkit
```

Then compare the runtime's own `shift scan` figure against its **2.0-3.0 ms a forward** band on a
quiet board (`--timing` prints it).

Note one honest caveat: on a small number of prompts the shift scan costs three to eight times its
norm even on a quiet board. It hit 3 of 31 battery prompts in the recorded pass and none of six
360-position runs. It is not diagnosed, it is not a regression, and it is the reason a battery
average and a bench average can disagree by more than noise.

## The first run after copying files is dreadful

Expected. The model is read from storage rather than the page cache: 8.13 s of setup at 51 MB/s
against 0.87 s at 482 MB/s warm, and the first prompt is worthless (6.52 tok/s, a 967 ms shift scan)
while the readahead is still in flight. The same process's second and third repeats gave 29.11 and
16.81 tok/s. **Throw the first run away.**

## `./selftest.sh` says the token ids differ

Take it seriously — greedy decoding is deterministic, and the recorded ids agree across four
independent runs of the reference board. In order of likelihood:

1. a different bitstream (check `doctor.sh`'s bitstream md5 line);
2. a different or damaged `model3.bin` (`python3 tools/checkfiles.py <dir>`);
3. a different `tokenizer.json` — this shows up as **prompt** ids differing, which `check_ids.py`
   reports separately;
4. the ARM head where the recording used the fabric head (`--head fabric` explicitly).

## The stage check is skipped

It needs `refdump/stages.bin` from the numpy reference, which is built from the **original
checkpoint** and is not part of the packed download. It is not made unless you ask for it:

```sh
tools/prep.sh --local /some/dir --dump         # fetches the checkpoint too and builds refdump/
python3 tools/ref_model.py --dump refdump      # or by hand, then copy refdump/ to the model directory
```

`refdump/` is about 25 MB and is not in the repository, because it is derived from the weights.
`--dump` needs `numpy`, `tokenizers` and `huggingface_hub`, and downloads 1.18 GB of
`model.safetensors`, which is why it is opt-in. See [weights.md](weights.md).

## `pack_head_ternary.py` says it skipped the shortlist self-check

Expected, and not a problem. That part of the check scores the ternary shortlist against real final
hidden states (`head_states.npz`), which is a research artefact this repository does not ship. The
file checks that matter — the head read back, unpacked and compared with the pattern in memory, and
the first beat rebuilt by hand — still run, and the head is written and usable. The measurement
being skipped came out, on the reference board, as a worst rank of 56 out of K = 256 over 2115 real
states; `head_t.json` records `states_skipped` so nobody mistakes a skip for a pass.

If the packer fails outright, the runtime still answers with `--head arm`, at about 52 ms more a
generated token. This only comes up if you are repacking; the default path downloads the head
already packed, and `prep.sh` continues either way.

## `xmutil loadapp` misbehaves

**The Kria app path in `firmware/install-firmware.sh` has never been run on the reference board:**
every number in this repository was measured with the `fpgautil` path that `setup.sh --fabric` uses,
and the installer checks `pl0_ref` afterwards exactly because nothing here has proved that path.

The installer compiles the overlay on the board — the `.dtbo` is not in this repository, because the
blob has to match the device tree of the board it is loaded into — and installs the three files
every Kria app needs, `kv260-bitnet.bit.bin`, `kv260-bitnet.dtbo` and `shell.json`, into
`/lib/firmware/xilinx/kv260-bitnet/`:

```sh
sudo firmware/install-firmware.sh
sudo xmutil unloadapp
sudo xmutil loadapp kv260-bitnet
sudo firmware/install-firmware.sh --verify      # fpga0 operating, pl0_ref 249999998
sudo firmware/install-firmware.sh --remove      # unload it and delete the app directory
```

If it misbehaves, use the proven path. Everything measured here was loaded with:

```sh
sudo fpgautil -R; sudo xmutil unloadapp
sudo fpgautil -b kv260-bitnet.bit.bin -o kv260-bitnet.dtbo
```

## The power or temperature readings look wrong

`runtime/edge_monitor.py` hardcodes `/sys/class/hwmon/hwmon2` for the INA260 and
`/sys/class/hwmon/hwmon0` for the AMS. That numbering is a driver-probe-order accident of the
reference board's image, not a stable interface. Check with
`cat /sys/class/hwmon/hwmon*/name` and edit the two constants.
