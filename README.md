# bitnet-kv260

## 1. The board

A **Kria KV260 Vision AI Starter Kit** (`xck26-sfvc784-2LV-c`) with its 12 V supply, the **Ubuntu
22.04 image for Kria** on a microSD card with about 20 GB free, and a network path once.

You do not need Vivado, a Vivado licence or an AMD account. The built bitstream is in this
repository, at `firmware/kv260-bitnet.bit.bin`.

## 2. Fetch the weights

On the board:

```sh
git clone <this repository> && cd bitnet-kv260
sudo ./setup.sh
sudo reboot
sudo ./setup.sh --stage2
```

`setup.sh` prints `Now: sudo reboot, then  sudo ./setup.sh --stage2` when the reboot is needed and
goes straight through when it is not. To fetch and check the weights on their own:

```sh
sudo ./setup.sh --weights
```

## 3. Load the application

```sh
sudo firmware/install-firmware.sh
sudo xmutil unloadapp
sudo xmutil loadapp kv260-bitnet
sudo firmware/install-firmware.sh --verify
```

## 4. Check it

```sh
./doctor.sh
./selftest.sh
```

The first thing `doctor.sh` checks and reports is the **fabric clock**. It reads `pl0_ref` out of
`/sys/kernel/debug/clk/clk_summary` and requires `249999998`. At `99999999` the fabric is running at
100 MHz instead of 250, every answer is still correct, everything is 2.5 times slower and nothing
else on the board reports an error — so `doctor.sh` fails loudly on it and `selftest.sh` refuses to
run the board tests at all until it passes.

The forty alarms:

```sh
cd /home/ubuntu/bitnet-kria
sudo env PYTHONPATH=/home/ubuntu/.local/lib/python3.10/site-packages \
     python3 edge_monitor.py --job triage --no-sudo --json-out edge.json
```

---

Everything else — what it is, how it works, what was measured, what was verified and what was not,
and what to do when a step fails — is in [docs/](docs/), starting at
[docs/overview.md](docs/overview.md).
