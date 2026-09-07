# The one configuration this was measured on

Every number in this repository was taken on **one** KV260. This file says exactly which one, so
that when something behaves differently on yours you know where to look first.

## Tested on

| | |
|---|---|
| board | Kria KV260 Vision AI Starter Kit, `xck26-sfvc784-2LV-c` |
| image | Ubuntu 22.04.4 for Kria |
| kernel | `5.15.0-1027-xilinx-zynqmp aarch64` |
| RAM | 3911 MB, no swap |
| kernel command line | includes `cma=1000M` (`CmaTotal 1024000 kB`) |
| u-dma-buf | built from `github.com/ikwzm/udmabuf` against those headers; `udmabuf0=545259520 udmabuf1=8388608 udmabuf2=83886080` |
| compiler | gcc 11.4.0 |
| bitstream | `firmware/kv260-bitnet.bit.bin`, md5 `1f7e4fcc5ee4c905ebcfb57f60fe9675`, 7,797,692 B |
| built with | AMD Vivado Design Suite 2026.1, on an Enterprise licence, in 15 min 57 s on a Ryzen 9 9950X |
| timing | post-route, slow corner: WNS 0.108 ns, TNS 0.000, WHS 0.010, THS 0.000; 0 failing endpoints of 178,007 |
| fabric | 42,271 CLB LUTs (36.09%), 53,342 registers (22.77%), 95 of 144 BRAM tiles, 40 of 64 URAM, 0 DSP |
| PL clock | `pl0_ref` 249999998 Hz, set by the device-tree overlay, not by `fpgautil -b` alone |
| storage | one microSD card that reads at 15-51 MB/s cold and 465-483 MB/s from the page cache |
| model | `microsoft/bitnet-b1.58-2B-4T`, packed base 3: `model3.bin` md5 `dbb41a46744c176d31b0ce79bd82f166`, `head3_t.bin` md5 `97ba6c13749370a39adf42e821578e12`, `manifest.json` md5 `45cabaf7c91635df93069195d11202d0` |

## Not tested, and not claimed

- **Any other kernel.** u-dma-buf is an out-of-tree module built against the running headers. A
  different kernel may not build it, may name its sysfs entries differently, or may change
  `sync_for_cpu` semantics — the runtime opens `/sys/class/u-dma-buf/udmabufN/sync_for_cpu` by path.
- **Ubuntu 24.04 for Kria, or PetaLinux.** It is not known whether they carry `fpgautil` and
  `xmutil` at all.
- **Any other KV260 revision or speed grade.** The design closes with 0.108 ns of slack, which is
  0.4% of a 4 ns period. On a `-1` part, or another SOM revision, the timing would have to be
  re-closed and nothing here says it would close. That is expected, not a bug in the design.
- **Any other Zynq UltraScale+ board.** KR260, KD240 and the ZCU boards are untried.
- **The free Vivado edition.** `build-ffn.log` line 1 reads *"A valid Vivado Design Suite ENTERPRISE
  license has been detected."* Kria is listed as supported at every Vivado tier in UG973, and the
  design uses no fee-based IP, so it should build — but it was not tried, so it is not claimed.
  **You do not need Vivado to run any of this.** The bitstream is in the repository.
- **The physical addresses.** `udmabuf0 0x0038300000`, `udmabuf1 0x0058b00000`,
  `udmabuf2 0x0059300000` are what *this* board's allocator returned with *this* CMA reservation.
  The requirement is only that all three land below 4 GiB, because the runtime maps `DDR_LOW`;
  `doctor.sh` checks it. Nothing here proves another allocator will satisfy it, and the failure
  would be a DMA to the wrong place rather than an error.
- **The power and temperature readings.** `runtime/edge_monitor.py` hardcodes
  `/sys/class/hwmon/hwmon2` (the INA260) and `/sys/class/hwmon/hwmon0` (the AMS). That numbering is
  a driver-probe-order accident of this image, not a stable interface. On another image those
  readings would silently come from the wrong device.
- **The idle power baseline.** 3.808 W idle with this bitstream loaded (3.535 W with the earlier
  2-bit one; the batched engine's 40 UltraRAMs cost about 0.274 W at rest whether or not a token is
  being made). Every joules-a-token figure is a difference against *this* board's idle.
- **Cold-load times.** 15-51 MB/s and up to 55.30 s of first setup are this microSD card's. A
  different card changes the first-run experience by tens of seconds.

If you run it somewhere else and it works, the numbers would be welcome. If the timing does not
close on your part, that is expected.

## The one thing that is not a property of the board

`polkitd` spinning at 88.9% of one A53 charged a busy core to two whole passes of measurement in
this project's history. That is a property of one installation, not necessarily of the image.
`doctor.sh` refuses to pass when any process is above 10% of a core, and names the fix.
