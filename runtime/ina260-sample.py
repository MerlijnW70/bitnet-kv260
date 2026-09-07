"""Sample the KV260's INA260 (hwmon2, ina260_u14) into a log: one line "epoch_s uW mA mV".

One process that keeps the three sysfs files open instead of a shell loop forking date+cat every
period: on the four A53s the forking loop is itself a load worth about 0.15 W and 10% of the
generation rate, which would be charged to whatever is being measured.

usage: python3 ina260-sample.py LOGFILE [PERIOD_S]
"""
import sys
import time

H = "/sys/class/hwmon/hwmon2"


def main():
    log = sys.argv[1] if len(sys.argv) > 1 else "power.log"
    period = float(sys.argv[2]) if len(sys.argv) > 2 else 0.05
    files = [open(f"{H}/{n}", "rb") for n in ("power1_input", "curr1_input", "in1_input")]
    with open(log, "w", buffering=1) as out:
        while True:
            vals = []
            for f in files:
                f.seek(0)
                vals.append(f.read().strip().decode())
            out.write(f"{time.time():.6f} {vals[0]} {vals[1]} {vals[2]}\n")
            time.sleep(period)


if __name__ == "__main__":
    main()
