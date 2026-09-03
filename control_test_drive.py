"""
Straight-line stops for one or more lenses on the single LSTMABS config, switching the
net at runtime through controller.setLens. Names containing "DynamicABS" are treated as
vehicle configs (.pc) instead. Appends every stop to experiments/<lens>/runs.csv.
"""
import csv
import re
import sys
import time
from datetime import datetime
from pathlib import Path

import test_lenses_drive as T

NAMES = (sys.argv[1] if len(sys.argv) > 1 else "baseline_dyn").split(",")
MPH = int(sys.argv[2]) if len(sys.argv) > 2 else 60
RUNS = int(sys.argv[3]) if len(sys.argv) > 3 else 1
TAG = sys.argv[4] if len(sys.argv) > 4 else "control"
HERE = Path(__file__).resolve().parent
OUT = HERE / "results" / f"control_trace_{TAG}_{datetime.now():%Y%m%d_%H%M}.log"


def set_lens(name):
    (T.CUR / "lstmabs_lens.txt").write_text(name + "\n", encoding="utf-8")
    T.send("local v=be:getPlayerVehicle(0) if v then v:queueLuaCommand("
           "\"local c=controller.getController('lstm_abs') if c and c.setLens then c.setLens('" + name + "') end\") end")
    time.sleep(1.5)
    return T.veh_probe("electrics.values.lstmabs_lens or 'none'")


def record(name, mph, r, tag):
    if name.startswith("DynamicABS") or not r:
        return
    exp = HERE / "experiments" / name
    exp.mkdir(parents=True, exist_ok=True)
    f = exp / "runs.csv"
    new = not f.exists()
    with open(f, "a", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        if new:
            w.writerow(["date", "session", "mph", "avg_g", "dist_m", "runID", "status"])
        w.writerow([datetime.now().strftime("%Y-%m-%d"), tag, mph, r.get("avg_g"), r.get("dist_m"), r.get("runID"), r.get("status")])


def main():
    if not T.launch_game():
        print("ABORT: channel never came up")
        sys.exit(1)
    log_size = T.LOG.stat().st_size if T.LOG.exists() else 0
    results = []
    on_lstm = False
    for name in NAMES:
        if name.startswith("DynamicABS"):
            T.switch_config(name)
            on_lstm = False
        else:
            if not on_lstm:
                T.switch_config("LSTMABS")
                on_lstm = True
            tag = set_lens(name)
            print(f"lens tag: {tag}  ->  {'OK' if tag == name else 'MISMATCH'}")
            if tag != name:
                print("  !! lens did not load, skipping")
                continue
        cfg = T.probe("(be:getPlayerVehicle(0) and be:getPlayerVehicle(0).partConfig) or 'none'")
        print("partConfig:", cfg, " hasABS:", T.veh_probe("tostring(electrics.values.hasABS)"))
        for i in range(RUNS):
            T.wait_stopped()
            print(f"[{name}] run {i+1}/{RUNS} @ {MPH}mph ...", end="", flush=True)
            r = T.run_one(MPH)
            print(" ", r)
            results.append((name, r))
            record(name, MPH, r, TAG)
        print("engagements:", T.veh_probe("electrics.values.lstmabs_engagements or 0"))

    with open(T.LOG, "r", encoding="utf-8", errors="ignore") as fh:
        fh.seek(log_size)
        tail = fh.read()
    lines = [l for l in tail.splitlines() if "LSTM_" in l]
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"{len(lines)} trace/log lines -> {OUT}")
    for name, r in results:
        print("RESULT", (name, r))
    T.send("shutdown(0)", wait=False)


if __name__ == "__main__":
    main()
