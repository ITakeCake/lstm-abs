r"""
Harvest braking segments from the in-game 2 kHz recorder dumps
(%LOCALAPPDATA%\BeamNG\BeamNG.drive\current\raw2khz_ctrl_*.csv) into the corpus clean
schema. Those files are ~90% idle; only contiguous braking runs are kept, one parquet
per segment. The recorder rides in the etk_DSE_Dynamic_ABS part, so the controller is
DynamicABS; achieved g is measured per segment and stored so the grader can rank them.
"""
import argparse
import glob
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.expandvars(r"%LOCALAPPDATA%\BeamNG\BeamNG.drive\current")
OUT = os.path.join(HERE, "recorded_data_2khz_clean_ctrl", "TEL_dyn_ctrl")
G = 9.81
KEEP = ["timestamp", "dt", "ws_fr", "ws_fl", "ws_rr", "ws_rl", "sensor_x", "sensor_y", "sensor_z",
        "yaw_rate", "steer_input", "steer_angle", "front_wheel_angle", "rpm", "gear",
        "brake_input", "throttle_input", "brake_torque_fr", "brake_torque_fl",
        "brake_torque_rr", "brake_torque_rl", "roll_angle", "pitch_angle", "speed_velo",
        "damage", "vehicle_model", "grip_pattern", "wheelbase", "safety_variant"]
MIN_TICKS = 1500          # 0.75 s at 2 kHz
GAP_TICKS = 400           # split segments separated by more than this
PAD = 200                 # keep a little lead-in before the pedal
CHUNK = 750_000           # rows per read; larger blows memory on the multi-GB dumps


def segments(brake, speed):
    m = (brake > 0.05) & (speed > 5.0)
    idx = np.nonzero(m)[0]
    if not len(idx):
        return []
    splits = np.split(idx, np.nonzero(np.diff(idx) > GAP_TICKS)[0] + 1)
    return [s for s in splits if len(s) >= MIN_TICKS]


def harvest(path, out_dir, keep_min_g):
    base = os.path.basename(path)[:-4]
    kept = []
    carry = None
    for ci, chunk in enumerate(pd.read_csv(path, chunksize=CHUNK, usecols=lambda c: c in KEEP, low_memory=False)):
        if carry is not None:
            chunk = pd.concat([carry, chunk], ignore_index=True)
            carry = None
        sp = np.abs(chunk["speed_velo"].values)
        segs = segments(chunk["brake_input"].values, sp)
        # a segment touching the chunk end may continue; hold it back for the next chunk
        if segs and segs[-1][-1] >= len(chunk) - GAP_TICKS - 1:
            carry = chunk.iloc[max(0, segs[-1][0] - PAD):].reset_index(drop=True)
            segs = segs[:-1]
        for si, s in enumerate(segs):
            lo, hi = max(0, s[0] - PAD), s[-1] + 1
            d = chunk.iloc[lo:hi].copy()
            v = np.abs(d["speed_velo"].values)
            t = d["timestamp"].values
            dur = float(t[-1] - t[0])
            g = (v.max() - v[-1]) / max(dur, 1e-6) / G
            if g < keep_min_g:
                continue
            d = d[[c for c in KEEP if c in d.columns]].copy()
            for c in ("vehicle_model", "grip_pattern", "wheelbase", "damage"):
                if c not in d:
                    d[c] = {"vehicle_model": "etk800", "grip_pattern": "unknown",
                            "wheelbase": 2.86, "damage": 0.0}[c]
            d["safety_variant"] = "dyn_ctrl"
            d["grip_pattern"] = f"ctrl_g{g:.2f}"
            for c in d.columns:
                if d[c].dtype == np.float64 and c != "timestamp":
                    d[c] = d[c].astype(np.float32)
            os.makedirs(out_dir, exist_ok=True)
            d.to_parquet(os.path.join(out_dir, f"{base}__c{ci}s{si}.parquet"),
                         engine="pyarrow", compression="snappy", index=False)
            kept.append((len(d), round(float(v.max()), 1), round(float(g), 3)))
    return kept


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=SRC)
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--min-g", type=float, default=0.5, help="drop segments below this measured g")
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    files = sorted(glob.glob(os.path.join(a.src, "raw2khz_ctrl_*.csv")), key=os.path.getsize)[: a.limit]
    print(f"{len(files)} recorder dumps -> {a.out}")
    total = []
    for i, f in enumerate(files, 1):
        try:
            kept = harvest(f, a.out, a.min_g)
        except Exception as e:
            print(f"  [{i}/{len(files)}] {os.path.basename(f)[:44]} ERROR {str(e)[:60]}")
            continue
        total += kept
        print(f"  [{i}/{len(files)}] {os.path.basename(f)[:44]} {os.path.getsize(f)/1e9:5.1f}GB "
              f"-> {len(kept)} segments", flush=True)
    if total:
        rows = sum(t[0] for t in total)
        gs = np.array([t[2] for t in total])
        vs = np.array([t[1] for t in total])
        print(f"\n{len(total)} segments, {rows:,} rows ({rows / 2000 / 60:.1f} min of braking)")
        print(f"  measured g: min {gs.min():.2f} med {np.median(gs):.2f} max {gs.max():.2f}")
        print(f"  entry speed m/s: min {vs.min():.1f} med {np.median(vs):.1f} max {vs.max():.1f}")
        print(f"  segments above 35 m/s entry: {(vs > 35).sum()}")


if __name__ == "__main__":
    main()
