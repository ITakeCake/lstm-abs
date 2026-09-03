"""
Clones the full 99-col 2kHz corpus (S1-S5 + SAC) into a training-clean parquet
copy, keeping the realistic-obs whitelist, the torque targets, and the
grading-only columns. Masters are read-only - never modified.

Column groups:
  INPUT_COLS  - legal model inputs (whitelist agreed with Blake)
  TARGET_COLS - per-wheel brake torque (the action being learned)
  GRADE_COLS  - cheat columns, legal for offline grading only, never as input
  META_COLS   - provenance / filtering
"""
import glob
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
SRC_ROOT = SCRIPT_DIR / "recorded_data_2khz"
DST_ROOT = SCRIPT_DIR / "recorded_data_2khz_clean"
SUBSETS = ["S1", "S2", "S3", "S4", "S5", "SAC"]  # DriverBoy excluded: 100Hz, don't mix

INPUT_COLS = [
    "ws_fr", "ws_fl", "ws_rr", "ws_rl",
    "sensor_x", "sensor_y", "sensor_z",
    "yaw_rate",
    "steer_input", "steer_angle", "front_wheel_angle",
    "brake_input", "throttle_input",
    "roll_angle", "pitch_angle",
    "rpm", "gear",
]
TARGET_COLS = ["brake_torque_fr", "brake_torque_fl", "brake_torque_rr", "brake_torque_rl"]
GRADE_COLS = ["speed_velo", "wheelbase", "damage"]  # offline grading only, NOT inputs
META_COLS = ["timestamp", "dt", "vehicle_model", "safety_variant", "grip_pattern"]

KEEP_COLS = META_COLS + INPUT_COLS + TARGET_COLS + GRADE_COLS
STR_COLS = {"vehicle_model", "safety_variant", "grip_pattern"}


def convert(args):
    src, dst = args
    try:
        if os.path.exists(dst):
            return ("skip", src, 0)
        df = pd.read_csv(src, usecols=KEEP_COLS)
        for c in df.columns:
            if c not in STR_COLS:
                df[c] = pd.to_numeric(df[c], errors="coerce").astype("float32")
        for c in STR_COLS:
            df[c] = df[c].astype("category")
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        df.to_parquet(dst, engine="pyarrow", compression="snappy", index=False)
        return ("ok", src, len(df))
    except Exception as ex:
        return (f"FAIL: {ex}", src, 0)


def main():
    jobs = []
    for sub in SUBSETS:
        for src in sorted(glob.glob(str(SRC_ROOT / sub / "*.csv"))):
            if "_meta" in os.path.basename(src):
                continue
            name = Path(src).stem + ".parquet"
            jobs.append((src, str(DST_ROOT / sub / name)))

    print(f"{len(jobs)} files to convert -> {DST_ROOT}")
    print(f"keeping {len(KEEP_COLS)} cols ({len(INPUT_COLS)} inputs + {len(TARGET_COLS)} "
          f"targets + {len(GRADE_COLS)} grade-only + {len(META_COLS)} meta)\n")

    ok = skipped = failed = 0
    rows = 0
    with ProcessPoolExecutor(max_workers=12) as ex:
        futs = [ex.submit(convert, j) for j in jobs]
        for i, fut in enumerate(as_completed(futs), 1):
            status, src, n = fut.result()
            if status == "ok":
                ok += 1
                rows += n
            elif status == "skip":
                skipped += 1
            else:
                failed += 1
                print(f"  {status}  {os.path.basename(src)}")
            if i % 200 == 0:
                print(f"  ...{i}/{len(jobs)}  ok={ok} skip={skipped} fail={failed} "
                      f"rows={rows:,}", flush=True)

    print(f"\nDone: {ok} converted, {skipped} skipped, {failed} failed, {rows:,} rows total")


if __name__ == "__main__":
    main()
