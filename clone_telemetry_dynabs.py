r"""
DynamicABS telemetry logs (2kHz, .drive userpath telemetry\) -> corpus clean schema.
One parquet per run under <out>/TEL_<variant>/, so train_lenses_corpus.py can read
them with --roots. Masters are only ever read.

variant = telemetry file label: dyn_abs (Dynamic_ABS), stock_tel (StockABS),
abs_1f_tel (1F), e_tel, 1fex_tel, absf_tel. Sensors are gx2/gy2/gz2 (smoothed),
roll/pitch are absent (0), gear is gearIndex: train with
--sensor-src gy2 --zero-cols sensor_z,roll_angle,pitch_angle,gear
"""
import argparse, os, re, glob
from multiprocessing import Pool
import numpy as np
import pandas as pd

TEL = os.path.expandvars(r"%LOCALAPPDATA%\BeamNG\BeamNG.drive\current\telemetry")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "recorded_data_2khz_clean_tel")
LABEL_VARIANT = {"Dynamic_ABS": "dyn_abs", "StockABS": "stock_tel", "Blake_ABS_1F": "abs_1f_tel",
                 "DynamicABS_E": "e_tel", "ABS_1FEX": "1fex_tel", "Dynamic_ABSF": "absf_tel"}
WHEELBASE = {"etk800": 2.86, "sbr": 2.50, "etkc": 2.86, "vivace": 2.60, "pickup": 3.0}
USE = ["Time", "CarSpeed", "Airspeed", "BrakeInput", "FL_Speed", "FR_Speed", "RL_Speed", "RR_Speed",
       "ThrottleInput", "SteerInput", "SteerAngle", "FrontWheelAngle", "RPM", "Gear",
       "Gx", "Gy", "Gz", "YawRate", "FL_DesiredBrake", "FR_DesiredBrake", "RL_DesiredBrake", "RR_DesiredBrake"]


def convert(args):
    src, out_root = args
    try:
        head = open(src, encoding="utf-8-sig", errors="ignore").readline()
        car = (re.search(r"Car=/?(\w+)", head) or [None, "unknown"])[1]
        codekey = (re.search(r"ABS_CodeKey=([0-9a-f-]+)", head) or [None, "none"])[1]
        runid = (re.search(r"BrakeTestRunID=([0-9-]+)", head) or [None, "none"])[1]
        label = re.sub(r".*_\d\d-\d\d-\d\d_", "", os.path.basename(src))[:-4]
        variant = LABEL_VARIANT.get(label, label.lower() + "_tel")
        df = pd.read_csv(src, skiprows=1, usecols=lambda c: c in USE, encoding="utf-8-sig", low_memory=False)
        if len(df) < 400 or "FL_DesiredBrake" not in df:
            return None
        t = df["Time"].values.astype(np.float64)
        steer_sign = np.where(df["SteerInput"].values < 0, -1.0, 1.0)
        o = pd.DataFrame({
            "timestamp": t, "dt": np.diff(t, prepend=t[0] - 0.0005),
            "ws_fr": df["FR_Speed"].abs(), "ws_fl": df["FL_Speed"].abs(),
            "ws_rr": df["RR_Speed"].abs(), "ws_rl": df["RL_Speed"].abs(),
            "sensor_x": df["Gx"], "sensor_y": df["Gy"], "sensor_z": df["Gz"], "yaw_rate": df["YawRate"],
            "steer_input": df["SteerInput"], "steer_angle": df["SteerAngle"],
            "front_wheel_angle": df["FrontWheelAngle"].abs().values * steer_sign,
            "rpm": df["RPM"], "gear": df["Gear"], "brake_input": df["BrakeInput"], "throttle_input": df["ThrottleInput"],
            "brake_torque_fr": df["FR_DesiredBrake"], "brake_torque_fl": df["FL_DesiredBrake"],
            "brake_torque_rr": df["RR_DesiredBrake"], "brake_torque_rl": df["RL_DesiredBrake"],
            "roll_angle": 0.0, "pitch_angle": 0.0, "speed_velo": df["Airspeed"].abs(), "damage": 0.0,
            "vehicle_model": car, "grip_pattern": "tel_" + codekey, "wheelbase": WHEELBASE.get(car, 2.7),
            "safety_variant": variant,
        })
        for c in o.columns:
            if o[c].dtype == np.float64:
                o[c] = o[c].astype(np.float32)
        o["timestamp"] = t.astype(np.float64)
        dst_dir = os.path.join(out_root, "TEL_" + variant)
        os.makedirs(dst_dir, exist_ok=True)
        dst = os.path.join(dst_dir, os.path.basename(src)[:-4] + f"__{runid}.parquet")
        o.to_parquet(dst, engine="pyarrow", compression="snappy", index=False)
        return variant, car, len(o)
    except Exception as e:
        return ("ERR", os.path.basename(src), str(e)[:80])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=TEL)
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    files = sorted(glob.glob(os.path.join(a.src, "DynamicABS_Run_*.csv")))[: a.limit]
    print(f"{len(files)} telemetry runs -> {a.out}")
    with Pool(a.workers) as p:
        res = p.map(convert, [(f, a.out) for f in files], chunksize=4)
    from collections import Counter
    ok = [r for r in res if r and r[0] != "ERR"]
    err = [r for r in res if r and r[0] == "ERR"]
    print("by variant/car:", Counter((r[0], r[1]) for r in ok))
    print("rows:", sum(r[2] for r in ok), " errors:", len(err), err[:5])


if __name__ == "__main__":
    main()
