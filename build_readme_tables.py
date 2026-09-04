"""Regenerate the observation and experiment tables in README.md from code and experiment folders."""
import glob
import json
import os
import re

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
U = os.path.expandvars(r"%LOCALAPPDATA%\BeamNG\BeamNG.drive\current")

OBS = [
    ("1", "ws_fr", "m/s", "wheels.wheelRotators[2].wheelSpeed", "front-right wheel speed"),
    ("2", "ws_fl", "m/s", "wheels.wheelRotators[3].wheelSpeed", "front-left wheel speed"),
    ("3", "ws_rr", "m/s", "wheels.wheelRotators[0].wheelSpeed", "rear-right wheel speed"),
    ("4", "ws_rl", "m/s", "wheels.wheelRotators[1].wheelSpeed", "rear-left wheel speed"),
    ("5", "sensor_x", "m/s2", "ffiSensors.sensorX or gx2", "lateral acceleration"),
    ("6", "sensor_y", "m/s2", "ffiSensors.sensorY or gy2", "longitudinal acceleration (positive under braking)"),
    ("7", "sensor_z", "m/s2", "ffiSensors.sensorZ or gz2", "vertical acceleration (zeroed for telemetry-trained nets)"),
    ("8", "yaw_rate", "rad/s", "obj:getYawAngularVelocity()", "body yaw rate"),
    ("9", "steer_input", "-1..1", "electrics.values.steering_input", "driver steering input"),
    ("10", "steer_angle", "rad", "electrics.values.steering", "steering column angle"),
    ("11", "front_wheel_angle", "rad", "obj:nodeVecPlanarCosRightForward", "actual road-wheel angle from node geometry"),
    ("12", "brake_input", "0..1", "input.brake", "driver brake pedal (pinned to 1.0 while engaged)"),
    ("13", "throttle_input", "0..1", "input.throttle", "driver throttle"),
    ("14", "roll_angle", "rad", "obj:getRollPitchYaw()", "body roll (zeroed for telemetry-trained nets)"),
    ("15", "pitch_angle", "rad", "obj:getRollPitchYaw()", "body pitch (zeroed for telemetry-trained nets)"),
    ("16", "rpm", "rev/min", "electrics.values.rpm", "engine speed"),
    ("17", "gear", "index", "electrics.values.gear_A", "selected gear (zeroed for telemetry-trained nets)"),
]
GRADE_ONLY = [
    ("speed_velo / Airspeed", "m/s", "true vehicle speed; drives every decel and slip grade"),
    ("wheelbase", "m", "bicycle-model expected yaw"),
    ("damage", "-", "episode rejection"),
    ("grip_pattern", "label", "never used: measured to be fictional"),
]
OUTPUTS = [
    ("brake_torque_fr/fl/rr/rl", "0..1", "per-wheel brake torque as a fraction of that car's measured maximum"),
]


def experiments_table():
    gui = pd.read_csv(os.path.join(U, "BrakeTestResults_Straight.csv"), encoding="utf-8-sig")
    gui = gui.rename(columns={"Run ID": "runID"})[["runID", "Achieved Yaw (deg)"]]
    runs = [pd.read_csv(f).assign(experiment=os.path.basename(os.path.dirname(f)))
            for f in glob.glob(os.path.join(HERE, "experiments", "*", "runs.csv"))]
    runs = [d[d.get("status", "OK").eq("OK")] if "status" in d else d for d in runs]
    hist = pd.read_csv(os.path.join(HERE, "results", "all_runs_2026-09-02.csv"))
    hist["experiment"] = hist["config"].str.replace("LSTMABS_", "", regex=False)
    r = pd.concat(runs + [hist[["experiment", "mph", "avg_g", "dist_m", "runID"]]], ignore_index=True)
    r = r[r["mph"] == 60].merge(gui, on="runID", how="left")
    agg = r.groupby("experiment").agg(n=("avg_g", "size"), g=("avg_g", "mean"),
                                      dist=("dist_m", "mean"), yaw=("Achieved Yaw (deg)", "mean"))

    metas = {"ens_band_g_mix5": {"description": "ensemble: 5 seeds of band_g_mix averaged at runtime in Lua"},
             "baseline FORCE_CMD=1.0": {"description": "control run: output hardcoded to full torque, no model"}}
    for f in glob.glob(os.path.join(HERE, "experiments", "*", "meta.json")):
        m = json.load(open(f, encoding="utf-8"))
        metas[os.path.basename(os.path.dirname(f))] = m

    def describe(name):
        m = metas.get(name, {})
        a = m.get("args") or {}
        if not a:
            return m.get("description", "(args not recorded)")
        bits = []
        v = a.get("variants") or "all"
        if a.get("exclude_variants"):
            v += f" minus {a['exclude_variants']}"
        bits.append(v)
        if a.get("models"):
            bits.append(a["models"])
        lens = m.get("lens", "")
        if lens.startswith("band"):
            bits.append(f"bands {m.get('band_mph')}mph x{m.get('band_overlap')}")
        else:
            bits.append(f"lens {lens}")
        if m.get("lock_slip"):
            bits.append(f"lock>{m['lock_slip']} x{m.get('lock_weight')}")
        if (m.get("slip_good") or [0, 0])[1]:
            bits.append(f"slipband {tuple(m['slip_good'])} x{m.get('slip_good_weight')}")
        if m.get("yaw_dz"):
            bits.append(f"yaw dz {m['yaw_dz']}")
        if (m.get("slip_keep") or [0, 0])[1]:
            bits.append(f"slipkeep {tuple(m['slip_keep'])}")
        bits.append(f"seed {m.get('seed', 0)}")
        return ", ".join(str(b) for b in bits)

    lines = ["| experiment | training data and weighting | runs | avg g | dist (m) | yaw (deg) |",
             "|---|---|---|---|---|---|"]
    dyn = agg.loc["DynamicABS1"] if "DynamicABS1" in agg.index else None
    if dyn is not None:
        lines.append(f"| **DynamicABS (reference)** | hand-written PID, not learned | {int(dyn['n'])} | "
                     f"**{dyn['g']:.3f}** | {dyn['dist']:.1f} | {dyn['yaw']:.1f} |")
    for name, row in agg.sort_values("g", ascending=False).iterrows():
        if name == "DynamicABS1":
            continue
        yaw = "" if pd.isna(row["yaw"]) else f"{row['yaw']:.1f}"
        lines.append(f"| `{name}` | {describe(name)} | {int(row['n'])} | {row['g']:.3f} | {row['dist']:.1f} | {yaw} |")
    untested = sorted(n for n in (set(metas) - set(agg.index)) if os.path.isdir(os.path.join(HERE, "experiments", n)))
    if untested:
        lines.append("")
        lines.append("Trained but not run in the car: " + ", ".join(f"`{u}`" for u in untested) + ".")
    return "\n".join(lines)


def obs_table():
    lines = ["| # | channel | unit | source in the vehicle | what it is |", "|---|---|---|---|---|"]
    for i, name, unit, src, desc in OBS:
        lines.append(f"| {i} | `{name}` | {unit} | `{src}` | {desc} |")
    lines.append("")
    lines.append("**Outputs (4):**")
    lines.append("")
    lines.append("| channel | range | meaning |")
    lines.append("|---|---|---|")
    for name, rng, desc in OUTPUTS:
        lines.append(f"| `{name}` | {rng} | {desc} |")
    lines.append("")
    lines.append("**Grade-only columns.** Legal for computing training weights offline, never model inputs:")
    lines.append("")
    lines.append("| column | unit | used for |")
    lines.append("|---|---|---|")
    for name, unit, desc in GRADE_ONLY:
        lines.append(f"| `{name}` | {unit} | {desc} |")
    return "\n".join(lines)


def main():
    p = os.path.join(HERE, "README.md")
    s = open(p, encoding="utf-8").read()
    for marker, body in (("OBS", obs_table()), ("EXPERIMENTS", experiments_table())):
        a, b = f"<!-- {marker}:START -->", f"<!-- {marker}:END -->"
        if a not in s:
            raise SystemExit(f"marker {a} missing from README.md")
        s = re.sub(re.escape(a) + r".*?" + re.escape(b), a + "\n\n" + body + "\n\n" + b, s, flags=re.S)
    open(p, "w", encoding="utf-8").write(s)
    print("README tables regenerated")


if __name__ == "__main__":
    main()
