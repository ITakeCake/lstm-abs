"""Build results/LSTM_ABS_RESULTS.xlsx from experiments/*/meta.json, experiments/*/runs.csv and the historical all-runs CSV."""
import glob
import json
import os

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
U = os.path.expandvars(r"%LOCALAPPDATA%\BeamNG\BeamNG.drive\current")
HIST = os.path.join(HERE, "results", "all_runs_2026-09-02.csv")


def main():
    gui = pd.read_csv(os.path.join(U, "BrakeTestResults_Straight.csv"), encoding="utf-8-sig")
    gui = gui.rename(columns={"Run ID": "runID"})[["runID", "True Path (m)", "Achieved Yaw (deg)", "Duration (s)"]]

    metas = []
    for f in sorted(glob.glob(os.path.join(HERE, "experiments", "*", "meta.json"))):
        m = json.load(open(f, encoding="utf-8"))
        a = m.get("args", {})
        metas.append({"experiment": m["name"], "lens": m.get("lens", ""), "trained": m.get("trained", ""),
                      "variants": a.get("variants", ""), "exclude": a.get("exclude_variants", ""), "models": a.get("models", ""),
                      "min_peak_g": a.get("min_peak_g", ""), "zero_cols": ",".join(m.get("zero_cols", [])),
                      "sensor_src": m.get("sensor_src", ""), "band_mph": m.get("band_mph", ""), "band_overlap": m.get("band_overlap", ""),
                      "train_episodes": m.get("train_episodes", ""), "val_mae_final": str(m.get("val_mae_final", "")),
                      "note": m.get("note", "")})
    metas = pd.DataFrame(metas)

    hist = pd.read_csv(HIST) if os.path.exists(HIST) else pd.DataFrame()
    if len(hist):
        hist["experiment"] = hist["config"].str.replace("LSTMABS_", "", regex=False)
        hist = hist[["date", "session", "experiment", "mph", "avg_g", "dist_m", "runID"]]
    new = []
    for f in sorted(glob.glob(os.path.join(HERE, "experiments", "*", "runs.csv"))):
        d = pd.read_csv(f)
        d = d[d["status"] == "OK"] if "status" in d else d
        d["experiment"] = os.path.basename(os.path.dirname(f))
        new.append(d[["date", "session", "experiment", "mph", "avg_g", "dist_m", "runID"]])
    runs = pd.concat([hist] + new, ignore_index=True) if new or len(hist) else pd.DataFrame()
    runs = runs.merge(gui, on="runID", how="left")

    summary = runs.groupby(["experiment", "mph"]).agg(
        runs=("avg_g", "size"), avg_g_mean=("avg_g", "mean"), avg_g_min=("avg_g", "min"), avg_g_max=("avg_g", "max"),
        dist_mean=("dist_m", "mean"), yaw_mean=("Achieved Yaw (deg)", "mean")).round(4).reset_index()
    summary = summary.sort_values(["mph", "avg_g_mean"], ascending=[True, False])

    out = os.path.join(HERE, "results", "LSTM_ABS_RESULTS.xlsx")
    with pd.ExcelWriter(out, engine="openpyxl") as w:
        summary.to_excel(w, sheet_name="Summary", index=False)
        runs.to_excel(w, sheet_name="Runs", index=False)
        metas.to_excel(w, sheet_name="Experiments", index=False)
        for ws in w.book.worksheets:
            for col in ws.columns:
                width = max(len(str(c.value or "")) for c in col) + 2
                ws.column_dimensions[col[0].column_letter].width = min(48, max(10, width))
    print(out, "runs:", len(runs), "experiments:", len(metas))
    print(summary.head(8).to_string(index=False))


if __name__ == "__main__":
    main()
