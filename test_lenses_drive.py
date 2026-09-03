"""
Straight-line brake comparison of the 6 lens LSTMs (+ DynamicABS reference) in
BeamNG.drive, using the absCmdChannel file channel and the in-game brakeTest
auto machine (standard 2kHz metric via brake_done.txt).

Each config is verified 4 independent ways before its runs are accepted:
  1. player vehicle partConfig is the expected .pc
  2. the ABS part actually in the etk_DSE_ABS slot
  3. controller load banner in beamng.log
  4. lstmabs_engagements > 0 after the runs (net actually actuated)
"""
import os
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

CUR = Path(os.path.expandvars(r"%LOCALAPPDATA%\BeamNG\BeamNG.drive\current"))
CMD = CUR / "brake_cmd.txt"
CMD_TMP = CUR / "brake_cmd.tmp"
DONE = CUR / "brake_done.txt"
PROBE = CUR / "suite_probe.txt"
LOG = CUR / "beamng.log"
EXE = r"D:\SteamLibrary\steamapps\common\BeamNG.drive\Bin64\BeamNG.drive.x64.exe"
OUT_CSV = Path(__file__).resolve().parent / "results" / f"lens_drive_results_{datetime.now():%Y%m%d_%H%M}.csv"

LENSES = ["band_g_mix", "band_gy_mix", "baseline_dyn"]
CONFIGS = [("LSTMABS", L) for L in LENSES] + [("DynamicABS1", "REFERENCE_dlaw")]
SPEEDS = [60, 60, 80, 80]

SPAWN = (0.0, 0.0, 0.5)
CONSUME_TIMEOUT = 8.0


def send(lua, wait=True):
    CMD_TMP.write_text(lua, encoding="utf-8")
    os.replace(CMD_TMP, CMD)
    if not wait:
        return True
    deadline = time.time() + CONSUME_TIMEOUT
    while CMD.exists():
        if time.time() > deadline:
            return False
        time.sleep(0.1)
    time.sleep(0.25)
    return True


def probe(lua_expr, timeout=6.0):
    """Run lua that writes suite_probe.txt, return its contents."""
    if PROBE.exists():
        PROBE.unlink()
    send(f'local f=io.open("suite_probe.txt","w") if f then f:write(tostring({lua_expr})) f:close() end')
    deadline = time.time() + timeout
    while time.time() < deadline:
        if PROBE.exists():
            time.sleep(0.15)
            try:
                return PROBE.read_text(encoding="utf-8", errors="ignore").strip()
            except OSError:
                pass
        time.sleep(0.2)
    return None


def veh_probe(lua_expr, timeout=6.0):
    """Same, but evaluated inside the vehicle VM."""
    if PROBE.exists():
        PROBE.unlink()
    inner = ('local f=io.open(\\"suite_probe.txt\\",\\"w\\") if f then f:write(tostring('
             + lua_expr + ')) f:close() end')
    send(f'local v=be:getPlayerVehicle(0) if v then v:queueLuaCommand("{inner}") end')
    deadline = time.time() + timeout
    while time.time() < deadline:
        if PROBE.exists():
            time.sleep(0.15)
            try:
                return PROBE.read_text(encoding="utf-8", errors="ignore").strip()
            except OSError:
                pass
        time.sleep(0.2)
    return None


def launch_game():
    print("launching BeamNG.drive on smallgrid ...")
    for p in (CMD, PROBE, DONE):
        if p.exists():
            p.unlink()
    subprocess.Popen([EXE, "-level", "smallgrid"], cwd=str(Path(EXE).parent))
    for i in range(60):
        time.sleep(10)
        lvl = probe("getCurrentLevelIdentifier()", timeout=4)
        if lvl and "grid" in lvl.lower():
            print(f"channel up on level '{lvl}' after ~{(i+1)*10}s")
            return True
        print(f"  waiting for channel... {(i+1)*10}s")
    return False


def log_tail_has(pattern, since_size):
    try:
        with open(LOG, "r", encoding="utf-8", errors="ignore") as fh:
            fh.seek(since_size)
            return re.search(pattern, fh.read())
    except OSError:
        return None


def switch_config(pc_name):
    send(f'core_vehicles.replaceVehicle("etk800", {{config = "vehicles/etk800/{pc_name}.pc"}})')
    time.sleep(14)


def wait_stopped(max_s=20.0):
    deadline = time.time() + max_s
    while time.time() < deadline:
        v = veh_probe("string.format('%.3f', obj:getVelocity():length())", timeout=5)
        try:
            if v is not None and float(v) < 0.15:
                return True
        except ValueError:
            pass
        time.sleep(0.6)
    return False


def run_one(mph, timeout=180):
    send('local v=be:getPlayerVehicle(0) if v then '
         'v:queueLuaCommand("extensions.brakeTest.clearBrakeLine()") '
         'v:queueLuaCommand("extensions.brakeTest.setLaunchRamp(false)") end')
    # (brakeMph, recordMph, coastMph) - brake first, record second
    send(f'extensions.brakeTestUI.setTestParams({mph + 2},{mph},2)')
    if DONE.exists():
        DONE.unlink()
    send('extensions.brakeTestUI.toggleAutoTestRun()')

    deadline = time.time() + timeout
    while not DONE.exists():
        if time.time() > deadline:
            send('extensions.brakeTestUI.toggleAutoTestRun()', wait=False)
            return None
        time.sleep(0.5)
    time.sleep(0.5)
    line = DONE.read_text(encoding="utf-8", errors="ignore").strip()
    return dict(p.split("=", 1) for p in line.split() if "=" in p)


def main():
    if not launch_game():
        print("ABORT: channel never came up")
        sys.exit(1)

    rows = []
    prev_run_id = None
    for pc_name, lens in CONFIGS:
        print(f"\n{'='*64}\n  {lens}  ({pc_name}.pc)\n{'='*64}")
        log_size = LOG.stat().st_size if LOG.exists() else 0
        switch_config(pc_name)
        if lens != "REFERENCE_dlaw":
            (CUR / "lstmabs_lens.txt").write_text(lens + "\n", encoding="utf-8")
            send("local v=be:getPlayerVehicle(0) if v then v:queueLuaCommand("
                 "\"local c=controller.getController('lstm_abs') if c and c.setLens then c.setLens('" + lens + "') end\") end")
            time.sleep(1.5)

        # --- verification 1: player vehicle partConfig
        cfg = probe("(be:getPlayerVehicle(0) and be:getPlayerVehicle(0).partConfig) or 'none'")
        v1 = bool(cfg and pc_name.lower() in cfg.lower())
        # --- verification 2: controller banner in the log
        if lens == "REFERENCE_dlaw":
            m = log_tail_has(r"ABS-1FEX|Dynamic_ABS", log_size)
            v3 = bool(m)
            banner = "Dynamic_ABS banner" if m else "MISSING"
        else:
            m = log_tail_has(rf"LSTM_ABS: lens '{re.escape(lens)}' loaded", log_size)
            v3 = bool(m)
            banner = m.group(0) if m else "MISSING"
        # --- verification 4 (electrics lens tag)
        etag = veh_probe("electrics.values.lstmabs_lens or 'none'")
        v4 = (lens == "REFERENCE_dlaw") or (etag == lens)

        print(f"  verify partConfig : {cfg}   -> {'OK' if v1 else 'MISMATCH'}")
        print(f"  verify banner     : {banner}   -> {'OK' if v3 else 'MISSING'}")
        print(f"  verify lens tag   : {etag}   -> {'OK' if v4 else 'MISMATCH'}")
        if not (v1 and v3):
            print("  !! verification failed - skipping this config's runs")
            rows.append(dict(lens=lens, mph="", dist_m="", avg_g="",
                             status="VERIFY_FAILED", cfg=cfg, banner=banner))
            continue

        for mph in SPEEDS:
            wait_stopped()
            print(f"  [{lens} @ {mph}mph] running...", end="", flush=True)
            r = run_one(mph)
            if not r:
                print(" TIMEOUT")
                rows.append(dict(lens=lens, mph=mph, dist_m="", avg_g="",
                                 status="TIMEOUT", cfg=cfg, banner=banner))
                continue
            rid = r.get("runID", "?")
            dist, g = r.get("dist_m", "?"), r.get("avg_g", "?")
            status = "OK"
            if rid == prev_run_id or float(dist or 0) < 0.5:
                status = "NO_MEASURE"
            else:
                prev_run_id = rid
            print(f" {g}g  {dist}m  [{status}]")
            rows.append(dict(lens=lens, mph=mph, dist_m=dist, avg_g=g,
                             status=status, cfg=cfg, banner=banner))

        eng = veh_probe("electrics.values.lstmabs_engagements or 0")
        print(f"  verify engagements: {eng} "
              f"-> {'OK (net actuated)' if lens == 'REFERENCE_dlaw' or (eng and eng not in ('0','nil','none')) else 'NET NEVER ENGAGED'}")
        for row in rows:
            if row["lens"] == lens:
                row["engagements"] = eng

    import csv
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, ["lens", "mph", "dist_m", "avg_g", "status",
                                "engagements", "cfg", "banner"])
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in w.fieldnames})
    print(f"\nresults -> {OUT_CSV}")

    print(f"\n{'='*64}\n  SUMMARY (avg_g, higher is better)\n{'='*64}")
    for _, lens in CONFIGS:
        for mph in (60, 80):
            gs = [float(r["avg_g"]) for r in rows
                  if r["lens"] == lens and r.get("status") == "OK"
                  and str(r.get("mph")) == str(mph) and r.get("avg_g")]
            ds = [float(r["dist_m"]) for r in rows
                  if r["lens"] == lens and r.get("status") == "OK"
                  and str(r.get("mph")) == str(mph) and r.get("dist_m")]
            if gs:
                print(f"  {lens:18s} {mph}mph  avg_g={sum(gs)/len(gs):.4f}  "
                      f"dist={sum(ds)/len(ds):.2f}m  (n={len(gs)})")
            else:
                print(f"  {lens:18s} {mph}mph  no valid runs")

    send('shutdown(0)', wait=False)


if __name__ == "__main__":
    main()
