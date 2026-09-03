"""Export the named lenses into the mod, then copy mod/lstm_abs and configs/*.pc into the game userpath."""
import argparse
import glob
import os
import shutil
import subprocess
import sys

REPO = os.path.dirname(os.path.abspath(__file__))
DRIVE = os.path.expandvars(r"%LOCALAPPDATA%\BeamNG\BeamNG.drive\current")
MOD_SRC = os.path.join(REPO, "mod", "lstm_abs")
MOD_DST = os.path.join(DRIVE, "mods", "Unpacked", "lstm_abs")
PC_DST = os.path.join(DRIVE, "vehicles", "etk800")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lenses", default="", help="comma list of experiments to export first")
    ap.add_argument("--set-lens", default="", help="write lstmabs_lens.txt so the game boots with this lens")
    ap.add_argument("--ensemble", default="", help="name=a,b,c passed through to the exporter")
    a = ap.parse_args()
    if a.lenses or a.ensemble:
        cmd = [sys.executable, os.path.join(REPO, "export_lstm_abs.py"), "--lens", a.lenses or ""]
        if a.ensemble:
            cmd += ["--ensemble", a.ensemble]
        subprocess.check_call(cmd)
    n = 0
    for root, _, files in os.walk(MOD_SRC):
        for f in files:
            if ".BAK" in f:
                continue
            src = os.path.join(root, f)
            dst = os.path.join(MOD_DST, os.path.relpath(src, MOD_SRC))
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copy2(src, dst)
            n += 1
    os.makedirs(PC_DST, exist_ok=True)
    for src in glob.glob(os.path.join(REPO, "configs", "*.pc")):
        shutil.copy2(src, os.path.join(PC_DST, os.path.basename(src)))
        n += 1
    if a.set_lens:
        open(os.path.join(DRIVE, "lstmabs_lens.txt"), "w", encoding="utf-8").write(a.set_lens + "\n")
    print(f"deployed {n} files -> {MOD_DST}")


if __name__ == "__main__":
    main()
