"""Copy the live controller/jbeam/.pc back from the BeamNG.drive userpath into the repo (skips .BAK files)."""
import glob
import os
import shutil

REPO = os.path.dirname(os.path.abspath(__file__))
DRIVE = os.path.expandvars(r"%LOCALAPPDATA%\BeamNG\BeamNG.drive\current")
MOD_SRC = os.path.join(DRIVE, "mods", "Unpacked", "lstm_abs")
MOD_DST = os.path.join(REPO, "mod", "lstm_abs")


def main():
    n = 0
    for root, _, files in os.walk(MOD_SRC):
        for f in files:
            if ".BAK" in f:
                continue
            src = os.path.join(root, f)
            dst = os.path.join(MOD_DST, os.path.relpath(src, MOD_SRC))
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copy2(src, dst); n += 1
    for src in glob.glob(os.path.join(DRIVE, "vehicles", "etk800", "LSTMABS_*.pc")):
        shutil.copy2(src, os.path.join(REPO, "configs", os.path.basename(src))); n += 1
    print(f"pulled {n} files into {MOD_DST}")


if __name__ == "__main__":
    main()
