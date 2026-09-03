"""Hold one in-game brake-onset observation for N steps and read each lens net's output."""
import os, sys, glob
import numpy as np, torch
from train_lenses_corpus import LSTMBrakeNet, INPUT_COLS, TARGET_COLS

rows = [np.array([float(x) for x in l.split(",")]) for l in open("onset_obs_rows.txt") if l.strip()]
names = sys.argv[1:] or sorted(os.path.basename(os.path.dirname(p)) for p in glob.glob("experiments/*/model.pt"))
print("obs rows:", len(rows), " nets:", names)
for name in names:
    ck = torch.load(f"experiments/{name}/model.pt", map_location="cpu", weights_only=False)
    net = LSTMBrakeNet(len(INPUT_COLS), len(TARGET_COLS)); net.load_state_dict(ck["model"]); net.eval()
    mean, std = np.asarray(ck["feat_mean"]), np.asarray(ck["feat_std"])
    outs = []
    for r in rows:
        o = r.copy()
        for c in ck.get("zero_cols") or []: o[INPUT_COLS.index(c)] = 0
        x = np.clip((o - mean) / std, -10, 10).astype(np.float32)
        seq = torch.from_numpy(np.repeat(x[None], 60, 0))[None]
        with torch.no_grad():
            out, _ = net.lstm(seq)
            y = torch.sigmoid(net.head(out))[0]   # (60, 4) per-step outputs
        outs.append((y[0].numpy(), y[19].numpy(), y[59].numpy()))
    for i, (a, b, c) in enumerate(outs):
        print(f"  {name:22s} obs{i}: t0={np.round(a,3)} t20={np.round(b,3)} t60={np.round(c,3)}")
