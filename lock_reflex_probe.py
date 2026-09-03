"""Simulated front lock: onset obs for 20 steps, then fronts at 50%/0% of car speed with sy=10. Reports front cmd."""
import os, sys, glob
import numpy as np, torch
from train_lenses_corpus import LSTMBrakeNet, INPUT_COLS, TARGET_COLS

base = np.array([float(x) for x in open("onset_obs_rows.txt").readline().split(",")])
names = sys.argv[1:] or sorted(os.path.basename(os.path.dirname(p)) for p in glob.glob("experiments/*/model.pt"))
def scenario():
    seq = []
    for t in range(80):
        o = base.copy()
        if t >= 20:
            o[5] = 10.0                       # decelerating hard
            o[:4] = o[:4] * (1 - 0.02 * (t - 20)) if t < 40 else o[:4] * 0.6   # everything slowing
        if t >= 40:
            o[0] = o[1] = base[0] * 0.6 * 0.5   # fronts at 50% of car speed (heavy slip)
        if t >= 60:
            o[0] = o[1] = 0.0                   # fronts locked
        seq.append(o)
    return np.array(seq)
S = scenario()
for name in names:
    ck = torch.load(f"experiments/{name}/model.pt", map_location="cpu", weights_only=False)
    net = LSTMBrakeNet(len(INPUT_COLS), len(TARGET_COLS)); net.load_state_dict(ck["model"]); net.eval()
    mean, std = np.asarray(ck["feat_mean"]), np.asarray(ck["feat_std"])
    X = np.clip((S - mean) / std, -10, 10).astype(np.float32)
    with torch.no_grad():
        out, _ = net.lstm(torch.from_numpy(X)[None]); y = torch.sigmoid(net.head(out))[0].numpy()
    f = y[:, :2].mean(1); r = y[:, 2:].mean(1)
    print(f"  {name:22s} front cmd: onset={f[19]:.2f} decel={f[39]:.2f} slip50%={f[45]:.2f}/{f[59]:.2f} locked={f[65]:.2f}/{f[79]:.2f}   rear: {r[19]:.2f} {r[39]:.2f} {r[59]:.2f} {r[79]:.2f}")
