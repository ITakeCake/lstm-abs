"""Replay an in-game obs sequence (LSTM_OBS lines) through a net; ablate one channel at a time to corpus mean."""
import sys, re
import numpy as np, torch
from train_lenses_corpus import LSTMBrakeNet, INPUT_COLS, TARGET_COLS
log, name = sys.argv[1], sys.argv[2]
rows = []
for l in open(log, encoding="utf-8", errors="ignore"):
    m = re.search(r"LSTM_OBS t=(\d+) ([-0-9.,e]+) ===", l)
    if m: rows.append((int(m.group(1)), [float(x) for x in m.group(2).split(",")]))
# first engagement only
seq = []; last = -1
for t, r in rows:
    if t < last: break
    seq.append(r); last = t
S = np.array(seq); S = np.repeat(S, 2, axis=0)   # every-2nd-tick capture -> approx 200Hz
ck = torch.load(f"experiments/{name}/model.pt", map_location="cpu", weights_only=False)
net = LSTMBrakeNet(17, 4); net.load_state_dict(ck["model"]); net.eval()
mean, std = np.asarray(ck["feat_mean"]), np.asarray(ck["feat_std"])
def fwd(X):
    Xn = np.clip((X - mean) / std, -10, 10).astype(np.float32)
    with torch.no_grad():
        out, _ = net.lstm(torch.from_numpy(Xn)[None]); return torch.sigmoid(net.head(out))[0].numpy()[:, :2].mean(1)
v = S[:, :4].max(1)
def summ(f): return " ".join(f"{f[i]:.2f}" for i in range(0, len(f), max(1, len(f)//8)))
print(f"{len(S)} steps; speed samples: {summ(v)}")
print(f"{'replay (true obs)':22s} front cmd: {summ(fwd(S))}")
for ci, c in enumerate(INPUT_COLS):
    X = S.copy(); X[:, ci] = mean[ci]
    print(f"{'hold '+c+' at mean':22s} front cmd: {summ(fwd(X))}")
