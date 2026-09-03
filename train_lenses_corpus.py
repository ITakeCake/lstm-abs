"""
Reward-weighted offline learning of an ABS from the 25GB logged corpus.
No teacher / no expert demonstrations: every logged action is weighted by how
good its measured outcome was, ranked WITHIN its own surface so low-mu
episodes are not silently deleted.

Outcome for a tick = what happened in the ~200ms AFTER it:
  - decel achieved (g)
  - whether yaw matched what the steering angle was asking for (deadzoned)

Eight lenses, identical architecture, differing only in how ticks are weighted:
  baseline      uniform (control)
  avg_g_window  forward-window mean decel, surface-ranked      [required]
  peak_g        forward-window peak decel, surface-ranked
  yaw_intent    yaw-vs-steering-intent match, surface-ranked
  combined      avg_g_window * yaw_intent
  curation      top-quartile episodes only, uniform weight (data selection)
  band_g        speed-band decel from the speed trace, percentile-ranked per model
  band_gy       band_g * band yaw-match, percentile-ranked per model

Targets are normalized per vehicle model to a 0-1 fraction of that car's max
observed brake torque, so 4 different cars can share one model.
"""
import argparse
from datetime import datetime
import json
import glob
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

SCRIPT_DIR = Path(__file__).resolve().parent
CLEAN_ROOT = SCRIPT_DIR / "recorded_data_2khz_clean"
CACHE_PATH = SCRIPT_DIR / "recorded_data_2khz_clean" / "_cache_corpus.npz"
CLEAN_ROOTS = [CLEAN_ROOT]
SENSOR_SRC = "ffi"
RUN_ARGS = {}
SEED = 0
LOCK_SLIP = 0.0        # 0 disables; ticks with any wheel slip above this get LOCK_WEIGHT
LOCK_SPEED_MPH = 5.0
LOCK_WEIGHT = 0.1
SLIP_GOOD = (0.0, 0.0)  # (lo, hi) slip band that gets SLIP_GOOD_WEIGHT above LOCK_SPEED_MPH; (0,0) disables
SLIP_GOOD_WEIGHT = 1.5
YAW_DZ = 0.0            # rad/s; 0 disables yaw-match weighting
YAW_GOOD = 1.5
YAW_BAD = 0.5
WHEELBASE_BY_MODEL = {"etk800": 2.86, "etkc": 2.86, "sbr": 2.50, "vivace": 2.60}
EXP_DIR = SCRIPT_DIR / "experiments"
TAG = ""

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

DOWNSAMPLE = 10          # 2kHz -> 200Hz
SEQ_LEN = 30             # ~150ms causal context at 200Hz
OUTCOME_WINDOW = 40      # ~200ms forward-looking outcome window at 200Hz
BRAKE_THRESH = 0.05
DAMAGE_MAX = 500
YAW_DEADZONE = 0.05      # rad/s - inside this, yaw counts as "as expected"
MAX_SEQ_PER_LENS = 400_000
G = 9.81
ZERO_COLS = []          # channel names forced to 0 in training AND deployment
MPH_TO_MPS = 0.44704
BAND_MPH = 5.0          # speed-band width in mph, for band_g/band_gy lenses
BAND_OVERLAP = 1        # windows covering each speed; 1=no overlap, 2=half-step


# ---------------------------------------------------------------- cache build

def _episode_arrays(fp):
    df = pd.read_parquet(fp)
    if len(df) < (SEQ_LEN + OUTCOME_WINDOW) * DOWNSAMPLE:
        return None
    df = df.iloc[::DOWNSAMPLE].reset_index(drop=True)
    if df["damage"].max() > DAMAGE_MAX:
        return None

    braking = df["brake_input"].values > BRAKE_THRESH
    if braking.sum() < SEQ_LEN + OUTCOME_WINDOW:
        return None

    # decel from the speed derivative, not sensor_y: the IMU axis sign is not a
    # reliable "decelerating" indicator, and reading it backwards silently zeroes
    # every g-based lens. speed_velo is grade-only, never a model input.
    sp = np.abs(df["speed_velo"].values)
    ts = df["timestamp"].values
    dsp = np.diff(sp, prepend=sp[0])
    dts = np.diff(ts, prepend=ts[0] - (ts[1] - ts[0] if len(ts) > 1 else 5e-3))
    dts = np.where(dts <= 1e-9, 1e-9, dts)
    decel_g = np.clip(-dsp / dts, 0, None) / G
    decel_g = pd.Series(decel_g).rolling(5, center=True, min_periods=1).mean().values

    # yaw the steering angle was asking for (bicycle model), vs what happened
    v = np.abs(df["speed_velo"].values)
    wb = float(np.nanmedian(df["wheelbase"].values)) or 2.7
    expected_yaw = v * np.tan(df["front_wheel_angle"].values) / max(wb, 0.5)
    yaw_err = np.abs(df["yaw_rate"].values - expected_yaw)
    yaw_score = 1.0 / (1.0 + np.clip(yaw_err - YAW_DEADZONE, 0, None) * 10.0)

    # forward-looking outcome windows
    n = len(df)
    kern = np.ones(OUTCOME_WINDOW, dtype=np.float64) / OUTCOME_WINDOW
    pad = np.concatenate([decel_g, np.full(OUTCOME_WINDOW, decel_g[-1])])
    fwd_avg_g = np.convolve(pad, kern, mode="valid")[:n]
    fwd_peak_g = pd.Series(decel_g[::-1]).rolling(OUTCOME_WINDOW, min_periods=1) \
                   .max()[::-1].values[:n]
    pad_y = np.concatenate([yaw_score, np.full(OUTCOME_WINDOW, yaw_score[-1])])
    fwd_yaw = np.convolve(pad_y, kern, mode="valid")[:n]

    meta = df.iloc[0]
    Xa = df[INPUT_COLS].values.astype(np.float32)
    for c in ZERO_COLS:
        Xa[:, INPUT_COLS.index(c)] = 0.0
    return {
        "X": Xa,
        "Yraw": df[TARGET_COLS].values.astype(np.float32),
        "brake": braking,
        "speed": sp.astype(np.float32),
        "avg_g": fwd_avg_g.astype(np.float32),
        "peak_g": fwd_peak_g.astype(np.float32),
        "yaw": fwd_yaw.astype(np.float32),
        "model": str(meta["vehicle_model"]),
        "grip": str(meta["grip_pattern"]),
        "variant": str(meta["safety_variant"]),
        "file": os.path.basename(fp),
    }


def build_cache(limit_files=None):
    files = sorted(f for r in CLEAN_ROOTS for f in glob.glob(str(r / "*" / "*.parquet")))
    if limit_files:
        files = files[:limit_files]
    print(f"Building cache from {len(files)} parquet episodes...")

    eps = []
    for i, fp in enumerate(files, 1):
        try:
            e = _episode_arrays(fp)
        except Exception:
            e = None
        if e is not None:
            eps.append(e)
        if i % 250 == 0:
            print(f"  ...{i}/{len(files)}  kept={len(eps)}", flush=True)
    print(f"{len(eps)} usable episodes")

    # per-car max torque (empirical full authority) -> 0-1 normalization
    car_max = {}
    for e in eps:
        m = e["Yraw"].max(axis=0)
        car_max[e["model"]] = np.maximum(car_max.get(e["model"], np.zeros(4)), m)
    print("per-car max torque (Nm):")
    for k, v in car_max.items():
        print(f"   {k:10s} {np.round(v, 1)}")

    for e in eps:
        e["Y"] = (e["Yraw"] / np.maximum(car_max[e["model"]], 1.0)).astype(np.float32)
        del e["Yraw"]

    np.savez(
        CACHE_PATH,
        payload=np.array([eps], dtype=object),
        car_max=np.array([car_max], dtype=object),
        allow_pickle=True,
    )
    print(f"cached -> {CACHE_PATH}")
    return eps, car_max


def load_cache():
    # allow_pickle is safe here: the cache is written by build_cache() on this
    # machine from local parquet, never fetched or shared.
    d = np.load(CACHE_PATH, allow_pickle=True)
    eps = list(d["payload"][0])
    if eps and any("speed" not in e for e in eps):
        raise RuntimeError(
            "cached episodes lack 'speed' (needed by band_g/band_gy) - rerun with --rebuild-cache")
    return eps, d["car_max"][0]


# ------------------------------------------------------------ surface ranking

def surface_ranked(eps, key):
    """Percentile-rank each tick's outcome WITHIN ITS OWN EPISODE.

    Deliberately not grouped by grip_pattern: those labels were measured to be
    unreliable (black_ice episodes averaging 0.9g), so grouping by them ranks
    against a group that does not mean what it says. One episode is one
    car/surface/speed, so within-episode ranking asks "was this action good for
    this situation" without trusting any label.
    """
    ranks = []
    for e in eps:
        vals = e[key]
        pool = vals[e["brake"]]
        if len(pool) < 8:
            ranks.append(np.ones(len(vals), dtype=np.float32))
            continue
        qs = np.quantile(pool, np.linspace(0, 1, 101))
        r = np.searchsorted(qs, vals, side="right") / 100.0
        ranks.append(np.clip(r, 0.0, 1.0).astype(np.float32))
    return ranks


def band_edges_mph(eps, band_mph, band_overlap):
    """Speed-band [lo, lo+band_mph) edges in mph, step = band_mph/band_overlap."""
    step_mph = band_mph / band_overlap
    max_mph = 0.0
    for e in eps:
        sp = e["speed"][e["brake"]]
        if len(sp):
            max_mph = max(max_mph, float(sp.max()) / MPH_TO_MPS)
    n_bands = int(np.floor(max_mph / step_mph)) + 1
    los = np.arange(n_bands) * step_mph
    return np.stack([los, los + band_mph], axis=1)


def band_grades(eps, band_mph=None, band_overlap=None):
    """Per-episode, per-band decel and yaw grades over speed windows."""
    band_mph = BAND_MPH if band_mph is None else band_mph
    band_overlap = BAND_OVERLAP if band_overlap is None else band_overlap
    edges = band_edges_mph(eps, band_mph, band_overlap)
    dt = DOWNSAMPLE * 0.0005
    tol = 0.5 * MPH_TO_MPS
    g_out, y_out = [], []
    for e in eps:
        speed, brake, yaw = e["speed"], e["brake"], e["yaw"]
        g_e = np.full(len(edges), np.nan, dtype=np.float32)
        y_e = np.full(len(edges), np.nan, dtype=np.float32)
        for b, (lo_mph, hi_mph) in enumerate(edges):
            lo, hi = lo_mph * MPH_TO_MPS, hi_mph * MPH_TO_MPS
            idx = np.nonzero(brake & (speed >= lo) & (speed < hi))[0]
            if len(idx) < 4:
                continue
            span = np.arange(idx[0], idx[-1] + 1)
            if len(idx) < 0.9 * len(span) or not brake[span].all():
                continue
            v_entry, v_exit = speed[idx[0]], speed[idx[-1]]
            if v_entry < hi - tol or v_exit > lo + tol:
                continue
            n = len(span)
            g = (v_entry - v_exit) / (dt * (n - 1)) / G
            g_e[b] = max(float(g), 0.0)
            y_e[b] = float(yaw[span].mean())
        g_out.append(g_e)
        y_out.append(y_e)
    return edges, g_out, y_out


def band_percentile_ranks(eps, scores):
    """Per-band percentile rank of each episode's score within its own model."""
    n_bands = len(scores[0])
    ranks = [np.full(n_bands, np.nan, dtype=np.float32) for _ in eps]
    for b in range(n_bands):
        by_model = {}
        for i, e in enumerate(eps):
            v = scores[i][b]
            if not np.isnan(v):
                by_model.setdefault(e["model"], []).append(i)
        for idxs in by_model.values():
            vals = np.array([scores[i][b] for i in idxs])
            if len(idxs) == 1:
                ranks[idxs[0]][b] = 1.0
                continue
            order = np.argsort(vals)
            pct = np.empty(len(vals), dtype=np.float32)
            pct[order] = np.arange(len(vals), dtype=np.float32) / (len(vals) - 1)
            for pos, i in enumerate(idxs):
                ranks[i][b] = pct[pos]
    return ranks


def band_lens_weights(eps, band_mph, band_overlap, use_yaw):
    """Per-tick weight = mean band-rank of the bands covering that tick's speed."""
    edges, g_band, y_band = band_grades(eps, band_mph, band_overlap)
    scores = [g * y for g, y in zip(g_band, y_band)] if use_yaw else g_band
    ranks = band_percentile_ranks(eps, scores)
    lo = edges[:, 0] * MPH_TO_MPS
    hi = edges[:, 1] * MPH_TO_MPS
    out = []
    for e, rk in zip(eps, ranks):
        speed, brake = e["speed"], e["brake"]
        w = np.full(len(speed), 0.1, dtype=np.float32)
        bt = np.nonzero(brake)[0]
        sp = speed[bt]
        rsum = np.zeros(len(bt), dtype=np.float64)
        rcnt = np.zeros(len(bt), dtype=np.int32)
        for b in range(len(lo)):
            if np.isnan(rk[b]):
                continue
            m = (sp >= lo[b]) & (sp < hi[b])
            rsum[m] += rk[b]
            rcnt[m] += 1
        mean_rank = np.where(rcnt > 0, rsum / np.maximum(rcnt, 1), 0.5)
        w[bt] = (0.1 + 0.9 * np.clip(mean_rank, 0, 1)).astype(np.float32)
        out.append(w)
    return out


def lock_multiplier(eps):
    """Per-tick multiplier: LOCK_WEIGHT where a wheel exceeds LOCK_SLIP above LOCK_SPEED_MPH, else 1."""
    out = []
    v_min = LOCK_SPEED_MPH * MPH_TO_MPS
    for e in eps:
        speed = e["speed"]
        ws = e["X"][:, :4]
        slip = 1.0 - ws.min(axis=1) / np.maximum(speed, 0.5)
        locked = (slip > LOCK_SLIP) & (speed > v_min)
        out.append(np.where(locked, LOCK_WEIGHT, 1.0).astype(np.float32))
    return out


def slip_good_multiplier(eps):
    """1 + (SLIP_GOOD_WEIGHT-1) * fraction of wheels inside the good slip band, above LOCK_SPEED_MPH."""
    out = []
    lo, hi = SLIP_GOOD
    v_min = LOCK_SPEED_MPH * MPH_TO_MPS
    for e in eps:
        speed = e["speed"]
        slip = 1.0 - e["X"][:, :4] / np.maximum(speed, 0.5)[:, None]
        frac = ((slip >= lo) & (slip <= hi)).mean(axis=1)
        m = 1.0 + (SLIP_GOOD_WEIGHT - 1.0) * frac
        out.append(np.where(speed > v_min, m, 1.0).astype(np.float32))
    return out


def yaw_match_multiplier(eps):
    """YAW_GOOD inside the deadzone, YAW_BAD beyond twice it, linear between."""
    out = []
    for e in eps:
        X = e["X"]
        v = e["speed"]
        wb = WHEELBASE_BY_MODEL.get(e["model"], 2.7)
        expected = v * np.tan(X[:, 10]) / wb
        err = np.abs(X[:, 7] - expected)
        t = np.clip((err - YAW_DZ) / YAW_DZ, 0.0, 1.0)
        out.append((YAW_GOOD + (YAW_BAD - YAW_GOOD) * t).astype(np.float32))
    return out


def apply_lock_weight(eps, weights):
    if SLIP_GOOD[1] > 0:
        mult = slip_good_multiplier(eps)
        n_all = sum(int(e["brake"].sum()) for e in eps)
        n_hit = sum(int(((m > 1.0) & e["brake"]).sum()) for m, e in zip(mult, eps))
        print(f"  slip-good band {SLIP_GOOD} x{SLIP_GOOD_WEIGHT}: touches {n_hit:,}/{n_all:,} braking ticks ({100.0 * n_hit / max(n_all, 1):.1f}%)")
        weights = [w * m for w, m in zip(weights, mult)]
    if YAW_DZ > 0:
        mult = yaw_match_multiplier(eps)
        n_all = sum(int(e["brake"].sum()) for e in eps)
        n_good = sum(int(((m >= YAW_GOOD - 1e-6) & e["brake"]).sum()) for m, e in zip(mult, eps))
        print(f"  yaw-match dz={YAW_DZ} rad/s x{YAW_GOOD}/x{YAW_BAD}: {100.0 * n_good / max(n_all, 1):.1f}% of braking ticks inside deadzone")
        weights = [w * m for w, m in zip(weights, mult)]
    if LOCK_SLIP <= 0:
        return [np.maximum(w, 0.01).astype(np.float32) for w in weights]
    mult = lock_multiplier(eps)
    n_all = sum(int(e["brake"].sum()) for e in eps)
    n_hit = sum(int(((m < 1.0) & e["brake"]).sum()) for m, e in zip(mult, eps))
    print(f"  lock down-weight: slip>{LOCK_SLIP} above {LOCK_SPEED_MPH}mph -> x{LOCK_WEIGHT} on "
          f"{n_hit:,}/{n_all:,} braking ticks ({100.0 * n_hit / max(n_all, 1):.1f}%)")
    return [np.maximum(w * m, 0.01).astype(np.float32) for w, m in zip(weights, mult)]


def lens_weights(eps, lens):
    """Per-tick training weight in [0.1, 1.0]; never 0 so nothing is deleted."""
    return apply_lock_weight(eps, base_lens_weights(eps, lens))


def base_lens_weights(eps, lens):
    if lens == "baseline" or lens == "curation":
        return [np.ones(len(e["avg_g"]), dtype=np.float32) for e in eps]
    if lens == "band_g":
        return band_lens_weights(eps, BAND_MPH, BAND_OVERLAP, use_yaw=False)
    if lens == "band_gy":
        return band_lens_weights(eps, BAND_MPH, BAND_OVERLAP, use_yaw=True)
    if lens == "avg_g_window":
        r = surface_ranked(eps, "avg_g")
    elif lens == "peak_g":
        r = surface_ranked(eps, "peak_g")
    elif lens == "yaw_intent":
        r = surface_ranked(eps, "yaw")
    elif lens == "combined":
        a = surface_ranked(eps, "avg_g")
        y = surface_ranked(eps, "yaw")
        r = [ai * yi for ai, yi in zip(a, y)]
    else:
        raise ValueError(lens)
    return [(0.1 + 0.9 * np.clip(x, 0, 1)).astype(np.float32) for x in r]


def curate_top_quartile(eps):
    scored = [(float(e["avg_g"][e["brake"]].mean()) if e["brake"].sum() else 0.0, i)
              for i, e in enumerate(eps)]
    scored.sort(reverse=True)
    keep = max(1, len(scored) // 4)
    return set(i for _, i in scored[:keep])


# -------------------------------------------------------------------- dataset

class SeqDataset(Dataset):
    """Slices sequences on the fly - never materializes seq_len copies."""

    def __init__(self, eps, ep_ids, weights, feat_mean, feat_std, max_seq, seed=0):
        self.eps, self.weights = eps, weights
        self.mean, self.std = feat_mean, feat_std
        index = []
        for ei in ep_ids:
            brake = eps[ei]["brake"]
            n = len(brake)
            valid = np.nonzero(brake[SEQ_LEN:n - 1])[0] + SEQ_LEN
            for t in valid:
                index.append((ei, t))
        rng = np.random.default_rng(seed)
        if len(index) > max_seq:
            sel = rng.choice(len(index), max_seq, replace=False)
            index = [index[i] for i in sel]
        self.index = index

    def __len__(self):
        return len(self.index)

    def __getitem__(self, i):
        ei, t = self.index[i]
        e = self.eps[ei]
        x = (e["X"][t - SEQ_LEN:t] - self.mean) / self.std
        return (torch.from_numpy(x.astype(np.float32)),
                torch.from_numpy(e["Y"][t]),
                torch.tensor(self.weights[ei][t]))


class LSTMBrakeNet(nn.Module):
    def __init__(self, n_in, n_out, hidden=64):
        super().__init__()
        self.lstm = nn.LSTM(n_in, hidden, num_layers=2, batch_first=True)
        self.head = nn.Linear(hidden, n_out)

    def forward(self, x):
        out, _ = self.lstm(x)
        return torch.sigmoid(self.head(out[:, -1, :]))  # 0-1 fraction of car max


def print_band_weight_summary(eps, ids, w):
    """Graded-bands-per-episode and braking-tick weight histogram, for band_* lenses."""
    edges, g_band, _ = band_grades(eps, BAND_MPH, BAND_OVERLAP)
    graded = np.array([np.sum(~np.isnan(g_band[i])) for i in ids])
    print(f"  bands={len(edges)}  graded/episode mean={graded.mean():.2f}")
    ticks = np.concatenate([w[i][eps[i]["brake"]] for i in ids])
    bins = np.linspace(0.1, 1.0, 6)
    hist, _ = np.histogram(ticks, bins=bins)
    frac = hist / max(hist.sum(), 1)
    print("  weight bins " + ", ".join(
        f"[{bins[j]:.2f},{bins[j+1]:.2f})={frac[j]:.3f}" for j in range(5)))


# ------------------------------------------------------------------- training

def train_lens(lens, eps, train_ids, val_ids, feat_mean, feat_std, car_max, epochs):
    w = lens_weights(eps, lens)
    ids = train_ids
    if lens == "curation":
        keep = curate_top_quartile(eps)
        ids = [i for i in train_ids if i in keep]
        print(f"  curated to {len(ids)}/{len(train_ids)} episodes")

    if lens.startswith("band_"):
        print_band_weight_summary(eps, ids, w)

    tr = SeqDataset(eps, ids, w, feat_mean, feat_std, MAX_SEQ_PER_LENS, seed=SEED)
    uniform = [np.ones(len(e["avg_g"]), dtype=np.float32) for e in eps]
    va = SeqDataset(eps, val_ids, uniform, feat_mean, feat_std, MAX_SEQ_PER_LENS // 4, seed=1)
    if len(tr) == 0 or len(va) == 0:
        print(f"  [{lens}] empty dataset, skipped")
        return None
    print(f"  train seqs={len(tr):,}  val seqs={len(va):,}")

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    dl_tr = DataLoader(tr, batch_size=512, shuffle=True, num_workers=0)
    dl_va = DataLoader(va, batch_size=1024, shuffle=False, num_workers=0)

    torch.manual_seed(SEED)
    model = LSTMBrakeNet(len(INPUT_COLS), len(TARGET_COLS)).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    lossf = nn.SmoothL1Loss(reduction="none")

    mae_hist = []
    for ep in range(1, epochs + 1):
        model.train()
        tot = 0.0
        for xb, yb, wb in dl_tr:
            xb, yb, wb = xb.to(dev), yb.to(dev), wb.to(dev)
            opt.zero_grad()
            per = lossf(model(xb), yb).mean(dim=1)
            loss = (per * wb).sum() / wb.sum().clamp(min=1e-6)
            loss.backward()
            opt.step()
            tot += loss.item() * len(xb)
        model.eval()
        mae = np.zeros(4)
        with torch.no_grad():
            for xb, yb, _ in dl_va:
                xb, yb = xb.to(dev), yb.to(dev)
                mae += (model(xb) - yb).abs().sum(dim=0).cpu().numpy()
        mae /= len(va)
        mae_hist.append([round(float(x), 5) for x in mae])
        if ep % 2 == 0 or ep == 1:
            print(f"    epoch {ep:2d}  train={tot/len(tr):.5f}  "
                  f"val_MAE(frac of car max)={np.round(mae, 4)}", flush=True)

    name = lens + TAG
    exp = EXP_DIR / name
    exp.mkdir(parents=True, exist_ok=True)
    path = exp / "model.pt"
    torch.save({"model": model.state_dict(), "feat_mean": feat_mean, "feat_std": feat_std,
                "car_max": car_max, "input_cols": INPUT_COLS, "target_cols": TARGET_COLS,
                "seq_len": SEQ_LEN, "lens": name, "zero_cols": list(ZERO_COLS), "sensor_src": SENSOR_SRC,
                "output": "fraction_of_car_max"}, path)
    meta = {"name": name, "lens": lens, "trained": datetime.now().isoformat(timespec="seconds"),
            "args": RUN_ARGS, "train_episodes": len(ids), "val_episodes": len(val_ids),
            "train_seqs": len(tr), "val_seqs": len(va), "epochs": epochs,
            "val_mae_final": [round(float(x), 5) for x in mae], "val_mae_by_epoch": mae_hist,
            "zero_cols": list(ZERO_COLS), "sensor_src": SENSOR_SRC, "seq_len": SEQ_LEN,
            "band_mph": BAND_MPH, "band_overlap": BAND_OVERLAP, "seed": SEED,
            "lock_slip": LOCK_SLIP, "lock_speed_mph": LOCK_SPEED_MPH, "lock_weight": LOCK_WEIGHT,
            "slip_good": list(SLIP_GOOD), "slip_good_weight": SLIP_GOOD_WEIGHT, "yaw_dz": YAW_DZ, "yaw_good": YAW_GOOD, "yaw_bad": YAW_BAD}
    (exp / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return {"lens": lens, "val_mae": mae.tolist(), "path": str(path)}


def main():
    global TAG, CACHE_PATH, CLEAN_ROOTS, SENSOR_SRC, BAND_MPH, BAND_OVERLAP, RUN_ARGS
    ap = argparse.ArgumentParser()
    ap.add_argument("--rebuild-cache", action="store_true")
    ap.add_argument("--limit-files", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0, help="init + sequence-sampling seed; the val split stays fixed")
    ap.add_argument("--lock-slip", type=float, default=0.0, help="down-weight ticks with any wheel slip above this (0 = off)")
    ap.add_argument("--lock-speed-mph", type=float, default=5.0, help="only above this true speed")
    ap.add_argument("--lock-weight", type=float, default=0.1, help="multiplier for those ticks")
    ap.add_argument("--slip-good", type=str, default="", help="lo,hi slip band to up-weight above --lock-speed-mph, e.g. 0.08,0.23")
    ap.add_argument("--slip-good-weight", type=float, default=1.5)
    ap.add_argument("--yaw-deadzone", type=float, default=0.0, help="rad/s; enables yaw-match weighting")
    ap.add_argument("--yaw-good", type=float, default=1.5)
    ap.add_argument("--yaw-bad", type=float, default=0.5)
    ap.add_argument("--zero-cols", type=str, default="")
    ap.add_argument("--tag", type=str, default="")
    ap.add_argument("--lenses", type=str, default="baseline,avg_g_window,peak_g,yaw_intent,combined,curation")
    ap.add_argument("--variants", type=str, default="", help="keep only these safety_variant values")
    ap.add_argument("--exclude-variants", type=str, default="", help="drop these safety_variant values")
    ap.add_argument("--models", type=str, default="", help="keep only these vehicle models")
    ap.add_argument("--roots", type=str, default="", help="comma list of clean-parquet roots (default CLEAN_ROOT)")
    ap.add_argument("--cache", type=str, default="", help="cache path override (default CACHE_PATH)")
    ap.add_argument("--sensor-src", type=str, default="ffi", help="ffi (raw ffiSensors) or gy2 (smoothed gx2/gy2/gz2); stored in ckpt")
    ap.add_argument("--min-peak-g", type=float, default=0.0,
                    help="drop episodes whose best 200ms forward-avg decel is below this (measured, not grip_pattern)")
    ap.add_argument("--band-mph", type=float, default=BAND_MPH,
                    help="speed-band width in mph for band_g/band_gy lenses")
    ap.add_argument("--band-overlap", type=int, default=BAND_OVERLAP,
                    help="band windows covering each speed; step = band_mph/band_overlap")
    args = ap.parse_args()
    RUN_ARGS = vars(args)
    global SEED
    SEED = args.seed
    global LOCK_SLIP, LOCK_SPEED_MPH, LOCK_WEIGHT
    LOCK_SLIP, LOCK_SPEED_MPH, LOCK_WEIGHT = args.lock_slip, args.lock_speed_mph, args.lock_weight
    global SLIP_GOOD, SLIP_GOOD_WEIGHT, YAW_DZ, YAW_GOOD, YAW_BAD
    if args.slip_good:
        SLIP_GOOD = tuple(float(x) for x in args.slip_good.split(","))
    SLIP_GOOD_WEIGHT = args.slip_good_weight
    YAW_DZ, YAW_GOOD, YAW_BAD = args.yaw_deadzone, args.yaw_good, args.yaw_bad
    TAG = args.tag
    SENSOR_SRC = args.sensor_src
    BAND_MPH = args.band_mph
    BAND_OVERLAP = args.band_overlap
    if args.roots:
        CLEAN_ROOTS = [Path(r) for r in args.roots.split(",") if r]
    if args.cache:
        CACHE_PATH = Path(args.cache)
    if args.zero_cols:
        ZERO_COLS.extend([c for c in args.zero_cols.split(",") if c])
        print(f"zeroing channels in train+deploy: {ZERO_COLS}")

    if args.rebuild_cache or not CACHE_PATH.exists():
        eps, car_max = build_cache(args.limit_files)
    else:
        print(f"loading cache {CACHE_PATH}")
        eps, car_max = load_cache()
    print(f"{len(eps)} episodes in memory")
    keep_v = set(v for v in args.variants.split(",") if v)
    drop_v = set(v for v in args.exclude_variants.split(",") if v)
    keep_m = set(m for m in args.models.split(",") if m)
    before = len(eps)
    eps = [e for e in eps
           if (not keep_v or e["variant"] in keep_v)
           and e["variant"] not in drop_v
           and (not keep_m or e["model"] in keep_m)
           and (e["brake"].sum() == 0 or float(e["avg_g"][e["brake"]].max()) >= args.min_peak_g)]
    print(f"episode filter: {before} -> {len(eps)}  (variants={sorted(keep_v) or 'all'} "
          f"minus {sorted(drop_v) or 'none'}, models={sorted(keep_m) or 'all'}, min_peak_g={args.min_peak_g})" + chr(10))

    rng = np.random.default_rng(0)
    order = rng.permutation(len(eps))
    n_val = max(1, int(len(eps) * 0.15))
    val_ids = sorted(order[:n_val].tolist())
    train_ids = sorted(order[n_val:].tolist())
    print(f"split: {len(train_ids)} train episodes / {len(val_ids)} val episodes")

    feats = np.concatenate([eps[i]["X"][eps[i]["brake"]] for i in train_ids[:400]], axis=0)
    feat_mean = feats.mean(axis=0)
    feat_std = feats.std(axis=0)
    feat_std[feat_std < 1e-6] = 1.0

    results = []
    for lens in args.lenses.split(","):
        print(f"\n--- lens: {lens} ---")
        r = train_lens(lens, eps, train_ids, val_ids, feat_mean, feat_std,
                       car_max, args.epochs)
        if r:
            results.append(r)

    print("\n=== Summary: val MAE (fraction of car max torque, lower=closer fit) ===")
    for r in results:
        m = r["val_mae"]
        print(f"  {r['lens']:14s} FR={m[0]:.4f} FL={m[1]:.4f} RR={m[2]:.4f} RL={m[3]:.4f}")
    print(f"\nModels -> {EXP_DIR}")


if __name__ == "__main__":
    main()
