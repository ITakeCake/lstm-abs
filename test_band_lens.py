"""Synthetic-episode checks for the band_g/band_gy speed-band lenses. No parquet."""
import numpy as np

from train_lenses_corpus import (
    DOWNSAMPLE, G, MPH_TO_MPS,
    band_edges_mph, band_grades, band_percentile_ranks, band_lens_weights,
)

DT = DOWNSAMPLE * 0.0005


def make_episode(decel_fn, v0=30.0, model="etk800"):
    """decel_fn(v) -> g at that speed; steps until v hits 0."""
    vs = [v0]
    while vs[-1] > 0:
        g = decel_fn(vs[-1])
        vs.append(max(vs[-1] - g * G * DT, 0.0))
    speed = np.array(vs, dtype=np.float32)
    n = len(speed)
    return {
        "speed": speed,
        "brake": np.ones(n, dtype=bool),
        "yaw": np.ones(n, dtype=np.float32),
        "model": model,
        "avg_g": np.full(n, np.nan, dtype=np.float32),
    }


def check(cond, msg):
    if not cond:
        raise AssertionError(msg)
    print(f"  OK: {msg}")


def find_interior_wobble(speed, edges, bump):
    """First (band_index, tick_index) where +bump crosses the band's upper edge,
    with valid ticks both before and after it in that band."""
    for bi, (lo_mph, hi_mph) in enumerate(edges):
        lo, hi = lo_mph * MPH_TO_MPS, hi_mph * MPH_TO_MPS
        idx = np.nonzero((speed >= lo) & (speed < hi))[0]
        if len(idx) < 10:
            continue
        for m in range(1, len(idx) - 1):
            if speed[idx[m]] + bump >= hi:
                return bi, idx[m]
    return None


def main():
    ep_a = make_episode(lambda v: 1.0)                       # constant 1.0g
    ep_b = make_episode(lambda v: 0.6)                        # constant 0.6g
    ep_c = make_episode(lambda v: 1.0 if v > 12.0 else 0.3)   # split at 12 m/s
    eps = [ep_a, ep_b, ep_c]

    print("--- grade accuracy (default: band_mph=5, overlap=1) ---")
    edges, g_band, y_band = band_grades(eps, band_mph=5.0, band_overlap=1)
    ga, gb = g_band[0], g_band[1]
    graded_a = ~np.isnan(ga)
    check(graded_a.sum() > 5, f"episode a has graded bands ({graded_a.sum()})")
    check(np.all(np.abs(ga[graded_a] - 1.0) / 1.0 < 0.05), "episode a grades within 5% of 1.0g")
    graded_b = ~np.isnan(gb)
    check(np.all(np.abs(gb[graded_b] - 0.6) / 0.6 < 0.05), "episode b grades within 5% of 0.6g")

    print("--- ranking: a beats b in every commonly-graded band ---")
    both = graded_a & graded_b
    check(both.sum() > 3, f"a and b share graded bands ({both.sum()})")
    check(np.all(ga[both] > gb[both]), "a's raw grade exceeds b's in every shared band")
    ranks = band_percentile_ranks(eps, g_band)
    ra, rb = ranks[0], ranks[1]
    check(np.all(ra[both] >= rb[both]), "a outranks (or ties) b in every shared band")
    check(np.any(ra[both] > rb[both]), "a strictly outranks b in at least one band")

    print("--- ranking: c vs b, high speed vs low speed ---")
    gc = g_band[2]
    hi_mph_thresh = (12.0 / MPH_TO_MPS) + 5.0   # comfortably above the 12 m/s split
    lo_mph_thresh = (12.0 / MPH_TO_MPS) - 5.0   # comfortably below it
    hi_bands = edges[:, 0] >= hi_mph_thresh
    lo_bands = edges[:, 1] <= lo_mph_thresh
    rc = ranks[2]
    hi_ok = hi_bands & ~np.isnan(gc) & ~np.isnan(gb)
    lo_ok = lo_bands & ~np.isnan(gc) & ~np.isnan(gb)
    check(hi_ok.sum() > 0, f"have high-speed bands to compare ({hi_ok.sum()})")
    check(lo_ok.sum() > 0, f"have low-speed bands to compare ({lo_ok.sum()})")
    check(np.all(rc[hi_ok] > rb[hi_ok]), "c outranks b at high speed")
    check(np.all(rb[lo_ok] > rc[lo_ok]), "b outranks c at low speed")

    print("--- weight bounds ---")
    w_g = band_lens_weights(eps, 5.0, 1, use_yaw=False)
    w_gy = band_lens_weights(eps, 5.0, 1, use_yaw=True)
    for w in w_g + w_gy:
        check(np.all(w >= 0.1 - 1e-6) and np.all(w <= 1.0 + 1e-6), "weights within [0.1, 1.0]")

    print("--- wobble robustness: single +0.05 m/s tick reversal (interior gap) ---")
    wobble_mph, wobble_overlap = 5.0, 2
    edges_w = band_edges_mph([ep_a], band_mph=wobble_mph, band_overlap=wobble_overlap)
    hit = find_interior_wobble(ep_a["speed"], edges_w, bump=0.05)
    check(hit is not None, "found a band/tick where the wobble crosses an interior boundary")
    bi, k = hit
    ep_w = {**ep_a, "speed": ep_a["speed"].copy()}
    ep_w["speed"][k] += 0.05
    lo_mph, hi_mph = edges_w[bi]
    lo_ms, hi_ms = lo_mph * MPH_TO_MPS, hi_mph * MPH_TO_MPS
    idx_after = np.nonzero((ep_w["speed"] >= lo_ms) & (ep_w["speed"] < hi_ms))[0]
    check(np.any(np.diff(idx_after) != 1), "wobble creates a genuine interior gap in band membership")
    _, g_band_w, _ = band_grades([ep_w], band_mph=wobble_mph, band_overlap=wobble_overlap)
    g_wobble = g_band_w[0][bi]
    check(not np.isnan(g_wobble), "wobbled band is still graded, not rejected")
    check(abs(g_wobble - 1.0) / 1.0 < 0.05, f"wobbled band grades within 5% of 1.0g (got {g_wobble:.3f})")

    print("--- band membership: overlap=1 (default) ---")
    edges1 = band_edges_mph(eps, band_mph=5.0, band_overlap=1)
    lo1, hi1 = edges1[:, 0] * MPH_TO_MPS, edges1[:, 1] * MPH_TO_MPS

    def n_bands_for(v, lo, hi):
        return int(np.sum((lo <= v) & (v < hi)))

    sample_v = ep_a["speed"][ep_a["speed"] > 0][len(ep_a["speed"]) // 3]
    check(n_bands_for(sample_v, lo1, hi1) == 1,
          f"a braking tick (v={sample_v:.2f} m/s) is in exactly one band (overlap=1)")
    for v in ep_a["speed"][::50]:
        n = n_bands_for(v, lo1, hi1)
        check(n == 1, f"every sampled tick in exactly one band (v={v:.2f} -> {n})")

    print("--- band membership: overlap=2 ---")
    edges2 = band_edges_mph(eps, band_mph=5.0, band_overlap=2)
    lo2, hi2 = edges2[:, 0] * MPH_TO_MPS, edges2[:, 1] * MPH_TO_MPS
    v_above_first_step = 3.0 * MPH_TO_MPS   # first step = 2.5 mph; 3 mph is above it
    n2 = n_bands_for(v_above_first_step, lo2, hi2)
    check(n2 == 2, f"a tick above the first step (3 mph) is in exactly two bands (got {n2})")
    v_below_first_step = 1.0 * MPH_TO_MPS
    n2b = n_bands_for(v_below_first_step, lo2, hi2)
    check(n2b == 1, f"a tick below the first step (1 mph) is in exactly one band (got {n2b})")

    print("\nALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    import sys
    try:
        sys.exit(main())
    except AssertionError as e:
        print(f"\nFAIL: {e}")
        sys.exit(1)
