import numpy as np


def _db(x, floor_db):
    return 20 * np.log10(np.maximum(np.abs(x), 10 ** (floor_db / 20)))


def _local_minima(a, circular):
    """Boolean mask of strict-left / non-strict-right local minima along axis 1."""
    if circular:
        left, right = np.roll(a, 1, axis=1), np.roll(a, -1, axis=1)
        return (a < left) & (a <= right)
    m = np.zeros_like(a, dtype=bool)
    m[:, 1:-1] = (a[:, 1:-1] < a[:, :-2]) & (a[:, 1:-1] <= a[:, 2:])
    return m


def _mainlobe_mask(mag, minima, circular):
    """Per beam: indices between the first local minima on either side of the peak."""
    B, n = mag.shape
    mask = np.zeros_like(mag, dtype=bool)
    for i in range(B):
        p = int(np.argmax(mag[i]))
        lo = p
        for _ in range(n):
            nxt = lo - 1
            if not circular and nxt < 0:
                break
            nxt %= n
            if nxt == p:
                break
            lo = nxt
            if minima[i, lo]:
                break
        hi = p
        for _ in range(n):
            nxt = hi + 1
            if not circular and nxt >= n:
                break
            nxt %= n
            if nxt == p:
                break
            hi = nxt
            if minima[i, hi]:
                break
        if circular:
            idx = np.arange(lo, lo + ((hi - lo) % n) + 1) % n
        else:
            idx = np.arange(lo, hi + 1)
        mask[i, idx] = True
    return mask


def pattern_metrics(b_k, b_ref, circular=False, floor_db=-60.0, phase_mask_db=-40.0):
    
    bk, br = b_k, b_ref
    mag_k, mag_r = np.abs(bk), np.abs(br)
    diff = bk - br

    out = {
        "mbe": np.max(np.abs(mag_k - mag_r)),
        "mce": np.max(np.abs(diff)),
        "ibe": np.sqrt(np.sum(np.abs(diff) ** 2) / np.sum(mag_r ** 2)),
    }

    dphi = np.abs(np.angle(bk * np.conj(br)))
    valid = mag_r >= 10 ** (phase_mask_db / 20)
    out["phase_deg"] = np.rad2deg(np.max(dphi[valid])) if np.any(valid) else np.nan
    peak = np.argmax(mag_r, axis=1)
    out["phase_ml_deg"] = np.rad2deg(np.max(dphi[np.arange(len(peak)), peak]))

    minima = _local_minima(mag_r, circular)
    ml = _mainlobe_mask(mag_r, minima, circular)
    db_k, db_r = _db(bk, floor_db), _db(br, floor_db)
    sl = ~ml
    has_sl = np.any(sl, axis=1)
    psl_k = np.where(sl, db_k, -np.inf).max(axis=1)
    psl_r = np.where(sl, db_r, -np.inf).max(axis=1)
    out["psl_dev_db"] = np.max(np.abs(psl_k - psl_r)[has_sl]) if np.any(has_sl) else np.nan

    nulls = minima & sl
    out["null_dev_db"] = np.max(np.abs(db_k - db_r)[nulls]) if np.any(nulls) else np.nan
    return out


METRIC_LABELS = {
    "mbe": "MBE  max||b~|-|b||",
    "mce": "MCE  max|b~-b|",
    "ibe": "IBE  ||b~-b||_2/||b||_2",
    "phase_deg": "max phase err (deg, |b|>=mask)",
    "phase_ml_deg": "max phase err at look dir (deg)",
    "psl_dev_db": "max PSL deviation (dB)",
    "null_dev_db": "max null-depth deviation (dB)",
}


def print_metrics(tag, K, m):
    print(f"[{tag}] metrics at K={K}:")
    for key, label in METRIC_LABELS.items():
        print(f"    {label:<40s} {m[key]:.3e}")


def save_metrics_csv(path, K_range, rows):
    keys = list(METRIC_LABELS)
    data = np.column_stack([K_range] + [[r[k] for r in rows] for k in keys])
    np.savetxt(path, data, delimiter=",", header=",".join(["K"] + keys), comments="")
