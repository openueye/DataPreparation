from __future__ import annotations

from typing import Dict, Optional

import numpy as np


def nearest_indices(reference_ts: np.ndarray, target_ts: np.ndarray) -> np.ndarray:
    if len(target_ts) == 0:
        raise ValueError("Cannot match timestamps against an empty target array.")
    indices = np.searchsorted(target_ts, reference_ts)
    left = np.clip(indices - 1, 0, len(target_ts) - 1)
    right = np.clip(indices, 0, len(target_ts) - 1)
    left_diff = np.abs(reference_ts - target_ts[left])
    right_diff = np.abs(reference_ts - target_ts[right])
    return np.where(left_diff <= right_diff, left, right)


def nearest_neighbor_stats(reference_ts: np.ndarray, target_ts: np.ndarray) -> Optional[Dict[str, float]]:
    if len(reference_ts) == 0 or len(target_ts) == 0:
        return None
    overlap = reference_ts[(reference_ts >= target_ts[0]) & (reference_ts <= target_ts[-1])]
    if len(overlap) == 0:
        return None
    matches = nearest_indices(overlap, target_ts)
    deltas_ms = np.abs(overlap - target_ts[matches]) / 1e6
    return {
        "matched_count": int(len(overlap)),
        "mean_ms": float(deltas_ms.mean()),
        "median_ms": float(np.median(deltas_ms)),
        "p95_ms": float(np.percentile(deltas_ms, 95)),
        "max_ms": float(deltas_ms.max()),
    }
