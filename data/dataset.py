"""
NADiSSP Dataset Utilities
==========================
Converts the row-level CSV dataset into per-unit sequence arrays.
No PyTorch dependency — pure numpy/pandas.
"""

import numpy as np
import pandas as pd

CHANNELS = [
    "pressure_1", "pressure_2",
    "vibration_x", "vibration_y",
    "temperature_1", "temperature_2",
    "motor_current", "frequency",
    "torque", "rotational_speed",
]
SEQ_LEN = 50


def load_units(df: pd.DataFrame):
    """
    Group df by unit_id, sort by timestep, return list of dicts:
      {x: (seq_len, n_ch), rul: float, failure: float, domain: float, asset_class: str}
    """
    units = []
    for uid, g in df.groupby("unit_id", sort=False):
        g = g.sort_values("timestep")
        x = g[CHANNELS].fillna(0).values.astype(np.float32)
        # Pad / truncate to SEQ_LEN
        if len(x) < SEQ_LEN:
            pad = np.repeat(x[-1:], SEQ_LEN - len(x), axis=0)
            x = np.vstack([x, pad])
        else:
            x = x[-SEQ_LEN:]
        units.append({
            "x":           x,
            "rul":         float(g["rul"].iloc[-1]),
            "failure":     float(g["failure_near_term"].iloc[-1]),
            "domain":      float(g["domain_label"].iloc[0]),
            "asset_class": g["asset_class"].iloc[0],
            "unit_id":     uid,
        })
    return units


def build_arrays(units):
    """Stack unit list into flat arrays for sklearn."""
    X   = np.stack([u["x"] for u in units])           # (n, seq_len, n_ch)
    rul = np.array([u["rul"] for u in units])
    fail= np.array([u["failure"] for u in units])
    dom = np.array([u["domain"] for u in units])
    acs = np.array([u["asset_class"] for u in units])
    return X, rul, fail, dom, acs


def jitter_augment(X: np.ndarray, sigma: float = 0.05, rng=None) -> np.ndarray:
    """Additive Gaussian jitter for SimCLR pairs."""
    if rng is None:
        rng = np.random.default_rng()
    return X + rng.normal(0, sigma, X.shape).astype(np.float32)
