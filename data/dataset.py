import numpy as np
import pandas as pd

CHANNELS = [
    "pressure_1", "pressure_2",
    "vibration_x", "vibration_y",
    "temperature_1", "temperature_2",
    "motor_current", "frequency",
    "torque", "rotational_speed",
]

SEQ_LEN = 50   # ← This was missing


def load_units(df: pd.DataFrame):
    """Group by unit_id into list of sequences."""
    units = []
    for uid, group in df.groupby('unit_id'):
        group = group.sort_values('timestep')
        seq = group[CHANNELS].fillna(0).values.astype(np.float32)
        
        # Ensure fixed length
        if len(seq) < SEQ_LEN:
            pad = np.tile(seq[-1:], (SEQ_LEN - len(seq), 1))
            seq = np.vstack([seq, pad])
        else:
            seq = seq[-SEQ_LEN:]
            
        rul = float(group['rul'].iloc[-1])
        fail = float(group['failure_near_term'].iloc[-1])
        dom = float(group['domain_label'].iloc[0])
        ac = group['asset_class'].iloc[0]
        
        units.append((seq, rul, fail, dom, ac))
    return units


def build_arrays(units):
    """Convert to arrays for training."""
    X = np.array([u[0] for u in units], dtype=np.float32)
    rul = np.array([u[1] for u in units], dtype=np.float32)
    fail = np.array([u[2] for u in units], dtype=np.float32)
    dom = np.array([u[3] for u in units], dtype=np.float32)
    ac = np.array([u[4] for u in units])
    return X, rul, fail, dom, ac