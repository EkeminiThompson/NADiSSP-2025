import numpy as np
from scipy.stats import skew, linregress
import joblib

REPR_DIM = 64
SEQ_LEN = 50

# Define all 10 channels expected by the model
CHANNELS = [
    "pressure_1", "pressure_2",
    "vibration_x", "vibration_y",
    "temperature_1", "temperature_2",
    "motor_current", "frequency",
    "torque", "rotational_speed",
]


def extract_features(X: np.ndarray) -> np.ndarray:
    """
    Extract statistical features from sensor sequences.
    
    Features per channel (9):
      - mean, std, min, max, median, p25, p75, skew, slope
    Plus 6 cross-channel features:
      - pressure_ratio (p1/p2)
      - temperature_diff (t1 - t2)
      - vibration_magnitude (sqrt(vx^2 + vy^2))
      - power (current × torque)
      - efficiency (torque / current)
      - thermal_stress (pressure × temperature)
    
    Total features: 10 channels × 9 + 6 = 96 features
    
    Args:
        X: Input array of shape (batch, seq_len, n_channels)
    
    Returns:
        Feature matrix of shape (batch, 96)
    """
    if X.ndim == 2:
        X = X[np.newaxis]
    
    n, T, C = X.shape
    
    # Ensure we have the right number of channels
    expected_channels = len(CHANNELS)
    if C != expected_channels:
        # If we have fewer channels, pad with zeros
        if C < expected_channels:
            padded = np.zeros((n, T, expected_channels), dtype=np.float32)
            padded[:, :, :C] = X
            X = padded
            C = expected_channels
    
    feats = []
    for i in range(n):
        seq = X[i]
        row = []
        
        # ── Per-channel statistics (9 features each) ──────────────────
        for c in range(C):
            s = seq[:, c]
            # Skip if all NaN
            if np.all(np.isnan(s)):
                s = np.zeros_like(s)
            
            # Basic statistics (9 features per channel)
            row.extend([
                np.nanmean(s),           # mean
                np.nanstd(s),            # std
                np.nanmin(s),            # min
                np.nanmax(s),            # max
                np.nanmedian(s),         # median
                np.nanpercentile(s, 25), # p25
                np.nanpercentile(s, 75), # p75
                float(skew(s[~np.isnan(s)])) if len(s[~np.isnan(s)]) > 2 else 0.0,  # skew
                np.polyfit(np.arange(T), np.nan_to_num(s), 1)[0] if T > 1 else 0.0,  # slope
            ])
        
        # ── Cross-channel features (6) ──────────────────────────────────
        # 1. Pressure ratio (pressure_1 / pressure_2)
        p1 = seq[:, 0]  # pressure_1
        p2 = seq[:, 1]  # pressure_2
        p_ratio = np.nanmean(p1 / (p2 + 1e-8))
        row.append(p_ratio)
        
        # 2. Temperature difference (temp_1 - temp_2)
        t1 = seq[:, 4]  # temperature_1
        t2 = seq[:, 5]  # temperature_2
        temp_diff = np.nanmean(t1 - t2)
        row.append(temp_diff)
        
        # 3. Vibration magnitude (sqrt(vx^2 + vy^2))
        vx = seq[:, 2]  # vibration_x
        vy = seq[:, 3]  # vibration_y
        vib_mag = np.nanmean(np.sqrt(vx**2 + vy**2))
        row.append(vib_mag)
        
        # 4. Power (current × torque)
        curr = seq[:, 7]   # motor_current
        torque = seq[:, 8]  # torque
        power = np.nanmean(curr * torque)
        row.append(power)
        
        # 5. Efficiency (torque / current)
        efficiency = np.nanmean(torque / (curr + 1e-8))
        row.append(efficiency)
        
        # 6. Thermal stress (pressure × temperature)
        p_mean = np.nanmean(seq[:, 0])  # pressure_1
        t_mean = np.nanmean(seq[:, 4])  # temperature_1
        thermal_stress = p_mean * t_mean
        row.append(thermal_stress)
        
        feats.append(row)
    
    return np.array(feats, dtype=np.float32)


class ModelWrapper:
    """Wrapper for sklearn models to provide consistent interface."""
    
    def __init__(self, clf, reg, dom_clf, scaler, imputer):
        self.clf = clf
        self.reg = reg
        self.dom_clf = dom_clf
        self._feat_scaler = scaler
        self._imputer = imputer
        self._channels = CHANNELS
        self._fitted = True
        self.n_features_in_ = 96  # 10 channels × 9 + 6 cross-channel
    
    def predict(self, X):
        """
        Predict RUL and failure probability.
        
        Args:
            X: Input sequence (batch, seq_len, n_channels)
            
        Returns:
            dict with 'rul_pred' and 'failure_prob'
        """
        # Extract features (96 features: 9 per channel × 10 channels + 6 cross-channel)
        F = extract_features(X)
        
        # Scale and impute
        F_s = self._feat_scaler.transform(F)
        F_s = self._imputer.transform(F_s)
        
        # Predict
        rul_pred = np.clip(self.reg.predict(F_s), 0, 200)
        failure_prob = self.clf.predict_proba(F_s)[:, 1]
        
        return {
            'rul_pred': rul_pred,
            'failure_prob': failure_prob,
            'domain_pred': self.dom_clf.predict(F_s)
        }
    
    def predict_rul(self, X):
        """Predict RUL only."""
        F = extract_features(X)
        F_s = self._feat_scaler.transform(F)
        F_s = self._imputer.transform(F_s)
        return np.clip(self.reg.predict(F_s), 0, 200)
    
    def predict_failure_prob(self, X):
        """Predict failure probability only."""
        F = extract_features(X)
        F_s = self._feat_scaler.transform(F)
        F_s = self._imputer.transform(F_s)
        return self.clf.predict_proba(F_s)[:, 1]


class NADiSSP:
    """Main model class for backward compatibility."""
    
    def __init__(self):
        self._fitted = False
        self.scaler = None
        self.imputer = None
        self._feat_scaler = None
        self._imputer = None
        self.clf = None
        self.reg = None
        self.dom_clf = None
        self._channels = CHANNELS
        self.n_features_in_ = 96
    
    def predict(self, sequence):
        """Placeholder for live prediction - loaded from checkpoint in practice."""
        return {
            "rul_estimate": 45.0,
            "rul_unit": "cycles",
            "failure_probability": 0.12,
            "risk_level": "watch",
            "weibull_scale": 52.3,
            "weibull_shape": 2.1,
            "latency_ms": 18
        }
    
    def save(self, path):
        joblib.dump(self, path)
    
    @staticmethod
    def load(path):
        return joblib.load(path)


# ─── Test ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Test feature extraction
    batch_size = 2
    X_test = np.random.randn(batch_size, SEQ_LEN, len(CHANNELS)).astype(np.float32)
    features = extract_features(X_test)
    print(f"Input shape: {X_test.shape}")
    print(f"Output shape: {features.shape}")
    print(f"Expected: ({batch_size}, {len(CHANNELS) * 9 + 6}) = ({batch_size}, 96)")
    print(f"Match: {features.shape[1] == 96}")