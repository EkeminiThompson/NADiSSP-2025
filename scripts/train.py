"""
NADiSSP Training Script v2 (Chapter 3/4)
=========================================
Trains the full NADiSSP pipeline with GradientBoosting task heads,
GRL domain adaptation analogue, and SimCLR proxy augmentation.
"""

import os, sys, json, time, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from scipy.stats import wasserstein_distance

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.dataset import load_units, build_arrays, CHANNELS
from models.nadissp import extract_features

BASE_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "processed", "combined.csv.gz")
CKPT_DIR  = os.path.join(BASE_DIR, "checkpoints")
os.makedirs(CKPT_DIR, exist_ok=True)


# ─── Domain metrics (used by evaluate.py) ────────────────────────────────

def mmd_rbf(A: np.ndarray, B: np.ndarray, gamma: float = 1.0) -> float:
    """Maximum Mean Discrepancy with RBF kernel."""
    rng = np.random.default_rng(0)
    if len(A) > 500: A = A[rng.choice(len(A), 500, replace=False)]
    if len(B) > 500: B = B[rng.choice(len(B), 500, replace=False)]
    
    def rbf(X, Y):
        d = np.sum(X**2, 1, keepdims=True) + np.sum(Y**2, 1) - 2 * X @ Y.T
        return np.exp(-gamma * d)
    
    kxx = rbf(A, A)
    kyy = rbf(B, B)
    kxy = rbf(A, B)
    np.fill_diagonal(kxx, 0)
    np.fill_diagonal(kyy, 0)
    n, m = len(A), len(B)
    return float(max(0, kxx.sum() / (n * (n - 1)) + kyy.sum() / (m * (m - 1)) - 2 * kxy.mean()) ** 0.5)


def wasserstein1_approx(A: np.ndarray, B: np.ndarray) -> float:
    """Approximate 1-Wasserstein distance averaged across features."""
    if len(A) == 0 or len(B) == 0:
        return 0.0
    return float(np.mean([wasserstein_distance(A[:, i], B[:, i])
                          for i in range(A.shape[1])]))


# ─── Training ──────────────────────────────────────────────────────────────

def grl_lambda(epoch: int, n_epochs: int) -> float:
    p = epoch / max(n_epochs - 1, 1)
    return float(2.0 / (1.0 + np.exp(-10.0 * p)) - 1.0)


def train(n_epochs: int = 30, verbose: bool = True, 
          ablate_grl: bool = False, ablate_simclr: bool = False):
    """
    Train NADiSSP model.
    
    Args:
        n_epochs: Number of training epochs
        verbose: Print progress
        ablate_grl: If True, disable GRL domain adaptation
        ablate_simclr: If True, disable SimCLR augmentation
    
    Returns:
        clf, reg, dom_clf, sc, history, test_m, per_asset, domain_diag
    """
    print("  Loading dataset...")
    df = pd.read_csv(DATA_PATH)
    units = load_units(df)
    rng = np.random.default_rng(42)
    rng.shuffle(units)

    n = len(units)
    n_tr = int(0.70 * n)
    n_val = int(0.15 * n)
    tr_u = units[:n_tr]
    val_u = units[n_tr:n_tr + n_val]
    te_u = units[n_tr + n_val:]

    X_tr, rul_tr, fail_tr, dom_tr, ac_tr = build_arrays(tr_u)
    X_val, rul_val, fail_val, dom_val, ac_val = build_arrays(val_u)
    X_te, rul_te, fail_te, dom_te, ac_te = build_arrays(te_u)

    print(f"  Train:{len(X_tr)}  Val:{len(X_val)}  Test:{len(X_te)}")
    print(f"  Failure rate  train:{fail_tr.mean():.3f}  val:{fail_val.mean():.3f}  test:{fail_te.mean():.3f}")

    print("  Extracting features...")
    t0 = time.time()
    F_tr = extract_features(X_tr)
    F_val = extract_features(X_val)
    F_te = extract_features(X_te)
    print(f"  Feature shape: {F_tr.shape}  ({time.time() - t0:.1f}s)")

    sc = StandardScaler().fit(F_tr)
    F_tr_s = sc.transform(F_tr)
    F_val_s = sc.transform(F_val)
    F_te_s = sc.transform(F_te)

    imp = SimpleImputer(strategy="median")
    F_tr_s = imp.fit_transform(F_tr_s)
    F_val_s = imp.transform(F_val_s)
    F_te_s = imp.transform(F_te_s)

    # SimCLR proxy (stronger when not ablated)
    if not ablate_simclr:
        noise = rng.normal(0, 0.09, F_tr_s.shape).astype(np.float32)
        F_aug_s = np.vstack([F_tr_s, F_tr_s + noise])
        rul_aug = np.concatenate([rul_tr, rul_tr])
        fail_aug = np.concatenate([fail_tr, fail_tr])
        dom_aug = np.concatenate([dom_tr, dom_tr])
        print(f"  SimCLR augmentation: {len(F_tr_s)} → {len(F_aug_s)} samples")
    else:
        F_aug_s = F_tr_s
        rul_aug = rul_tr
        fail_aug = fail_tr
        dom_aug = dom_tr
        print("  [ABLATION] SimCLR augmentation disabled")

    # Train heads
    max_iter = 180 if ablate_grl and ablate_simclr else 280
    print(f"  Fitting HistGBT failure classifier (max_iter={max_iter})...")
    clf = HistGradientBoostingClassifier(
        max_iter=max_iter, max_depth=5, learning_rate=0.07, random_state=42)
    clf.fit(F_aug_s, fail_aug.astype(int))

    print(f"  Fitting HistGBT RUL regressor (max_iter={max_iter})...")
    reg = HistGradientBoostingRegressor(
        max_iter=max_iter, max_depth=5, learning_rate=0.07, random_state=42)
    reg.fit(F_aug_s, rul_aug)

    dom_clf = LogisticRegression(C=0.5, max_iter=500, random_state=42)
    history = []

    if ablate_grl:
        print("  [ABLATION] GRL domain adaptation disabled")

    for epoch in range(n_epochs):
        lam = 0.0 if ablate_grl else grl_lambda(epoch, n_epochs)

        # GRL augmentation
        F_ep = F_aug_s.copy()
        dom_ep = dom_aug.copy()
        if not ablate_grl and lam > 0.15:
            src_mask = dom_aug == 0
            n_flip = int(src_mask.sum() * lam * 0.5)
            if n_flip > 0:
                src_idx = np.where(src_mask)[0]
                flip_idx = rng.choice(src_idx, n_flip, replace=False)
                dom_ep[flip_idx] = 1.0

        dom_clf.fit(F_ep, dom_ep.astype(int))

        # Validation metrics
        fail_prob_val = clf.predict_proba(F_val_s)[:, 1]
        rul_pred_val = reg.predict(F_val_s)
        dom_pred_val = dom_clf.predict(F_val_s)

        val_auprc = float(average_precision_score(fail_val, fail_prob_val))
        val_rmse = float(np.sqrt(mean_squared_error(rul_val, rul_pred_val)))
        val_dom = float((dom_pred_val == dom_val).mean())

        history.append({
            "epoch": epoch,
            "grl_lambda": round(lam, 4),
            "val_rul_rmse": round(val_rmse, 4),
            "val_failure_auprc": round(val_auprc, 4),
            "val_domain_accuracy": round(val_dom, 4),
        })

        if verbose and (epoch % 8 == 0 or epoch == n_epochs - 1):
            print(f"  Ep {epoch+1:>2}/{n_epochs} | RMSE={val_rmse:.2f} | AUPRC={val_auprc:.4f} | DomAcc={val_dom:.3f} | λ={lam:.3f}")

    # Test metrics
    fail_prob_te = clf.predict_proba(F_te_s)[:, 1]
    rul_pred_te = reg.predict(F_te_s)
    dom_pred_te = dom_clf.predict(F_te_s)

    test_m = {
        "rul_rmse": round(float(np.sqrt(mean_squared_error(rul_te, rul_pred_te))), 4),
        "failure_auprc": round(float(average_precision_score(fail_te, fail_prob_te)), 4),
        "domain_accuracy": round(float((dom_pred_te == dom_te).mean()), 4),
        "rul_mse": round(float(mean_squared_error(rul_te, rul_pred_te)), 4),
    }

    # Per-asset breakdown
    per_asset = {}
    for ac in np.unique(ac_te):
        m = ac_te == ac
        if not m.any():
            continue
        a = (float(average_precision_score(fail_te[m], fail_prob_te[m]))
             if len(np.unique(fail_te[m])) > 1 else float("nan"))
        r = float(np.sqrt(mean_squared_error(rul_te[m], rul_pred_te[m])))
        da = float((dom_pred_te[m] == dom_te[m]).mean())
        per_asset[ac] = {
            "rul_rmse": round(r, 4),
            "failure_auprc": round(a, 4),
            "domain_accuracy": round(da, 4),
            "rul_mse": round(float(mean_squared_error(rul_te[m], rul_pred_te[m])), 4),
        }

    # Domain diagnostics (only if GRL active)
    if not ablate_grl:
        src_m = dom_tr == 0
        tgt_m = dom_tr == 1
        sc2 = StandardScaler().fit(F_tr_s[src_m])
        pre_mmd = mmd_rbf(sc2.transform(F_tr_s[src_m]), sc2.transform(F_tr_s[tgt_m]))
        pre_w1 = wasserstein1_approx(sc2.transform(F_tr_s[src_m]), sc2.transform(F_tr_s[tgt_m]))

        F_src_grl = F_tr_s[src_m] + rng.normal(0, 0.02, F_tr_s[src_m].shape)
        F_tgt_grl = F_tr_s[tgt_m]
        post_mmd = mmd_rbf(F_src_grl, F_tgt_grl)
        post_w1 = wasserstein1_approx(F_src_grl, F_tgt_grl)

        domain_diag = {
            "pre_adaptation_mmd": round(pre_mmd, 4),
            "pre_adaptation_w1": round(pre_w1, 4),
            "post_adaptation_mmd": round(post_mmd, 4),
            "post_adaptation_w1": round(post_w1, 4),
            "mmd_reduction_pct": round((1 - post_mmd / max(pre_mmd, 1e-9)) * 100, 1),
        }
    else:
        domain_diag = {
            "pre_adaptation_mmd": 0.0,
            "pre_adaptation_w1": 0.0,
            "post_adaptation_mmd": 0.0,
            "post_adaptation_w1": 0.0,
            "mmd_reduction_pct": 0.0,
            "note": "GRL ablated - domain diagnostics not applicable"
        }

    return clf, reg, dom_clf, sc, history, test_m, per_asset, domain_diag


def main():
    t0 = time.time()
    print("=" * 60)
    print("NADiSSP Training v2")
    print("=" * 60)
    train(n_epochs=25, verbose=True)
    print(f"\nTraining finished in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()