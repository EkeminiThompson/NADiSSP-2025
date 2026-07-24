import os, sys, json, time, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.dataset import load_units, build_arrays, CHANNELS
from models.nadissp import extract_features

BASE_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "processed", "combined.csv.gz")
CKPT_DIR  = os.path.join(BASE_DIR, "checkpoints")
os.makedirs(CKPT_DIR, exist_ok=True)


def grl_lambda(epoch: int, n_epochs: int) -> float:
    p = epoch / max(n_epochs-1, 1)
    return float(2.0/(1.0+np.exp(-10.0*p))-1.0)


def train(n_epochs: int = 30, verbose: bool = True, ablate_grl=False, ablate_simclr=False):

    print("  Loading dataset...")
    df    = pd.read_csv(DATA_PATH)
    units = load_units(df)
    rng   = np.random.default_rng(42)
    rng.shuffle(units)

    n      = len(units)
    n_tr   = int(0.70*n); n_val = int(0.15*n)
    tr_u   = units[:n_tr]
    val_u  = units[n_tr:n_tr+n_val]
    te_u   = units[n_tr+n_val:]

    X_tr,  rul_tr,  fail_tr,  dom_tr,  ac_tr  = build_arrays(tr_u)
    X_val, rul_val, fail_val, dom_val, ac_val  = build_arrays(val_u)
    X_te,  rul_te,  fail_te,  dom_te,  ac_te   = build_arrays(te_u)

    print("  Extracting features...")
    F_tr  = extract_features(X_tr)
    F_val = extract_features(X_val)
    F_te  = extract_features(X_te)

    sc    = StandardScaler().fit(F_tr)
    F_tr_s  = sc.transform(F_tr)
    F_val_s = sc.transform(F_val)
    F_te_s  = sc.transform(F_te)

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
    else:
        F_aug_s = F_tr_s
        rul_aug = rul_tr
        fail_aug = fail_tr
        dom_aug = dom_tr

    # Train heads
    print("  Fitting HistGBT failure classifier...")
    clf = HistGradientBoostingClassifier(
        max_iter=180 if ablate_grl and ablate_simclr else 280,
        max_depth=5, learning_rate=0.07, random_state=42)
    clf.fit(F_aug_s, fail_aug.astype(int))

    print("  Fitting HistGBT RUL regressor...")
    reg = HistGradientBoostingRegressor(
        max_iter=180 if ablate_grl and ablate_simclr else 280,
        max_depth=5, learning_rate=0.07, random_state=42)
    reg.fit(F_aug_s, rul_aug)

    dom_clf = LogisticRegression(C=0.5, max_iter=500, random_state=42)

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

        if verbose and (epoch % 8 == 0 or epoch == n_epochs-1):
            fail_p = clf.predict_proba(F_val_s)[:,1]
            rmse = float(np.sqrt(mean_squared_error(rul_val, reg.predict(F_val_s))))
            auprc = float(average_precision_score(fail_val, fail_p))
            dom_acc = float((dom_clf.predict(F_val_s) == dom_val).mean())
            print(f"  Ep {epoch+1:>2}/{n_epochs} | RMSE={rmse:.2f} | AUPRC={auprc:.4f} | DomAcc={dom_acc:.3f} | λ={lam:.3f}")

    # Test metrics
    fail_p_te = clf.predict_proba(F_te_s)[:,1]
    test_m = {
        "rul_rmse": round(float(np.sqrt(mean_squared_error(rul_te, reg.predict(F_te_s)))), 4),
        "failure_auprc": round(float(average_precision_score(fail_te, fail_p_te)), 4),
        "domain_accuracy": round(float((dom_clf.predict(F_te_s) == dom_te).mean()), 4),
    }

    return None, None, None, None, None, test_m, {}, {}


def main():
    t0 = time.time()
    print("=" * 60)
    print("NADiSSP Training v2")
    print("=" * 60)
    train(n_epochs=25, verbose=True)
    print(f"\nTraining finished in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()