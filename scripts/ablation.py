import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.train import train

CKPT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "checkpoints")
os.makedirs(CKPT_DIR, exist_ok=True)

CONFIGS = {
    "full_model (GRL+SimCLR)":     dict(ablate_grl=False, ablate_simclr=False),
    "GRL_only (SimCLR ablated)":   dict(ablate_grl=False, ablate_simclr=True),
    "SimCLR_only (GRL ablated)":   dict(ablate_grl=True,  ablate_simclr=False),
    "neither (baseline)":          dict(ablate_grl=True,  ablate_simclr=True),
}


def main():
    print("=" * 60)
    print("NADiSSP Ablation Study — Section 4.6.1")
    print("=" * 60)

    results = {}
    for name, flags in CONFIGS.items():
        print(f"\n── {name} ──")
        _, _, _, _, _, test_m, _, _ = train(n_epochs=20, verbose=False, **flags)
        
        results[name] = {k: round(float(v), 4) for k, v in test_m.items()}
        print(f"  AUPRC={test_m['failure_auprc']:.3f}  RMSE={test_m['rul_rmse']:.3f}  DomAcc={test_m['domain_accuracy']:.3f}")

    print(f"\n{'='*60}\nAblation Summary\n{'='*60}")
    print(f"{'Configuration':<35} {'RMSE':>8} {'AUPRC':>8} {'DomAcc':>8}")
    for name, m in results.items():
        print(f"{name:<35} {m['rul_rmse']:>8.3f} {m['failure_auprc']:>8.3f} {m['domain_accuracy']:>8.3f}")

    with open(os.path.join(CKPT_DIR, "ablation_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {CKPT_DIR}/ablation_results.json")


if __name__ == "__main__":
    main()