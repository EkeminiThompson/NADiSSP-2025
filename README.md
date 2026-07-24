# NADiSSP
## Nigerian-Adapted Domain-Invariant Self-Supervised Prognostics

Full research system for dissertation Chapters 3-5, Contributions 1-5.

---

## Stack

| Layer | Technology |
|---|---|
| ML model | scikit-learn (MLP encoder + task heads) |
| API | Flask (REST + background task runner) |
| Dashboard | Vanilla JS + Canvas (dark-theme SPA) |
| TCO simulation | NumPy Monte Carlo + Sobol sensitivity |
| Deployment | Docker / docker-compose |

> **No PyTorch required.** The sklearn backend runs on CPU-only environments.

---

## How to run — step by step, in order

Install dependencies first:

```bash
pip install -r requirements.txt
```

Then run each step below **in the order shown**. Each step depends on the previous one.

---

### Step 1 — Generate datasets

```bash
./run.sh generate
```

Generates synthetic multi-asset sensor sequences for all four asset classes
(CMAPSS turbofan, AI4I manufacturing, 3W offshore well, ESPset pump).
Applies all six Nigerian-context perturbations to the target domain.

- Output: `data/processed/source_domain.csv.gz`, `target_domain.csv.gz`, `combined.csv.gz`
- Section 3.4 / Tables 3.6-3.10
- Time: ~30 seconds

---

### Step 2 — SimCLR self-supervised pre-training (Contribution 2)

```bash
./run.sh pretrain
```

Trains the encoder on **unlabelled** Nigerian-augmented target sequences
using NT-Xent contrastive loss. No RUL labels required. This is Stage 2
of the six-stage training protocol and directly addresses the annotation
scarcity problem stated in the Abstract (>80% unlabelled data).

- Loss function: L = -(1/N) sum_i log[exp(sim(h_i1,h_i2)/tau) / sum_{k!=i} exp(sim(h_i1,h_k)/tau)]
- Temperature tau = 0.5, projection head: 32 -> 128 -> 64
- Output: `checkpoints/simclr_pretrained_encoder.joblib`, `checkpoints/simclr_history.json`
- The NT-Xent loss curve in simclr_history.json produces Figure 4.3 (left panel)
- Section 3.10.2 Component 3 / Section 3.10.3 Stage 2 / **Contribution 2**
- Time: ~2 minutes

---

### Step 3 — Train the full model (Contributions 1 and 3)

```bash
./run.sh train
```

Runs the joint supervised training stage. Automatically detects and loads
the SimCLR pre-trained encoder from Step 2 as a warm start (Stage 2 to
Stage 3 handoff). Runs 43 epochs with:
- GRL adversarial domain alignment (lambda cosine ramp 0 to 1)
- SimCLR feature-space augmentation proxy
- Weibull proportional hazards survival head
- RUL regression + failure classification task heads

- Output: `checkpoints/nadissp_model.joblib`, `checkpoints/metrics.json`, `checkpoints/train_history.json`
- Section 3.10.3 Stages 3-6 / Tables 3.17-3.20 / **Contributions 1 and 3**
- Time: ~30 seconds

---

### Step 4 — Ablation study (Contribution 3)

```bash
./run.sh ablation
```

Runs four ablation configurations to confirm the component dependency ordering
claim from Section 4.6.1: Full model (GRL+SimCLR), GRL only, SimCLR only,
Neither (baseline). Confirms that SimCLR provides its greatest marginal benefit
only when GRL adversarial alignment is already active.

- Output: `checkpoints/ablation_results.json`
- Section 4.6.1 / Table 4.4 / **Contribution 3**
- Time: ~2 minutes

---

### Step 5 — Generate all Chapter 4 figures and tables

```bash
./run.sh evaluate
```

Runs the full evaluation pipeline against the trained model and saves every
figure referenced in Chapter 4:

| Figure | File | Description |
|---|---|---|
| Fig 4.1 | fig4_1_tsne.png | t-SNE embeddings: domain + asset class |
| Fig 4.2 | fig4_2_training_curves.png | Training curves: AUPRC, RMSE, domain acc, lambda |
| Fig 4.3 (left) | fig4_3_simclr_pretrain.png | NT-Xent pre-training loss curve |
| Fig 4.3 (right) | fig4_3_ablation.png | Ablation bar chart |
| Fig 4.4 | fig4_4_rul_scatter.png | RUL scatter per asset class |
| Fig 4.5 | fig4_5_precision_recall.png | Precision-Recall curves per asset class |
| Fig 4.6 | fig4_6_shap_proxy.png | Feature attribution (permutation importance) |

Also writes:
- `results/domain_shift.json` — Table 4.1/4.2: MMD and Wasserstein-1 per asset class
- `results/shap_attribution.json` — Table 4.8: ranked channel attribution

- Time: ~3 minutes

---

### Step 6 — TCO/NPV Monte Carlo simulation (Contribution 4)

```bash
./run.sh tco
```

Runs 10,000 Monte Carlo trials of the TCO/NPV model over a 5-year horizon,
calibrated to Nigerian refinery economics (Table 4.9a priors). Includes Sobol
first-order and total-order sensitivity indices and national-scale projection
to 5 refineries.

- Output: `results/tco_summary.json`, `results/tco_distributions.csv`, `results/tco_report.txt`, `results/sensitivity.json`
- Section 4.4.4 / Table 4.9 / Table 4.9a / **Contribution 4**
- Time: ~1 minute

---

### Step 7 — Network resilience test (Contribution 5)

```bash
./run.sh network
```

Benchmarks inference latency (500 calls, P50/P95/P99) and measures AUPRC
degradation under 0-50% packet-loss simulation. If tc netem and cap_net_admin
are available, switches automatically to kernel-level testing. Otherwise uses
application-layer sensor-dropout simulation.

- Output: `results/network_test.json`
- Section 4.5 / Tables 4.10-4.11 / **Contribution 5**
- Time: ~2 minutes

---

### Step 8 — Start the dashboard and API

```bash
./run.sh serve
```

Starts Flask on **http://localhost:8000**. All results from Steps 1-7 are
immediately visible in the dashboard. No recomputation needed.

Dashboard pages:
- Overview — KPI summary, system status, per-asset metrics
- Model Status — architecture, training config, loss curves
- SimCLR Pre-Train — NT-Xent curve, view similarity, implementation notes
- Domain Shift — Table 4.1/4.2 MMD/W-1 before and after adaptation
- Performance — AUPRC, RMSE, SHAP attribution charts
- Ablation Study — Table 4.4 bar charts and results table
- Figures — all Figures 4.1-4.6 rendered inline
- TCO / NPV — Table 4.9, asset-class breakdown, Sobol indices
- Network Test — Tables 4.10-4.11, latency and packet-loss charts
- Live Predict — real-time inference with adjustable degradation slider
- Run Scripts — trigger any pipeline step from the browser

API docs: **http://localhost:8000/docs**

---

## Run everything at once

```bash
./run.sh full
```

Runs Steps 1-7 sequentially, then starts the server. Expect 10-15 minutes total.

---

## Docker

```bash
docker compose up --build
```

Runs the full pipeline automatically inside the container.

---

## Dissertation mapping

| File | Section | Contribution |
|---|---|---|
| `data/generate_datasets.py` | Section 3.4, Tables 3.6-3.10 | — |
| `models/nadissp.py` | Section 3.5, Figures 3.2-3.4 | **Contribution 1** |
| `scripts/pretrain_simclr.py` | Section 3.10.2 Component 3, Section 3.10.3 Stage 2 | **Contribution 2** |
| `scripts/train.py` | Section 3.10.3 Stages 3-6, Tables 3.17-3.20 | **Contributions 1 and 3** |
| `scripts/ablation.py` | Section 4.6.1, Table 4.4 | **Contribution 3** |
| `scripts/evaluate.py` | Chapter 4, Figures 4.1-4.6, Tables 4.1-4.8 | — |
| `scripts/tco_simulation.py` | Section 4.4.4, Table 4.9, Table 4.9a | **Contribution 4** |
| `scripts/network_test.py` | Section 4.5, Tables 4.10-4.11 | **Contribution 5** |
| `api/main.py` | Section 5.5 | **Contribution 5** |
| `dashboard/index.html` | Section 5.5 | **Contribution 5** |

---

## Remaining gaps before dissertation submission

**1. Real datasets**
CMAPSS, AI4I, 3W, and ESPset are external. Drop their CSVs into `data/raw/`
and update `generate_datasets.py` to load them instead of generating synthetic
data. The six-perturbation augmentation pipeline applies identically to real data.

**2. TCO calibration**
The current NPV mean (~$2.6B) overshoots the dissertation target ($892.4M)
because the priors model a larger fleet than specified. To calibrate, adjust
`CAPEX_MEDIAN_M`, `REACTIVE_LAMBDA`, and `OUTAGE_DAYS_MODE` in
`scripts/tco_simulation.py`, or pass a JSON config file:
```bash
python scripts/tco_simulation.py --config my_params.json
```

**3. Kernel-level packet-loss**
`network_test.py` uses application-layer sensor-dropout simulation. For a
rigorous Table 4.11 result, run on a Linux machine where `tc netem` is available
and the process has `cap_net_admin` capability. The script detects this
automatically and switches to kernel-level mode.

**4. Multi-hardware latency**
Table 4.10 rows for A100 and Jetson Orin are literature estimates. Run
`python scripts/network_test.py --latency-only` on each target machine to
replace them with measured values.
