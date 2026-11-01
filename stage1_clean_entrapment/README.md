# Stage 1: De-Confounding Live Pelagic Odontocete Entrapment from Decomposed Carcass Drift

This directory contains **Stage 1** of our follow-on marine conservation research building upon the Cape Cod Cetacean Stranding Capstone (`champ/capstone`).

For the complete self-contained technical report, figures, and plain-English data science explainers, read **[`STAGE1_MINI_PAPER.md`](STAGE1_MINI_PAPER.md)**.

---

## Directory Structure

```text
stage1_clean_entrapment/
├── README.md                              # Project overview & quickstart
├── STAGE1_MINI_PAPER.md                   # Complete Stage 1 technical report & reference guide
├── data/
│   ├── stage1_modeling_table.parquet      # 9,497 daily rows (2000-2025) with 6 de-confounded targets & cross-zone trap features
│   ├── cmems_lmtl_prey_daily_1999_2025.parquet # 26-year daily 0.083° CMEMS SEAPODYM trophic/prey biomass (npp, zooc, mnkc_epi, zeu)
│   ├── target_cohort_summary.csv          # Animal-level vs. day-level breakdown across all 6 target cohorts
│   ├── exp_a_target_deconfounding.csv     # Forward-chaining CV results (2009-2020) across all 6 targets & 6 models
│   ├── exp_b_gradient_synergy_ablation.csv# Feature ablation study for cross-zone trap gradients & spring-tide synergy
│   ├── exp_c_permutation_importance.csv   # Top feature importances across y_any, y_decomp_drift, y_live_odont_any, y_live_odont_mse
│   ├── exp_d_timescale_ablation.csv       # 30-day precondition vs. 0-7 day acute trap ablation
│   ├── exp_d_timescale_permutation.csv    # Variable x Timescale permutation importance
│   ├── exp_e_trophic_prey_cv.csv          # 12-fold CV results for Physical Trap vs. SEAPODYM Prey vs. Two-Expert Joint model
│   ├── exp_e_trophic_prey_holdout.csv     # 2021-2025 holdout results for Physical Trap vs. SEAPODYM Prey vs. Joint model
│   └── holdout_2021_2025_comparison.csv   # Final untouched 2021-2025 holdout evaluation across targets
├── figures/
│   ├── 01_cohort_and_seasonality.png      # Animal vs. Day contamination & opposite seasonal curves
│   ├── 02_target_deconfounding_lift.png   # PR-AUC lift over month-only baseline across cohorts + yearly win rates
│   ├── 02_permutation_importance_comparison.png # Shift from calendar season (carcasses) to wind/current physics (live odontocetes)
│   ├── 03_timescale_precondition_vs_acute.png   # 30-day precondition (Zellar et al.) vs. 0-7 day acute trap
│   └── 04_trophic_prey_and_ensemble.png   # Two-Expert (Physics x Prey) lift & food-web anomaly weights
└── notebooks/
    ├── 01_cohort_audit_and_targets.ipynb  # Cohort audit, Sick Flag check, seasonal divergence, & cross-zone feature engineering
    ├── 02_clean_target_modeling.ipynb     # 12-fold forward-chaining CV, feature ablation, permutation importance, & 2021-2025 holdout
    └── 03_trophic_prey_integration.ipynb  # CMEMS SEAPODYM trophic/prey integration & Two-Expert (Physics x Prey) Joint Model
```

---

## How to Reproduce

All notebooks run using the existing `capstone` conda environment (with `PYTHONNOUSERSITE=1` to avoid local user-site package conflicts):

```bash
PYTHONNOUSERSITE=1 /home/nicastronaut/.conda/envs/capstone/bin/jupyter nbconvert \
  --to notebook --execute --inplace notebooks/01_cohort_audit_and_targets.ipynb

PYTHONNOUSERSITE=1 /home/nicastronaut/.conda/envs/capstone/bin/jupyter nbconvert \
  --to notebook --execute --inplace notebooks/02_clean_target_modeling.ipynb

PYTHONNOUSERSITE=1 /home/nicastronaut/.conda/envs/capstone/bin/jupyter nbconvert \
  --to notebook --execute --inplace notebooks/03_trophic_prey_integration.ipynb
```

See [`RESEARCH_LOG.md`](RESEARCH_LOG.md) for the complete chronological research log and plain-English data science notes, and [`STAGE2_REVERSE_DRIFT_ROADMAP.md`](STAGE2_REVERSE_DRIFT_ROADMAP.md) for the Stage 2 roadmap on backtracking the `1,472` decomposed carcass days using reverse ocean drift modeling.
