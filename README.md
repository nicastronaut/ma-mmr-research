# Physical & Trophic Mechanisms of Cetacean Strandings in Massachusetts (2000–2025)

This repository contains a two-stage oceanographic, ecological, and machine learning investigation of **`4,831` cetacean stranding records** across Massachusetts (`2000–2025`), separating **Active Live Pelagic Odontocete Entrapment (Stage 1)** from **Passive Offshore Carcass Drift & Anthropogenic Bycatch/Vessel-Strike Backtracking (Stage 2)**.

---

## Papers & Reports

### 1. Stage 1 — De-Confounding Live Pelagic Odontocete Entrapment from Decomposed Carcass Drift
* **Research Paper:** [`stage1_clean_entrapment/STAGE1_PAPER.md`](stage1_clean_entrapment/STAGE1_PAPER.md)
* **Core Conclusion:** Two-thirds (`67.8%`, or `1,462` of `2,155` days) of the positive stranding days in the historical all-strandings target (`y_any`) contained zero healthy live or freshly dead pelagic dolphins; instead, they were days when decomposed carcasses or solitary nearshore harbor porpoises washed ashore from distant waters. Removing that passive drift noise and pairing **`136` physical weather, current, thermal-front, and Wellfleet spring-tide features** with **`48` Copernicus `SEAPODYM` trophic/prey features** (`npp`, `zooc`, `mnkc_epi` forage fish/squid, and `zeu`) in a **Two-Expert (`Physics × Prey`) Multiplicative Ensemble** achieves **`+188.0%` cross-validated PR-AUC lift on clean Mass Strandings** (`95% Holdout CI: [+6.2%, +73.4%]`, $P = 99.3\%$) and **`+33.2%` holdout lift on all live/fresh pelagic strandings** (`95% Holdout CI: [+12.4%, +60.1%]`, $P = 100.0\%$).

### 2. Stage 2 — Rewinding the Ocean: Lagrangian Reverse-Drift Backtracking of Decomposed & Bycaught Cetaceans (`Engine v2`)
* **Research Paper:** [`stage2_reverse_drift/STAGE2_PAPER.md`](stage2_reverse_drift/STAGE2_PAPER.md)
* **Core Conclusion:** Restoring all `4,831` records (including `379` confirmed human-interaction cases and `99` offshore `EEZ` floating carcass sightings), we classify Massachusetts towns into **Active Swimmer Trap Beaches** (`Wellfleet 59.8% alive`, `Dennis 52.1% alive`, `Brewster 46.5% alive`) versus **Passive Carcass Collector Beaches** (`Hull 83.3% decomposed`, `Chilmark 78.6%`, `Sandwich 78.3%`, `Westport 71.1%`, `Plymouth 66.7%`). Using a 50-particle Monte Carlo Lagrangian reverse-drift engine (`Engine v2`) enforced on the **U.S. Census `1:500,000` coastline rasterized at `0.001°` (`~100 m`)** at `30-minute` time steps across `2,282` coastal drifters (`114,100` trajectories), we validate backtracked hotspots against independent offshore `EEZ` sightings ($2.10\times$ mean / $5.49\times$ median enrichment under `Hold`, $1.91\times$ / $3.95\times$ under `Strict Strand`, $p < 0.0005$) and map the seasonal and basin-level origins of harbor porpoise gillnet bycatch, fishery entanglements, and vessel strikes.

---

## Repository Structure

```text
marine_bio/
├── README.md
├── RESEARCH_LOG.md
├── STAGE2_REVERSE_DRIFT_ROADMAP.md
├── stage1_clean_entrapment/
│   ├── STAGE1_PAPER.md                      # Stage 1 Journal Edition (numbered references [1]-[36])
│   ├── notebooks/
│   │   ├── 01_cohort_audit_and_targets.ipynb
│   │   ├── 02_clean_target_modeling.ipynb
│   │   └── 03_trophic_prey_integration.ipynb
│   ├── data/                            # Clean modeling tables, SEAPODYM prey series & CSV results
│   └── figures/                         # Stage 1 publication figures (Figures 1-5)
└── stage2_reverse_drift/
    ├── STAGE2_PAPER.md                      # Stage 2 Journal Edition (numbered references [1]-[36])
    ├── engine_v2/
    │   ├── reverse_drift_v2.py          # Core 0.001°-coastline Lagrangian particle engine (Hold & Strand)
    │   └── run_full_v2_pipeline.py      # Full 2,282-animal simulation & figure pipeline
    ├── notebooks/
    │   ├── 01_drifter_cohort_and_hi_audit.ipynb
    │   └── 02_lagrangian_backtracker_and_hotspots.ipynb
    ├── data/
    │   ├── v2/                          # Engine v2 outputs (Hold, Strand 2km, Strand 5km) & land mask
    │   └── *.csv / *.parquet            # Primary Stage 2 summary tables & cleaned records
    └── figures/                         # Stage 2 publication figures (s2_fig01 - s2_fig10)
```

---

## Reproducibility

All analyses run in Python `3.11+` (`numpy`, `pandas`, `scipy`, `scikit-learn`, `xarray`, `geopandas`, `shapely`, `matplotlib`).
* **Stage 1 Pipeline:** Execute `stage1_clean_entrapment/notebooks/01_cohort_audit_and_targets.ipynb` $\rightarrow$ `02_clean_target_modeling.ipynb` $\rightarrow$ `03_trophic_prey_integration.ipynb`.
* **Stage 2 Pipeline (`Engine v2`):** Execute `python stage2_reverse_drift/engine_v2/run_full_v2_pipeline.py` (note: the 75 MB daily `GLORYS12V1` NetCDF forcing cube `stage2_reverse_drift/data/glorys_wide_surface_1999_2025.nc` is excluded from git tracking via `.gitignore` to stay below GitHub's 50 MB file threshold and can be regenerated via Copernicus Marine Toolbox in `02_lagrangian_backtracker_and_hotspots.ipynb`).
