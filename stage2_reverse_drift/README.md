# Stage 2: Lagrangian Reverse-Drift Backtracking of Dead & Bycaught Cetaceans (`2000–2025`)

This directory contains the code, datasets, validation tables, and figures for **Stage 2** of the Massachusetts Cetacean Conservation Data Science project, turning the ~2,300 decomposed and freshly dead single carcasses (plus 379 confirmed Human Interaction records and 99 offshore `EEZ` sightings) into an **At-Sea Mortality & Bycatch Backtracking System**.

## Directory Structure

- [`STAGE2_MINI_PAPER.md`](STAGE2_MINI_PAPER.md) — Complete Stage 2 research paper with plain-English explainers, physics equations, validation tables, figures, and code summaries.
- `notebooks/`
  - [`01_drifter_cohort_and_hi_audit.ipynb`](notebooks/01_drifter_cohort_and_hi_audit.ipynb) — Cleans all `4,831` NOAA records (`2000–2025`), separates **Active Swimmer Trap Beaches** from **Passive Carcass Collector Beaches**, and defines the 5 coastal drifter cohorts and the `EEZ` offshore validation cohort.
  - [`02_lagrangian_backtracker_and_hotspots.ipynb`](notebooks/02_lagrangian_backtracker_and_hotspots.ipynb) — Implements the vectorized Monte Carlo Lagrangian Reverse-Drift Engine (`114,100` particle trajectories over 26 years of `GLORYS12V1` currents + `ERA5` winds), validates against `77` in-domain `EEZ` floating sightings (`4.59x` median KDE enrichment, `p < 0.001`), runs a 5-way physics sensitivity analysis, and maps the **Source–Receptor Matrix** and cohort mortality hotspots.
- `data/`
  - `glorys_wide_surface_1999_2025.nc` — 26-year (`1999-12-01` to `2025-12-31`, `9,528` days) `0.083°` (`~8 km`) `CMEMS GLORYS12V1` daily surface currents (`uo`, `vo`) and sea-surface temperature (`thetao`) over `40.2°N–43.2°N, -71.6°W–-68.5°W`.
  - `stage2_all_records_clean.parquet` & `stage2_drifter_cohorts.parquet` — Cleaned record-level stranding tables with imputed/corrected coordinates, decomposition drift tiers, human interaction flags, and coastal reception zones.
  - `backtracked_drifters_summary.parquet` — Inferred mortality origin coordinates (`origin_lat_mean`, `origin_lon_mean`), ensemble spread (`origin_spread_km`), SST-scaled drift durations (`drift_days_mean`), net drift distance (`net_drift_km`), and attributed offshore basin (`origin_basin`) for all `2,282` coastal drifters.
  - `collector_vs_trap_beaches.csv`, `cohort_summary_table.csv`, `cohort_basin_attribution.csv`, `source_receptor_matrix.csv`, `sensitivity_analysis_table.csv`, `seasonal_hotspot_shift.csv`.
- `figures/`
  - `01_drifter_cohorts_and_collector_beaches.png`
  - `02_reverse_trajectories_and_validation.png`
  - `03_mortality_hotspot_kde_maps_by_cohort.png`
