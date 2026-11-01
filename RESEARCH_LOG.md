# Research Log — Marine Mammal Conservation Data Science (`marine_bio`)

Chronological record of decisions, plain-English methodology notes, data checks, and modeling results. Entries are append-only so our reasoning trail is always clear and reproducible.

---

## 2026-10-23 — Project Kickoff: The Two-Stage "De-Confound & Backtrack" Strategy

**What:** Set up the `marine_bio` workspace to extend and sharpen the findings from the Champlain College MSIT Capstone (`original capstone baseline`). Created [`STAGE2_REVERSE_DRIFT_ROADMAP.md`](STAGE2_REVERSE_DRIFT_ROADMAP.md) to track the Stage 2 Lagrangian reverse-drift project, and initialized `stage1_clean_entrapment/` for Stage 1.

**Why (The Core Scientific Insight):**
In the original capstone (`03_data_cleaning.ipynb` and `07_label_engineering.ipynb`), all 4,368 non-human-interaction Massachusetts cetacean strandings from 2000–2025 were combined into a single daily target (`y_any`, 2,155 positive days), with a secondary target for mass strandings (`y_mse`, 382 positive days).
- In the main capstone, predicting `y_any` from environmental data barely beat guessing based on the calendar month alone (+0.017 PR-AUC in development, -0.003 on the 2021–2025 holdout), because **98% of sea surface temperature (SST) variation is just the seasonal cycle**.
- However, in the follow-up sub-project (`mse_only_research/`), tuning a model specifically for **mass strandings (`y_mse`)** produced a **+143% relative improvement** over the seasonal baseline in development (`0.1154` vs. `0.0475` PR-AUC). Even more importantly, when seasonal SST was stripped out using anomalies (`04_extended_development.ipynb`), the model stopped relying on the calendar and leaned on **real physical transport variables**: Bay north-south wind (`bay_wind_v`), Outer Cape north-south current (`outer_current_v`), Bay east-west wind (`bay_wind_u`), and wind stress (`bay_wind_stress_u`).

**Why was `y_any` so noisy while `y_mse` showed a real physical signal?**
Because the 4,368 records in `strandings_clean_all.parquet` actually mix **two completely different physical processes**:
1. **Live/Fresh Behavioral Entrapment (~57% of records):** Healthy, social pelagic dolphins (*Delphinus delphis*, *Lagenorhynchus acutus*, *Globicephala melas*, *Tursiops truncatus*) swim into Cape Cod Bay and get trapped by the hook geometry, shallow tidal flats, and wind/current anomalies.
2. **Offshore Death & Passive Carcass Drift (~43% of records):** Animals found in **Moderate Decomposition (1,479 rows)**, **Advanced Decomposition (467 rows)**, or **Mummified/Skeletal (103 rows)** condition—plus **797 Harbor Porpoises** (a solitary coastal/shelf species prone to gillnet bycatch) and **~300 large whales**. A rotting carcass died days or weeks earlier offshore and drifted onto the beach like a message in a bottle. Treating its beach discovery date as an "entrapment day" caused by local Cape Cod Bay weather adds pure **label contamination** (training a model on false positive labels that don't match the mechanism you're trying to predict).

**Decisions (Agreed with Jake):**
1. **Hybrid Two-Stage Plan:**
   - **Stage 1 (`stage1_clean_entrapment/`):** Clean the Option 1 entrapment target by separating **live/fresh-dead non-sick pelagic odontocetes** from **decomposed drifting carcasses and non-pelagic species**, engineer cross-region thermal front and wind×tide interaction features, and test whether removing carcass-drift noise sharpens the environmental entrapment signal.
   - **Stage 2 (`STAGE2_REVERSE_DRIFT_ROADMAP.md`):** Use the decomposed carcasses (`Moderate`/`Advanced Decomposition`) and the 379 `Findings of Human Interaction == 'Y'` / 99 `EEZ` offshore records to build a **Lagrangian Reverse-Drift Backtracker**, rewinding ocean currents and winds from the stranding beach to map where animals are dying at sea.
2. **Data & Environment Reuse:** Read the cleaned Parquet tables directly from `data/processed/` without duplicating raw data or modifying anything in the capstone folder.
3. **Communication & Documentation Style:** Keep explanations plain-English and accessible, avoiding unnecessary academic jargon and providing brief, intuitive explainers whenever introducing data science or oceanographic terms.

**Next:** Build and run `stage1_clean_entrapment/notebooks/01_cohort_audit_and_targets.ipynb` to measure the exact decomposition and species breakdown across `y_any` and `y_mse`, build the de-confounded daily targets, and engineer the cross-region contrast features.


---

## 2026-10-23 — Stage 1 Notebook 01: Cohort Audit, `Sick Flag` Asymmetry, & De-Confounded Target Engineering (`01_cohort_audit_and_targets.ipynb`)

**What:** Built, executed, and validated [`stage1_clean_entrapment/notebooks/01_cohort_audit_and_targets.ipynb`](stage1_clean_entrapment/notebooks/01_cohort_audit_and_targets.ipynb). Audited all `4,368` cleaned stranding records (`strandings_clean_all.parquet`) and joined them against the `9,497`-day environmental feature matrix (`modeling_table.parquet`, 2000–2025). Saved [`stage1_modeling_table.parquet`](stage1_clean_entrapment/data/stage1_modeling_table.parquet) (`9,497` rows × `250` columns), [`target_cohort_summary.csv`](stage1_clean_entrapment/data/target_cohort_summary.csv), and [`01_cohort_and_seasonality.png`](stage1_clean_entrapment/figures/01_cohort_and_seasonality.png).

**Key Findings & Data Science Insights:**

1. **Why Daily Aggregation Amplified the Noise (`67.8%` Contamination of `y_any`):**
   - At the individual **animal level (`4,368` records)**, healthy (`Sick Flag != 'Y'`) live or freshly dead pelagic odontocetes (*Delphinus delphis*, *Lagenorhynchus acutus*, *Globicephala melas*, *Grampus griseus*, *Stenella coeruleoalba*, *Tursiops truncatus*, *Kogia breviceps*) make up **`46.4%` (`2,028` animals)**, while decomposed carcasses (`Moderate Decomposition`, `Advanced Decomposition`, `Mummified/Skeletal`), Harbor Porpoises (`Phocoena phocoena`), and large whales make up the remaining `53.6%`.
   - However, our machine learning model predicts **calendar days (`9,497` days)**, not individual animals! Because social pelagic dolphins strand in groups (up to 146 animals on a single day), those `2,028` healthy live/fresh animals compress into only **`693` positive days (`32.2%` of `y_any`)**.
   - Conversely, decomposed carcasses (`1,472` days total) and solitary nearshore Harbor Porpoises (`753` records: `96.9%` single strandings, `70.8%` decomposed) arrive one at a time across the calendar, creating **`1,462` positive days in `y_any` (`67.8%`) where zero healthy live or freshly dead pelagic odontocetes stranded**.
   - *Data Science Explainer (Label Contamination):* When more than two-thirds (`67.8%`) of your positive `1` labels in `y_any` are actually passive floating carcasses or solitary coastal porpoises, any classifier trying to link same-day Cape Cod Bay weather to stranding days is being penalized two-thirds of the time for predicting true live-entrapment conditions!

2. **Discovery of `Sick Flag` Reporting Asymmetry:**
   - Auditing `Sick Flag == 'Y'` (`225` records total) revealed an important reporting nuance in the NOAA database: `Sick Flag` is populated almost exclusively on `Alive` animals (`224` of `225` rows; `0` on `Moderate` or `Advanced Decomposition`) because field responders can only evaluate clinical sickness on a live animal.
   - Furthermore, those `225` records split into two distinct biological contexts:
     - **Single Strandings (`133` records):** Solitary sick/emaciated animals coming ashore one by one due to individual illness.
     - **Mass Strandings (`91` records across `48` MSE days):** Animals recorded as "sick/shocked" during a group stranding on the tidal flats (where literature shows >90% of the herd was healthy prior to beaching).
   - To test this cleanly without guessing, we built both **`y_live_odont_all_any`** (`798` days; all live/fresh pelagic odontocetes regardless of `Sick Flag`) and **`y_live_odont_any`** (`693` days; strictly excluding `Sick Flag == 'Y'`), and kept MSEs inclusive of shock flags in **`y_live_odont_mse`** (`312` days, removing the **`70` MSE days (`18.3%`)** that only had decomposed aftermath carcasses or non-pelagic species).

3. **Opposite Seasonal Curves (`01_cohort_and_seasonality.png`):**
   - Plotting the monthly rate of each cohort revealed why `y_any` was so muddied:
     - **Decomposed Carcass Drift (`y_decomp_drift`, `1,472` days)** peaks in **late spring (`May–June`, ~25% of days)** as warming spring shelf waters accelerate bacterial gas bloat and re-float winter/spring carcasses, with a secondary peak in **March–April** (harbor porpoises).
     - **Live/Fresh Pelagic Entrapment (`y_live_odont_any`, `693` days; `y_live_odont_mse`, `312` days)** drops to its **lowest point of the year in May–June (~3.5% of days)** and peaks sharply in **late autumn and winter (`November–March`, ~10–13% of days)** when common and white-sided dolphins follow baitfish into Cape Cod Bay during rapid autumn cooling and winter storms.

4. **Cross-Zone Trap Gradient & Synergy Feature Engineering:**
   - Added **10 physically motivated features** to capture the spatial funnel and tidal trap mechanism:
     - `grad_bay_minus_outer_sst` & `grad_bay_minus_sound_sst` (and 7-day changes): Measures whether Cape Cod Bay is cooling faster than the open Atlantic (`outer`) or Nantucket Sound (`sound`), creating a thermal boundary across the mouth of the Bay.
     - `grad_bay_minus_outer_current_v` & `grad_bay_minus_outer_wind_v`: Cross-zone current and wind convergence into the Bay hook.
     - `synergy_onshore_wind_x_spring_tide` & `synergy_wind_speed_x_spring_tide`: Interaction terms multiplying onshore wind forcing (`-bay_wind_v`) and wind speed by the spring/neap tidal range (`bay_tide_spring_neap`), testing whether strong onshore winds are most dangerous during extreme spring tides when miles of Wellfleet mudflats drain rapidly.

---

## 2026-10-23 — Stage 1 Notebook 02: Target De-Confounding Modeling, Feature Ablation, & 2021–2025 Holdout Breakthrough (`02_clean_target_modeling.ipynb`)

**What:** Built, executed, and validated [`stage1_clean_entrapment/notebooks/02_clean_target_modeling.ipynb`](stage1_clean_entrapment/notebooks/02_clean_target_modeling.ipynb) using the exact leakage-free 12-fold forward-chaining validation protocol (`2009–2020`, `7,670` dev days) and the untouched 5-year holdout (`2021–2025`, `1,826` days) from the capstone. Saved [`exp_a_target_deconfounding.csv`](stage1_clean_entrapment/data/exp_a_target_deconfounding.csv), [`exp_b_gradient_synergy_ablation.csv`](stage1_clean_entrapment/data/exp_b_gradient_synergy_ablation.csv), [`exp_c_permutation_importance.csv`](stage1_clean_entrapment/data/exp_c_permutation_importance.csv), [`holdout_2021_2025_comparison.csv`](stage1_clean_entrapment/data/holdout_2021_2025_comparison.csv), and figures [`02_target_deconfounding_lift.png`](stage1_clean_entrapment/figures/02_target_deconfounding_lift.png) and [`02_permutation_importance_comparison.png`](stage1_clean_entrapment/figures/02_permutation_importance_comparison.png).

**Key Modeling Results:**

1. **Experiment A — De-Confounding the Target Flips All-Events (`Single + MSE`) from Failure to Strong Success:**
   - First verified exact numerical alignment with the capstone: on `y_mse`, `Logistic Regression (raw, C=0.3)` reproduced **`0.1154` pooled PR-AUC (`+142.9%` over `month-only`)** to the fourth decimal place.
   - Across our 6-target hierarchy (`Logistic Regression, C=0.01` for all-event targets, `C=0.3` for MSE targets):

| Target Cohort | Dev Pos Days (2000–2020) | Month-Only Pooled PR-AUC | Raw Env Model Pooled Lift | **Anomaly Model Pooled Lift** | **Anomaly Year-by-Year Mean Lift** | **Anomaly Years Won (of 12)** |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`y_decomp_drift`** (Decomposed Carcasses Only) | 1,083 (14.1%) | 0.2425 | -2.9% | **-4.4%** | +4.0% | 10/12 |
| **`y_any`** (Original All-Strandings: 68% Noise) | 1,568 (20.4%) | 0.3368 | -6.5% | **-4.5%** | +3.1% | 8/12 |
| **`y_live_odont_all_any`** (Live Pelagic All, Inc. Sick) | 557 (7.3%) | 0.1089 | +6.7% | **+11.4%** | **+26.1%** | 8/12 |
| **`y_live_odont_any`** (Healthy Live Pelagic: Single + MSE) | 487 (6.3%) | 0.0945 | +11.9% | **+16.3%** | **+30.8%** | **10/12** |
| **`y_mse`** (Original Mass Strandings) | 272 (3.5%) | 0.0475 | +142.9% | **+89.5%** | +73.3% | 8/12 |
| **`y_live_odont_mse`** (Clean Live Pelagic MSEs) | 214 (2.8%) | 0.0399 | +131.5% | **+124.3%** | +49.0% (`+60.7%` at C=0.001) | 7/12 |

   - **Why This Matters:**
     - In the original capstone, anomaly features *always* underperformed raw features on `y_any` (`-4.5%` pooled, `+3.1%` yearly mean) because `67.8%` of `y_any` days were decomposed carcasses (`y_decomp_drift`), where same-day weather anomalies have **negative lift (`-4.4%` pooled)**.
     - Once we remove the carcasses and solitary sick/coastal animals (`y_live_odont_any` — which still includes **both single and mass strandings** of healthy live/fresh pelagic odontocetes!), `Logistic Regression (anomaly, C=0.01)` jumps 10-fold in yearly mean lift from **`+3.1%` to `+30.8%`**, flips pooled lift from **`-4.5%` to `+16.3%`**, beats the seasonal baseline in **`10 out of 12` test years**, and **anomalies (`+30.8%` yearly / `+16.3%` pooled) outperform raw features (`+11.8%` yearly / `+11.9%` pooled)**!
     - On `y_live_odont_mse`, the pure anomaly model's pooled lift over `month-only` surges from **`+89.5%` (on noisy `y_mse`) to `+124.3%`**.

2. **Experiment B — Cross-Zone Thermal Gradients & Wind × Tide Synergy Add Further Lift (`exp_b_gradient_synergy_ablation.csv`):**
   - Adding the 10 engineered cross-zone gradient and spring-tide synergy features (`123` $\rightarrow$ `133` predictors):
     - On **`y_live_odont_any` (Healthy Live Pelagic, Single + MSE)**: boosts raw pooled lift from **`+11.9%` to `+20.3%`** (`0.1058` $\rightarrow$ `0.1137`) and anomaly pooled lift from **`+16.3%` to `+20.8%`** (`0.1100` $\rightarrow$ `0.1142`, winning `10/12` years with `+30.7%` yearly lift).
     - On **`y_live_odont_mse` (Clean Live Pelagic MSEs)**: boosts raw pooled PR-AUC lift from **`+131.5%` to `+164.3%`** (`0.0923` $\rightarrow$ `0.1053`, and yearly mean lift from `+49.4%` to **`+63.7%`**, winning `8/12` years) and boosts anomaly pooled lift from **`+124.3%` to `+135.7%`** (yearly mean lift `+49.0%` $\rightarrow$ **`+60.7%`**).

3. **Experiment C — Permutation Importance Confirms the Physical Mechanism Shift (`02_permutation_importance_comparison.png`):**
   - Shuffling one feature group at a time across the 12 test folds reveals what each model actually learned:
     - **Decomposed Carcass Drift (`y_decomp_drift`):** Calendar **`season` accounts for a `32.7%` drop in PR-AUC**, while every single wind and current variable drops PR-AUC by `< 2%`. The model is purely guessing the spring/summer carcass season.
     - **Healthy Live Pelagic All-Events (`y_live_odont_any`):** **`outer_current_v` (`9.1%` drop)** and **`sound_current_v` (`6.8%` drop)** rise to match calendar `season` (`9.2%`), followed by `sound_sst` (`6.1%`), `outer_mld` (`5.8%`), and `bay_current_v` (`3.0%`).
     - **Clean Live Pelagic MSEs (`y_live_odont_mse`):** Physical transport and tidal-trap variables overtake calendar `season` (`35.9%`) as the #1 and #2 drivers: **`outer_current_v` (`38.1%` drop)** and **`sound_current_v` (`37.3%` drop)** lead, alongside **`bay_tide_spring_neap` (`33.6%`)**, **`bay_wind_u` (`33.3%`)**, **`bay_wind_v` (`28.7%`)**, and **`bay_wind_stress_u` (`28.5%`)**.

4. **Section 4 — The 2021–2025 Untouched Holdout Breakthrough (`holdout_2021_2025_comparison.csv`):**
   - Training once on `2000–2020` (`7,671` days) and evaluating once on the untouched `2021–2025` holdout (`1,826` days) with 1,000-iteration block-bootstrap confidence intervals (preserving 14-day weather blocks):
     - **Original `y_any` (`Raw C=0.001` & `Anomaly C=0.01`):** Lost to the `month-only` baseline on the holdout (`-0.9%` and `-2.0%` pooled lift; bootstrap probability of positive lift was only `40.5%` and `26.6%`).
     - **Clean Healthy Live Pelagic All-Events (`y_live_odont_any`, Single + MSE, 133 features):**
       - **Raw (`C=0.01`):** Achieves **`+22.5%` pooled PR-AUC lift (`0.1397` $\rightarrow$ `0.1712`)** and **`+25.6%` yearly mean lift (`0.1578` $\rightarrow$ `0.1981`)**, with a **95% bootstrap confidence interval of `[+2.3%, +47.6%]` (`P(lift > 0) = 98.8%`)**!
       - **Anomaly (`C=0.01`):** Achieves **`+12.3%` pooled lift (`0.1397` $\rightarrow$ `0.1569`)** and **`+16.3%` yearly mean lift** (`P(lift > 0) = 91.1%`).
     - **Mass Stranding Events (`y_mse` vs. `y_live_odont_mse`):**
       - In the original `mse_only_research`, `y_mse` on the `2021–2025` holdout only achieved **`+5.3%` pooled lift** (`0.0832` $\rightarrow$ `0.0876`, `P(lift > 0) = 70.0%`) and `+13.1%` yearly mean lift (`3/5` years) because of the `18.3%` carcass-aftermath contamination.
       - On our de-confounded **`y_live_odont_mse` (`Raw C=0.3` with the 10 gradient/synergy features)**, 2021–2025 holdout performance jumps to **`+31.9%` pooled lift (`0.0653` $\rightarrow$ `0.0861`, `P(lift > 0) = 94.0%`)** and **`+49.4%` yearly mean lift (`0.0796` $\rightarrow$ `0.1190`, winning `4 of 5` holdout years)**!

**Conclusion of Stage 1:**
The **Label Contamination Hypothesis** is decisively confirmed in both 12-fold cross-validation (`2009–2020`) and on the untouched 5-year holdout (`2021–2025`). Separating live/fresh pelagic odontocete entrapment (`693` days) from passive decomposed carcass drift (`1,472` days) unlocks statistically significant predictive lift (`+22.5%` holdout lift on all live/fresh events, `+31.9%` pooled / `+49.4%` yearly mean holdout lift on clean MSEs) and sets up **Stage 2 (`STAGE2_REVERSE_DRIFT_ROADMAP.md`)** to backtrack those `1,472` decomposed carcass days and `379` human-interaction cases using reverse ocean drift modeling.


---

## 2026-10-23 — Stage 1 Experiment D: Timescale Split — 30-Day Preconditioning (Zellar et al.) vs. 0–7 Day Acute Trap (`02_clean_target_modeling.ipynb`)

**What:** Added Section 5 (Experiment D) to [`02_clean_target_modeling.ipynb`](stage1_clean_entrapment/notebooks/02_clean_target_modeling.ipynb), splitting our 136 environmental predictors into two time horizons:
1. **Long-Term Preconditioning (`30d` window, 42 predictors):** All 30-day rolling means (`_mean_30d`), testing the ~1-month lag reported by Zellar et al. (2021) and Warlick et al. (2022).
2. **Short-Term Acute Trap (`0–7d + Tides`, 94 predictors):** All 3-day and 7-day rolling means (`_mean_3d`, `_mean_7d`), 7-day changes (`_diff_7d`), and same-day astronomical tides (`bay_tide_spring_neap`, `bay_tide_range`).

Saved [`exp_d_timescale_ablation.csv`](stage1_clean_entrapment/data/exp_d_timescale_ablation.csv), [`exp_d_timescale_permutation.csv`](stage1_clean_entrapment/data/exp_d_timescale_permutation.csv), and [`03_timescale_precondition_vs_acute.png`](stage1_clean_entrapment/figures/03_timescale_precondition_vs_acute.png).

**Key Findings:**
1. **30-Day Preconditioning Predicts *Which Years/Months* Are High-Risk, While 0–7 Day Acute Features Predict *Which Exact Days* Animals Strand:**
   - On **`y_live_odont_any` (Healthy Live Pelagic, Single + MSE)**:
     - **30-Day Precondition Only (`42` cols):** Achieves **`+31.2%` year-by-year mean lift (`+21.9%` pooled, `8/12` years)**.
     - **0–7 Day Acute Trap Only (`94` cols):** Achieves **`+30.3%` pooled lift (`0.0945` $\rightarrow$ `0.1232`)** and wins **`9 of 12` test years**.
   - On **`y_live_odont_mse` (Clean Live Pelagic MSEs)**:
     - Combining **Both Timescales (`Full 136` cols)** achieves the highest year-by-year mean lift (**`+63.7%`, winning `8/12` years**) alongside **`+164.3%` pooled lift**.
2. **Variable $\times$ Timescale Permutation Importance (`03_timescale_precondition_vs_acute.png`):**
   - **Macro Timescale Drop:** On `y_live_odont_any`, shuffling all `0–7d Acute Trap` features drops PR-AUC by **`26.2%`** (vs. `3.0%` for `30d` and `0.8%` for `season`). On `y_live_odont_mse`, shuffling `0–7d Acute Trap` drops PR-AUC by **`47.5%`**, while shuffling `30d Precondition` drops PR-AUC by **`36.8%`**.
   - **#1 Single Feature $\times$ Timescale Across Both Targets:** **`grad_bay_sound_sst (30d)`** (`+7.8%` drop on `y_live_odont_any`, **`+45.9%` drop on `y_live_odont_mse`**—even beating `season` at `45.3%`!). A 30-day thermal contrast between Cape Cod Bay and Nantucket Sound sets up the habitat/prey boundary that draws offshore herds into the region.
   - **Acute 3–7 Day Triggers (`3–7d`):** Right behind the 30-day thermal setup, every other top-10 feature is a **3-to-7-day acute trigger**: `outer_current_v (3-7d)` (`+7.0%`), `bay_air_temp (3-7d)` (`+5.9%` on `y_live_odont_any`, **`+43.3%` on `y_live_odont_mse`**), `sound_current_v (3-7d)` (**`+36.3%`** on `y_live_odont_mse`), and **`syn_bay_wind_v_tide (3-7d)` (`+36.0%` on `y_live_odont_mse`)**—our engineered interaction between north/onshore wind and the Wellfleet Spring/Neap tidal cycle.


---

## 2026-10-24 — Stage 1 Notebook 03: Bottom-Up Trophic Cascade & The Two-Expert (`Physics × Prey`) Model (`03_trophic_prey_integration.ipynb`)

**What:** Tested Jake's hypothesis that lower- and mid-trophic-level food web dynamics (phytoplankton $\rightarrow$ zooplankton $\rightarrow$ small forage fish/squid) act as a bottom-up signal drawing pelagic dolphins into Cape Cod Bay. Pulled 26 years (`1999–2025`, `9,558` days) of daily, cloud-free `0.083°` (~8 km) biological fields from Copernicus Marine's **Global Ocean Low and Mid-Trophic Levels Biomass Hindcast (`cmems_mod_glo_bgc_my_0.083deg-lmtl_P1D-i` — SEAPODYM)** across `bay`, `outer`, and `sound` ([`cmems_lmtl_prey_daily_1999_2025.parquet`](stage1_clean_entrapment/data/cmems_lmtl_prey_daily_1999_2025.parquet)). Built, executed, and validated [`03_trophic_prey_integration.ipynb`](stage1_clean_entrapment/notebooks/03_trophic_prey_integration.ipynb), saving [`exp_e_trophic_prey_cv.csv`](stage1_clean_entrapment/data/exp_e_trophic_prey_cv.csv), [`exp_e_trophic_prey_holdout.csv`](stage1_clean_entrapment/data/exp_e_trophic_prey_holdout.csv), and [`04_trophic_prey_and_ensemble.png`](stage1_clean_entrapment/figures/04_trophic_prey_and_ensemble.png).

**Why SEAPODYM Solved the Capstone's Chlorophyll Gap:**
- In the original capstone, satellite `MODIS Aqua` Chlorophyll (`chl`) failed the `>=95%` completeness threshold because winter storm clouds blocked ~72% of raw daily observations and `MODIS Aqua` didn't launch until July 2002.
- The `CMEMS SEAPODYM` reanalysis (`0.083°`) is **100.0% complete across all 9,497 days (`2000–2025`)** and models four levels of the food web directly:
  1. `npp` — Net Primary Production (`[mg m⁻² day⁻¹]`, phytoplankton growth rate)
  2. `zooc` — Zooplankton wet biomass (`[g m⁻²]`, copepods/krill)
  3. `mnkc_epi` — **Epipelagic Micronekton wet biomass (`[g m⁻²]`, small upper-ocean forage fish and juvenile squid—the direct prey of Cape Cod dolphins!)**
  4. `zeu` — Euphotic zone depth (`[m]`, how deep sunlight penetrates—a direct measure of water clarity vs. turbidity)
  5. Cross-zone prey gradients (`grad_bay_outer_mnkc_epi`, `grad_bay_sound_mnkc_epi`, `grad_bay_outer_zooc`, `grad_bay_sound_zooc`).

**Data Science Lesson — Avoiding Multicollinearity with a Two-Expert (`Physics × Prey`) Ensemble:**
- Because plankton and forage fish (`48` lagged features) are physically driven by temperature and mixing depth (`136` physical features), concatenating all `184` features into a single Logistic Regression causes **multicollinearity (feature crowding)** on sparse MSE targets.
- Instead, we trained **two specialized experts**—**Expert 1 (`P_Physics`, 136 features)** for the physical trap and **Expert 2 (`P_Prey`, 48 features)** for trophic food-web attraction—and combined them via their joint probability product:
  $$\text{Risk}_{\text{Joint}}(t) = P_{\text{Physics}}(t) \times P_{\text{Prey}}(t)$$

**Key Results (`exp_e_trophic_prey_cv.csv` & `exp_e_trophic_prey_holdout.csv`):**
1. **Even Standalone, the Trophic/Prey Expert (`P_Prey`) Strongly Beats the Seasonal Baseline:**
   - Without a single wind, current, SST, or tide variable, `Expert 2: Trophic/Prey Only` beats `month-only` in 12-fold CV (`2009–2020`) by **`+28.4%` year-by-year on `y_live_odont_any` (winning `10/12` years)** and by **`+115.5%` year-by-year on `y_live_odont_mse` (`anomaly`, winning `9/12` years)**—and on the untouched **2021–2025 holdout**, `Prey Only [anomaly]` beats `month-only` by **`+39.1%` pooled on `y_live_odont_mse` (winning `4/5` holdout years)**!
2. **The Two-Expert Joint Model (`Physics × Prey`) Achieves Our Best Results Overall:**
   - **On `y_live_odont_any` (Healthy Live Pelagic: Single + MSE):**
     - **12-Fold CV (`2009–2020`, Anomaly):** Jumps from `+20.8%` pooled / `+30.7%` yearly mean (`Physics Only`) to **`+26.9%` pooled (`0.1199`) and `+38.2%` yearly mean (`0.1461`), winning `10/12` test years**!
     - **2021–2025 Untouched Holdout (Anomaly):** Jumps from `+14.8%` pooled (`Physics Only`) and `+14.4%` (`Prey Only`) to **`+33.2%` pooled lift (`0.1397` $\rightarrow$ `0.1861`, `+27.7%` yearly mean lift)** with a **95% bootstrap confidence interval of `[+12.4%, +60.1%]` and `P(Lift > 0) = 100.0%` (`1,000 of 1,000` bootstrap resamples positive!)**.
   - **On `y_live_odont_mse` (Clean Mass Strandings):**
     - **12-Fold CV (`2009–2020`, Raw):** Surges from `+164.3%` pooled / `+63.7%` yearly mean (`Physics Only`) to **`+188.0%` pooled lift (`0.0399` $\rightarrow$ `0.1148`) and `+80.7%` yearly mean lift** (and on `Anomaly`, jumps from `+135.7%` to **`+153.2%` pooled / `+97.4%` yearly mean lift**).
     - **2021–2025 Untouched Holdout (Raw):** Achieves **`+30.2%` pooled lift (`+32.4%` yearly mean, winning `4/5` holdout years)** and tightens the **95% bootstrap confidence interval to strictly positive: `[+6.2%, +73.4%]` (`P(Lift > 0) = 99.3%`)**!
3. **How the Food Web Drives Risk (`04_trophic_prey_and_ensemble.png`, Panel B):**
   - Inspecting the standardized anomaly weights of the Prey Expert reveals a clear ecological mechanism: stranding risk rises when **30-day Primary Production inside Cape Cod Bay (`bay_npp_mean_30d_anom`, `+0.116`)** is elevated, **Zooplankton is more concentrated inside Cape Cod Bay than offshore (`grad_bay_outer_zooc_mean_30d_anom` `+0.085` & `7d` `+0.080`, paired with negative `outer_zooc_mean_7d_anom` `-0.082`)**, and **Forage Fish / Micronekton (`outer_mnkc_epi_mean_3d_anom`, `+0.091`)** surges at the entrance of the Cape.


---

## 2026-10-24 — Stage 2 Notebook 01: Drifter Cohort Inventory, Human Interactions & "Trap vs. Collector" Beaches (`01_drifter_cohort_and_hi_audit.ipynb`)

**What:** Built, executed, and validated [`stage2_reverse_drift/notebooks/01_drifter_cohort_and_hi_audit.ipynb`](stage2_reverse_drift/notebooks/01_drifter_cohort_and_hi_audit.ipynb). Loaded the full uncleaned `4,831`-row NOAA cetacean stranding dataset (`2000–2025`), restoring the **`379` confirmed `Human Interaction == 'Y'`** cases and **`99` offshore `County == 'EEZ'`** floating carcass sightings that were excluded in Stage 1. Cleaned all coordinates (fixing manual DDMM.mmm and digit typos in `NE-2004-1042007` and `NE-2019-1212978`, imputing town centroids for `33` rows missing raw lat/lon, and separating **Bay-side (`Cape Cod Bay`)** vs. **Sound-side (`Nantucket Sound / Buzzards Bay`)** coastlines for dual-coastline Cape Cod towns like Barnstable, Yarmouth, and Bourne). Saved [`stage2_all_records_clean.parquet`](stage2_reverse_drift/data/stage2_all_records_clean.parquet), [`stage2_drifter_cohorts.parquet`](stage2_reverse_drift/data/stage2_drifter_cohorts.parquet), [`collector_vs_trap_beaches.csv`](stage2_reverse_drift/data/collector_vs_trap_beaches.csv), [`cohort_summary_table.csv`](stage2_reverse_drift/data/cohort_summary_table.csv), and Figure 1 ([`01_drifter_cohorts_and_collector_beaches.png`](stage2_reverse_drift/figures/01_drifter_cohorts_and_collector_beaches.png)).

**Key Findings:**

1. **Active "Trap Beaches" vs. Passive "Collector Beaches" (`collector_vs_trap_beaches.csv`):**
   - Measuring the **Collector Index** (`% Decomposed` vs. `% Alive` at initial discovery across towns with $\ge 30$ strandings) cleanly splits the Massachusetts coastline into two functional regimes:
     - **Active Swimmer Traps (`% Alive >= 45%`):** Found exclusively on the wide, shallow mudflats of the southeastern Cape Cod Bay hook: **Wellfleet (`59.8%` Alive, `28.4%` Decomposed, `n = 1,889`)**, **Dennis (`52.1%` Alive, `29.9%` Decomposed, `n = 194`)**, and **Brewster (`46.5%` Alive, `28.0%` Decomposed, `n = 411`)**.
     - **Passive Carcass Collectors (`% Decomposed >= 57%`, `% Total Drifters >= 66%`):** Found on open coastlines facing long wind/current fetch: **Hull (`83.3%` Decomposed, `93.3%` Total Drifter)**, **Sandwich (`78.3%` Decomposed, `84.8%` Total Drifter)**, **Westport (`71.1%` Decomposed)**, **Plymouth (`66.7%` Decomposed)**, **Edgartown / Martha's Vineyard (`63.6%` Decomposed)**, **Duxbury (`60.0%` Decomposed)**, **Barnstable (`59.2%` Decomposed)**, **Truro (`59.2%` Decomposed)**, **Gloucester (`58.5%` Decomposed)**, and **Nantucket (`57.8%` Decomposed)**.
     - *Why Sandwich (`78.3%` Decomposed) vs. Wellfleet (`59.8%` Alive) is such a striking proof of concept:* Both towns sit inside Southern/Eastern Cape Cod Bay just 30 km apart! Wellfleet has the shallow 3.4-meter tidal flats that trap live swimming pods, whereas Sandwich sits at the southwestern bottom of Cape Cod Bay facing directly into winter/spring north winds that push floating carcasses southward into the Sandwich shoreline.

2. **Species Guild & Seasonal Asymmetry (`cohort_summary_table.csv`):**
   - **Harbor Porpoises (*Phocoena phocoena*, `n = 802` total, `618` coastal drifters):** **`69.0%` arrive Moderately/Severely Decomposed** and another **`13.0%` arrive as Fresh Dead singletons (`82.0%` dead on arrival; only `17.2%` alive)**. Furthermore, **`59.2%` of all dead Harbor Porpoises strand in Spring (`March–May`, with March alone accounting for `33.7%` of the entire year)** and `29.6%` in Winter (`Dec–Feb` = **`88.8%` in Dec–May**).
   - **Baleen & Large Whales (`n = 352` total):** **`72.4%` Decomposed + `8.0%` Fresh Dead Singleton = `80.4%` Dead Drifters**.
   - **Offshore `EEZ` Floating Sightings (`n = 99`):** **`80.8%` Decomposed + `10.1%` Fresh Dead = `90.9%` Dead Floating Carcasses** spotted at sea, providing a true independent at-sea validation dataset.


---

## 2026-10-24 — Stage 2 Notebook 02: Lagrangian Reverse-Drift Engine, `EEZ` Ground-Truth Validation & At-Sea Mortality Hotspots (`02_lagrangian_backtracker_and_hotspots.ipynb`)

**What:** Pulled a 26-year (`1999-12-01` to `2025-12-31`, `9,528` days) wide-domain `0.083°` (~8 km) `CMEMS GLORYS12V1` daily surface current (`uo`, `vo`) and sea-surface temperature (`thetao`) dataset covering **`40.2°N–43.2°N, -71.6°W–-68.5°W`** ([`glorys_wide_surface_1999_2025.nc`](stage2_reverse_drift/data/glorys_wide_surface_1999_2025.nc)). Built and executed a vectorized **Monte Carlo Lagrangian Reverse-Drift Particle Engine** in [`02_lagrangian_backtracker_and_hotspots.ipynb`](stage2_reverse_drift/notebooks/02_lagrangian_backtracker_and_hotspots.ipynb), launching **$M = 50$ virtual particles per carcass across all `2,282` coastal drifters (`114,100` reverse trajectories stepped at $\Delta t = 3\text{ hours}$)** with shoreline barrier reflection (`matplotlib.path.Path` polygons for Cape Cod, Martha's Vineyard, and Nantucket), wind leeway ($\alpha_{\text{leeway}} \sim \text{Uniform}(1.5\%, 3.5\%)$), leeway divergence ($\theta_{\text{div}} \sim \mathcal{N}(0^\circ, 8^\circ)$), turbulent eddy diffusion ($K_h = 20\text{ m}^2/\text{s}$), and Arrhenius/degree-day SST-scaled drift durations ($f(T_{\text{water}}) = \exp[-0.055(T_{\text{water}} - 10^\circ\text{C})]$).

Saved [`backtracked_drifters_summary.parquet`](stage2_reverse_drift/data/backtracked_drifters_summary.parquet), [`cohort_basin_attribution.csv`](stage2_reverse_drift/data/cohort_basin_attribution.csv), [`source_receptor_matrix.csv`](stage2_reverse_drift/data/source_receptor_matrix.csv), [`sensitivity_analysis_table.csv`](stage2_reverse_drift/data/sensitivity_analysis_table.csv), [`seasonal_hotspot_shift.csv`](stage2_reverse_drift/data/seasonal_hotspot_shift.csv), and Figures 2 and 3 ([`02_reverse_trajectories_and_validation.png`](stage2_reverse_drift/figures/02_reverse_trajectories_and_validation.png) and [`03_mortality_hotspot_kde_maps_by_cohort.png`](stage2_reverse_drift/figures/03_mortality_hotspot_kde_maps_by_cohort.png)).

**Key Breakthroughs:**

1. **Independent Ground-Truth Validation Against Offshore `EEZ` Floating Sightings (`02_reverse_trajectories_and_validation.png`, Panel B):**
   - Evaluating our backtracked whale and human-interaction mortality Kernel Density Estimate (KDE) at the exact offshore coordinates where **`77` independent `County == 'EEZ'` floating carcasses** were sighted at sea shows a **`1.96x` mean density enrichment and `4.59x` median density enrichment over random ocean grid cells (`p < 0.0005` across `2,000` Monte Carlo permutations)**!
   - Rewinding carcasses from the beach without ever showing the model the `EEZ` data reconstructs the exact offshore corridors (Stellwagen Bank, Great South Channel approach, and Southern Island Shelf) where floating carcasses are spotted at sea.

2. **The Source–Receptor Matrix (`source_receptor_matrix.csv` & Panel C):**
   - **More than half of the decomposed carcasses landing inside Cape Cod Bay actually died *outside* Cape Cod Bay:**
     - On **`1. Inner Cape Hook (Wellfleet / Brewster / Eastham)`**, **`51.0%` of decomposed carcasses originated north of Cape Cod Bay** (**`43.3%` on Stellwagen Bank & Massachusetts Bay** + `7.7%` on the North Shore / Jeffreys Ledge), while `43.1%` died inside Cape Cod Bay and `5.9%` on the Outer Cape Shelf.
     - On **`2. Southern Cape Cod Bay (Sandwich / North Barnstable / North Yarmouth)`**, **`54.6%` originated north of Cape Cod Bay** (**`39.5%` on Stellwagen Bank & Mass Bay** + **`15.1%` on North Shore / Jeffreys Ledge**), blown southward across the mouth of Cape Cod Bay by winter/spring northerly winds!
     - On **`3. Outer Cape & Provincetown Spit`**, **`36.2%` originated on Stellwagen Bank & Mass Bay**, **`20.7%` on North Shore / Jeffreys Ledge**, `22.4%` in Cape Cod Bay, and `14.8%` on the Outer Cape / Nauset Shelf.
     - On **`4. Southern Islands, Sound & Buzzards Bay` (`Martha's Vineyard`, `Nantucket`, `Falmouth`, `Westport`, South `Barnstable/Yarmouth`)**, **`83.7%` originated in the Southern Shelf & Sounds** and **`7.3%` in the Great South Channel / Nantucket Shoals (`91.0%` combined southern shelf)**, completely isolated from Cape Cod Bay (`1.5%`).

3. **Hidden Offshore Mortality Hotspots by Cohort (`cohort_basin_attribution.csv` & `03_mortality_hotspot_kde_maps_by_cohort.png`):**
   - **Harbor Porpoise Gillnet Corridor (`n = 618` carcasses, mean SST `6.2°C`, mean drift `7.4 days` / `48.0 km`):**
     - **`43.2%` backtrack to Stellwagen Bank & Massachusetts Bay**, **`24.1%` to Cape Cod Bay**, and **`22.8%` to the North Shore & Jeffreys Ledge**—meaning **`90.1%` of all dead Harbor Porpoises originate in the northern Massachusetts Bay / Stellwagen Bank / Jeffreys Ledge / Cape Cod Bay shelf system** (where winter/spring sink-gillnet fisheries operate), even when their carcasses wash ashore 45–60 km south on Cape Cod beaches!
   - **Confirmed Fishery Entanglements (`n = 112`, mean drift `3.7 days` / `34.2 km`):**
     - Cluster in the primary fixed-gear (lobster/pot and gillnet) grounds: **`28.6%` in Cape Cod Bay**, **`24.1%` on Stellwagen Bank & Mass Bay (`52.7%` combined)**, **`18.8%` in the Southern Shelf / Sounds**, **`12.5%` on the Outer Cape / Nauset Shelf**, and **`11.6%` on the North Shore / Jeffreys Ledge**.
   - **Confirmed Vessel Strikes (`n = 23`, mean drift `4.3 days` / `36.2 km`):**
     - Map directly onto the **two major commercial vessel approaches in Massachusetts**:
       1. **`43.4%` along the Boston Harbor Traffic Separation Scheme (TSS)** (`30.4%` Stellwagen Bank & Mass Bay + `13.0%` Outer Cape TSS approach), and
       2. **`34.8%` in the Southern Shelf / Buzzards Bay–Cape Cod Canal & Island Ferry corridors** (`+ 13.0%` Cape Cod Bay).

4. **Physics Sensitivity Analysis (`sensitivity_analysis_table.csv` & Figure 2 Panel D):**
   - Comparing `Ocean Currents Only (0% windage)` (`30.4 km` mean drift) against realistic `1.5%–3.5% Wind Leeway` (`41.4–52.8 km` mean drift) demonstrates that **wind leeway is essential for transporting offshore carcasses from Stellwagen Bank (`35.6%` vs. `23.8%`) across coastal shear fronts onto Cape Cod collector beaches**, while our headline ecological finding—that **`88.5% to 90.1%` of Harbor Porpoise mortality originates on the northern shelf**—remains invariant across every single parameter configuration tested.

