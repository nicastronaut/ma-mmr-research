# Stage 2 Roadmap: Reverse-Drift Backtracking of Dead & Bycaught Cetaceans in Massachusetts

## 1. The Big Idea (Plain-English Overview)

When a dolphin, porpoise, or whale strands **alive** on Cape Cod (especially in a mass stranding in Wellfleet), the animal swam there under its own power and got trapped by the hook-shaped coastline and big tidal swings.

When a **moderately or severely decomposed carcass** washes up on a beach, something completely different happened:
- The animal **died at sea** days or weeks earlier (from gillnet entanglement, a vessel strike, or illness).
- Once it died, the floating carcass acted like a passive ocean drifter, pushed along by **surface ocean currents** and **wind** until it happened to hit land.

In the original capstone, those washed-up carcasses were mixed in with live strandings, which added noise because the weather on the day someone finds a rotting carcass on the beach isn't what killed or trapped the animal.

**The Goal of Stage 2:** Instead of throwing away the ~2,000+ decomposed carcasses and ~380 human-interaction records in your 2000–2025 NOAA dataset, we flip the problem around. Starting from the **beach coordinates and discovery date**, we rewind the historical ocean currents and winds **backward in time** to estimate **where and when the animal actually died at sea**.

### What Real Conservation Problems Does This Solve?
1. **Finding Hidden Fishery Bycatch Hotspots:** Many animals (especially Harbor Porpoises and Common Dolphins) die in commercial fishing gear offshore, are cut loose or fall out of nets, and wash ashore days later. Backtracking those carcasses maps the offshore "mortality zones" even when bycatch goes unreported.
2. **Separating "Collector Beaches" from "Trap Beaches":** Some beaches (like Nantucket, Martha's Vineyard, or the Outer Cape) may collect carcasses simply because offshore currents point at them, whereas inner Cape Cod Bay traps live swimmers.
3. **Connecting Offshore Shipping & Fishing Data to Beach Strandings:** Once we generate "probability clouds" of where carcasses originated at sea, we can overlay them against shipping lanes (e.g., Boston Harbor / Stellwagen Bank approaches) and commercial fishing effort maps.

---

## 2. What Data We Already Have in Your NOAA Extract

From your 4,831-row NOAA dataset (`2000-2025 MA Cetaceans.csv`), here are the exact groups of animals that belong in Stage 2:

| Cohort / Filter | Record Count | Why It Matters for Reverse-Drift |
| :--- | :--- | :--- |
| **`Moderate Decomposition`** | **1,479 animals** | Primary drifter group: typically dead ~2 to 10 days before beaching (depending on water temperature). |
| **`Advanced Decomposition`** | **467 animals** | Longer-term drifters: typically dead ~10 to 25 days before beaching. |
| **`Mummified/Skeletal`** | **103 animals** | Very old remains (usually excluded from drift modeling because the time of death is too uncertain). |
| **`Fresh Dead` (Single / Non-MSE)** | **~450+ animals** | Recently deceased (typically <24–48 hours of drift); gives the tightest, most accurate short-range backtrack. |
| **Confirmed `Human Interaction == Y`** | **379 animals** | Includes **104 `Fishery Interaction == Y`**, **34 `Boat Collision == Y`**, and **2 `Shot == Y`**—dropped from the capstone, but gold-standard cases for backtracking human-caused mortality. |
| **`Entangled OD` Populated** | **1,005 animals** | Records carrying entanglement observation details. |
| **Offshore `County == EEZ`** | **99 animals** | Animals spotted floating at sea (mostly Humpback, Minke, Fin, and Right Whales)—dropped from the capstone, useful both as offshore mortality points and for validating drift tracks. |
| **Harbor Porpoises (`Phocoena phocoena`)** | **797 animals** | Solitary coastal/shelf species heavily impacted by sink-gillnet bycatch in New England; mostly individual dead strandings rather than live mass strandings. |

---

## 3. How the Physics & Data Science Work (With Plain-English Explainers)

### Key Concept 1: Lagrangian Particle Tracking
> **Explainer:** In oceanography, *Eulerian* data measures water flowing past a fixed point (like a buoy anchored in one spot). *Lagrangian* tracking follows a moving object (like a floating bottle, a sea turtle, or a whale carcass) as it travels across the grid over time.

At any hour $t$, a floating carcass at position $\mathbf{x} = (\text{lat}, \text{lon})$ moves at velocity:

$$\mathbf{v}_{\text{carcass}}(\mathbf{x}, t) = \mathbf{u}_{\text{current}}(\mathbf{x}, t) + \alpha \cdot \mathbf{W}_{10\text{m}}(\mathbf{x}, t) + \boldsymbol{\epsilon}(t)$$

- **$\mathbf{u}_{\text{current}}(\mathbf{x}, t)$ — Surface Ocean Current:** About 90% of a cetacean carcass is underwater, so the top layer of the ocean current carries it directly.
- **$\alpha \cdot \mathbf{W}_{10\text{m}}(\mathbf{x}, t)$ — Windage (Leeway):** About 10% of a bloated carcass sticks above the water surface and catches the wind (plus surface wave push, known as *Stokes drift*). Field studies on dolphin carcasses (such as the European *MOTHY* drift model by Peltier et al.) show that carcasses drift at roughly **$\alpha = 1.5\%$ to $3.5\%$ ($0.015\text{–}0.035$) of the 10-meter wind speed**.
- **$\boldsymbol{\epsilon}(t)$ — Random Turbulent Diffusion (Monte Carlo Uncertainty):** Because ocean grids smooth out small eddies and we don't know the exact windage $\alpha$ of each individual animal, we don't just trace a single line backward. Instead, we use a **Monte Carlo simulation**: we drop **500 to 1,000 virtual particles** at the beach stranding spot, give each particle a slightly different windage $\alpha$ and small random jitter, and step them backward in time.

### Key Concept 2: Temperature-Adjusted Drift Window ($\Delta t$)
How far back in time should we rewind the clock? Decomposition speed depends heavily on **water temperature (SST)**—a carcass bloats and decomposes much faster in 20°C August water than in 3°C February water:
- **`Fresh Dead`:** Rewind **6 to 36 hours**.
- **`Moderate Decomposition`:** Rewind **2 to 5 days in warm summer water** vs. **5 to 12 days in cold winter/spring water**.
- **`Advanced Decomposition`:** Rewind **8 to 20+ days** (with wider uncertainty clouds).

### Key Concept 3: Kernel Density Estimation (KDE) for Hotspot Mapping
> **Explainer:** *Kernel Density Estimation (KDE)* is a smoothing technique that turns thousands of individual scattered particle points on a map into a smooth contour "heatmap" showing where the highest concentration of particles came from.

By combining the backtracked particle clouds across all stranded carcasses in a cohort (for example, all 104 confirmed Fishery Interaction cases), KDE reveals the most probable **at-sea mortality hotspots**.

---

## 4. Environmental Datasets: What We Have vs. What We Will Add

### Already on Disk (`data/raw/`)
1. **CMEMS `GLORYS12V1` Daily 2D Surface Currents & Temperature (1999–2025):**
   - You already downloaded 27 yearly NetCDF files (`glorys12v1_daily_1999.nc` to `2025.nc`) covering **`41.0°N–42.5°N, 71.0°W–69.5°W`** at `1/12°` (~8 km) resolution (`302` ocean grid cells).
   - Contains eastward current (`uo`), northward current (`vo`), and surface water temperature (`thetao`).
2. **Continuous Wind Records (1999–2025):**
   - Hourly buoy winds (`44013`, `A01`, `44018`, `44020`, `buzm3`) and `ERA5` hourly 10m wind vectors (`u10`, `v10`).

### Planned Upgrades When We Start Stage 2
1. **Wider Spatial Box for `GLORYS12V1` & Gridded `ERA5` Winds (`40.5°N–43.5°N, 71.5°W–68.5°W`):**
   - A carcass drifting for 7–10 days at `0.15 m/s` can travel ~100–130 km (~1° of latitude/longitude). Expanding the bounding box by ~1° in each direction (a ~5-minute pull using your existing `copernicusmarine` setup) ensures trajectories into the Gulf of Maine, Stellwagen Bank, Great South Channel, and Georges Bank never hit the edge of the map.
2. **Optional High-Resolution Coastal Tides via `FVCOM` (UMass Dartmouth / NERACOOS) or `OpenDrift`:**
   - Inside Cape Cod Bay and Nantucket Sound, tidal currents slosh back and forth every 6 hours. Adding hourly tidal current vectors (or using the open-source Python ocean trajectory library `opendrift` / a custom Runge-Kutta 4th-order integrator with land-masking) will prevent virtual particles from crossing land spits like Provincetown or Monomoy.

---

## 5. Step-by-Step Implementation Plan for Stage 2

- **Notebook `01_drifter_cohort_inventory.ipynb`:**
  - Load raw `2000-2025 MA Cetaceans.csv` (including the `Human Interaction == Y` and `EEZ` rows that were dropped in Stage 1).
  - Clean coordinates and categorize every dead stranding into drift cohorts: (A) Confirmed Fishery / Entanglement, (B) Confirmed Vessel Strike, (C) Harbor Porpoise Dead Strandings, (D) Pelagic Dolphin Dead Single Strandings, and (E) Large Whale Carcasses.
  - Map where decomposed carcasses wash ashore compared to where live strandings happen.
- **Notebook `02_drift_engine_and_validation.ipynb`:**
  - Build the backward-in-time particle tracker (`u_current + alpha * W_10m`) with land-masking (so particles stop if they hit a coastline boundary) and SST-scaled drift durations.
  - Validate the physics on known test cases (e.g., multi-day stranding events or carcasses spotted offshore in `EEZ` before washing ashore).
- **Notebook `03_mortality_hotspot_mapping.ipynb`:**
  - Run Monte Carlo ensembles across the 2000–2025 drifter cohorts.
  - Generate At-Sea Mortality Origin Heatmaps for Massachusetts Bay, Cape Cod Bay, Outer Cape/Nauset, and Nantucket Sound, broken down by season and species/human-interaction type.
