"""
Stage 2 Reverse-Drift Engine — v2 (real-coastline land handling)
================================================================

What changed relative to v1 (notebooks/02_lagrangian_backtracker_and_hotspots.ipynb):

1. LAND BOUNDARY = U.S. Census 1:500k coastline (cb_2023_us_county_500k), rasterized at
   0.001° (~111 m N–S x ~83 m E–W) with a conservative 50 m landward buffer. This replaces the
   GLORYS 8-km land mask + hand-drawn Cape Cod / MV / Nantucket polygons + the hard-coded
   "Cape Cod Bay is water" box.
2. FULL-SEGMENT LAND CHECK. Every step's straight-line path is sampled at <= half a raster cell
   spacing, so a particle can no longer jump across a narrow peninsula between checks.
3. SHORTER TIME STEP. dt = 30 min (v1: 3 h).
4. EXPLICIT, SWITCHABLE SHORE-CONTACT RULE (this is an ASSUMPTION, so we run more than one):
     rule='strand' : a backward path that touches shore more than `grace_km` from its own
                     discovery beach is physically inconsistent (the carcass would have beached
                     there first) -> particle is flagged STRANDED_ELSEWHERE and excluded from
                     origin estimates. Contacts within `grace_km` of the discovery beach are
                     treated with the 'hold' behaviour.
     rule='hold'   : on any shore contact the particle stays at its last water position for that
                     step (OpenDrift 'previous' convention) — i.e. carcasses may touch shore and
                     refloat. Nothing is discarded.
5. DOMAIN EXIT is recorded (status EXITED) instead of particles being silently clamped to the
   domain edge.

What did NOT change (kept identical so differences are attributable to land handling):
- GLORYS12V1 daily surface currents (uo, vo) with nearest-wet-cell fill, bilinear in space,
  nearest day in time; same 3-point inverse-distance wind interpolation; same leeway
  U(1.5%, 3.5%), divergence N(0°, 8°), K_h = 20 m²/s; same decomposition windows and SST scaling;
  same random-draw order for per-animal parameters (seeded), so per-particle drift duration,
  leeway and divergence angle match v1 exactly for the same seed.

Known limitations that v2 does NOT fix (must be stated in the paper):
- GLORYS12V1 has no tides and only daily means.
- GLORYS resolves ~8 km; near shore its currents are extrapolated from the nearest wet cell.
- GLORYS has NO land on the Outer Cape (narrower than one cell), so its currents there describe an
  ocean without the Outer Cape. The real-coast boundary stops particles, but the current field
  around the Outer Cape is physically inconsistent with that boundary.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import geopandas as gpd
import shapely
from scipy.ndimage import distance_transform_edt

ROOT = Path('/home/nicastronaut/marine_bio/stage2_reverse_drift')
DATA = ROOT / 'data'
V2_DATA = DATA / 'v2'
V2_DATA.mkdir(exist_ok=True)
ENV_PATH = Path('/home/nicastronaut/champ/capstone/data/processed/environmental_daily.parquet')

# Model domain (same as v1)
LON_MIN, LON_MAX, LAT_MIN, LAT_MAX = -71.55, -68.52, 40.26, 43.15
RES = 0.001          # land raster resolution (deg)
PAD = 0.06           # raster padding beyond domain (deg) so segment samples just outside are valid
BUFFER_M = 50.0      # conservative landward buffer (m)
M_PER_DEG = 111132.0
KM_PER_DEG = M_PER_DEG / 1000.0

STATUS_OK, STATUS_STRANDED, STATUS_EXITED = 0, 1, 2

BASIN_LABELS = [
    '1. Cape Cod Bay',
    '2. Stellwagen Bank & Mass Bay',
    '3. Cape Ann & Jeffreys Ledge (North Shore)',
    '4. Outer Cape & Nauset Shelf',
    '5. Great South Channel & Shoals',
    '6. Southern Shelf & Sounds',
]


# --------------------------------------------------------------------------------------------
# Coastline / land mask
# --------------------------------------------------------------------------------------------
def load_land_geometries():
    """Return (exact_land, buffered_land) shapely geometries in EPSG:4326."""
    g = gpd.read_file(DATA / 'cb_2023_us_county_500k.zip')
    g = g.cx[LON_MIN - 0.3:LON_MAX + 0.3, LAT_MIN - 0.3:LAT_MAX + 0.3]
    exact = g.to_crs('EPSG:4326').union_all()
    buffered = g.to_crs('EPSG:26919').buffer(BUFFER_M).to_crs('EPSG:4326').union_all()
    return exact, buffered


class LandMask:
    """Boolean raster (True = land) built from the buffered 1:500k coastline."""

    def __init__(self, cache=V2_DATA / 'land_mask_0p001deg_buf50m.npz'):
        self.lon0 = LON_MIN - PAD
        self.lat0 = LAT_MIN - PAD
        self.nx = int(round((LON_MAX + PAD - self.lon0) / RES)) + 1
        self.ny = int(round((LAT_MAX + PAD - self.lat0) / RES)) + 1
        cache = Path(cache)
        if cache.exists():
            self.mask = np.load(cache)['mask']
        else:
            _, buffered = load_land_geometries()
            shapely.prepare(buffered)
            lon_c = self.lon0 + RES * np.arange(self.nx)
            mask = np.zeros((self.ny, self.nx), dtype=bool)
            for j in range(self.ny):  # row-by-row keeps memory small
                lat_j = self.lat0 + RES * j
                mask[j] = shapely.contains_xy(buffered, lon_c, np.full(self.nx, lat_j))
            np.savez_compressed(cache, mask=mask)
            self.mask = mask
        assert self.mask.shape == (self.ny, self.nx)

    def idx(self, lon, lat):
        i = np.clip(np.rint((lon - self.lon0) / RES).astype(np.int64), 0, self.nx - 1)
        j = np.clip(np.rint((lat - self.lat0) / RES).astype(np.int64), 0, self.ny - 1)
        return j, i

    def is_land(self, lon, lat):
        j, i = self.idx(lon, lat)
        return self.mask[j, i]

    def cell_center(self, j, i):
        return self.lon0 + RES * i, self.lat0 + RES * j


# --------------------------------------------------------------------------------------------
# Start-point placement: nearest water cell, respecting Bay-side vs Sound-side shoreline
# --------------------------------------------------------------------------------------------
BAY_ZONES = ['1. Inner Cape Hook (Trap Flats)', '2. Bay-Side Upper/Mid Cape (Collector Shore)']
SOUND_ZONES = ['4. Southern Islands, Sound & Buzzards Bay']


def _nearest_water_indices(blocked):
    dy = RES * KM_PER_DEG
    dx = RES * KM_PER_DEG * np.cos(np.radians(42.0))
    _, (jj, ii) = distance_transform_edt(blocked, sampling=(dy, dx), return_indices=True)
    return jj.astype(np.int32), ii.astype(np.int32)


def place_starts(lon, lat, zones, land: LandMask):
    """Move each discovery coordinate to the nearest raster water cell.
    Bay-side animals may not be placed in water south of the Cape; Sound-side animals may not be
    placed in Cape Cod Bay. Returns (lon_start, lat_start, nudge_km)."""
    lon_c = land.lon0 + RES * np.arange(land.nx)
    lat_c = land.lat0 + RES * np.arange(land.ny)
    LON, LAT = np.meshgrid(lon_c, lat_c)
    south_of_cape = (LAT < 41.725) & (LON > -70.66)
    cape_cod_bay = (LAT > 41.735) & (LAT < 42.08) & (LON > -70.56) & (LON < -69.97)
    del LON, LAT

    zones = np.asarray(zones)
    lon_s = np.asarray(lon, float).copy()
    lat_s = np.asarray(lat, float).copy()
    groups = [
        (np.isin(zones, BAY_ZONES), land.mask | south_of_cape),
        (np.isin(zones, SOUND_ZONES), land.mask | cape_cod_bay),
        (~np.isin(zones, BAY_ZONES + SOUND_ZONES), land.mask),
    ]
    for sel, blocked in groups:
        if not sel.any():
            continue
        jj, ii = _nearest_water_indices(blocked)
        j0, i0 = land.idx(lon_s[sel], lat_s[sel])
        on_blocked = blocked[j0, i0]
        jn, in_ = jj[j0, i0], ii[j0, i0]
        new_lon, new_lat = land.cell_center(jn, in_)
        lon_s[sel] = np.where(on_blocked, new_lon, lon_s[sel])
        lat_s[sel] = np.where(on_blocked, new_lat, lat_s[sel])
        del jj, ii
    nudge_km = np.hypot((lat_s - lat) * KM_PER_DEG,
                        (lon_s - lon) * KM_PER_DEG * np.cos(np.radians(lat)))
    assert not land.is_land(lon_s, lat_s).any()
    return lon_s, lat_s, nudge_km


# --------------------------------------------------------------------------------------------
# Forcing (identical to v1)
# --------------------------------------------------------------------------------------------
class Forcing:
    def __init__(self):
        ds = xr.open_dataset(DATA / 'glorys_wide_surface_1999_2025.nc')
        self.dates = pd.DatetimeIndex(ds.time.values).normalize()
        self.date_to_idx = {d: i for i, d in enumerate(self.dates)}
        self.lats = ds.latitude.values.astype(np.float64)
        self.lons = ds.longitude.values.astype(np.float64)
        self.dlat = float(self.lats[1] - self.lats[0])
        self.dlon = float(self.lons[1] - self.lons[0])
        self.n_lat, self.n_lon = len(self.lats), len(self.lons)
        uo, vo, sst = ds['uo'].values, ds['vo'].values, ds['thetao'].values
        self.wet = ~np.isnan(uo[0])
        _, (jn, in_) = distance_transform_edt(~self.wet, return_indices=True)
        self.uo = np.where(np.isnan(uo), uo[:, jn, in_], uo)
        self.vo = np.where(np.isnan(vo), vo[:, jn, in_], vo)
        self.sst = np.where(np.isnan(sst), sst[:, jn, in_], sst)

        env = pd.read_parquet(ENV_PATH)
        env['date'] = pd.to_datetime(env['date']).dt.normalize()
        env = env.set_index('date').reindex(self.dates).ffill().bfill()
        self.w = {k: env[k].values for k in ['bay_wind_u', 'bay_wind_v', 'outer_wind_u',
                                             'outer_wind_v', 'sound_wind_u', 'sound_wind_v']}

    def ocean(self, t, lon, lat):
        fy = np.clip((lat - self.lats[0]) / self.dlat, 0, self.n_lat - 1.001)
        fx = np.clip((lon - self.lons[0]) / self.dlon, 0, self.n_lon - 1.001)
        j0, i0 = fy.astype(int), fx.astype(int)
        wy, wx = fy - j0, fx - i0

        def bil(a):
            return ((1 - wy) * ((1 - wx) * a[t, j0, i0] + wx * a[t, j0, i0 + 1])
                    + wy * ((1 - wx) * a[t, j0 + 1, i0] + wx * a[t, j0 + 1, i0 + 1]))
        return bil(self.uo), bil(self.vo)

    def wind(self, t, lon, lat):
        wb = 1.0 / ((lat - 42.00) ** 2 + (lon + 70.25) ** 2 + 0.02)
        wo = 1.0 / ((lat - 42.00) ** 2 + (lon + 69.75) ** 2 + 0.02)
        ws = 1.0 / ((lat - 41.50) ** 2 + (lon + 70.25) ** 2 + 0.02)
        s = wb + wo + ws
        w = self.w
        u = (wb * w['bay_wind_u'][t] + wo * w['outer_wind_u'][t] + ws * w['sound_wind_u'][t]) / s
        v = (wb * w['bay_wind_v'][t] + wo * w['outer_wind_v'][t] + ws * w['sound_wind_v'][t]) / s
        return u, v

    def local_sst(self, t0_idx, lon, lat):
        j = np.clip(np.round((lat - self.lats[0]) / self.dlat).astype(int), 0, self.n_lat - 1)
        i = np.clip(np.round((lon - self.lons[0]) / self.dlon).astype(int), 0, self.n_lon - 1)
        return self.sst[t0_idx, j, i]


# --------------------------------------------------------------------------------------------
# Per-animal stochastic parameters (same draw order as v1 -> identical values for same seed)
# --------------------------------------------------------------------------------------------
def draw_parameters(tiers, local_sst, M=50, seed=2026, alpha_range=(0.015, 0.035), use_sst=True):
    rng = np.random.default_rng(seed)
    N = len(tiers)
    sst_scale = np.clip(np.exp(-0.055 * (local_sst - 10.0)), 0.45, 1.65) if use_sst else np.ones(N)
    base = np.zeros((N, M))
    for i, tier in enumerate(tiers):
        if tier == 'Moderate Decomposition':
            base[i] = rng.uniform(3.0, 8.0, size=M)
        elif tier == 'Advanced Decomposition':
            base[i] = rng.uniform(8.0, 16.0, size=M)
        else:
            base[i] = rng.uniform(0.5, 2.0, size=M)
    drift_days = np.clip(base * sst_scale[:, None], 0.375, 21.0)
    alpha = rng.uniform(alpha_range[0], alpha_range[1], size=(N, M))
    div = np.radians(rng.normal(0.0, 8.0, size=(N, M))) if alpha_range[1] > 0 else np.zeros((N, M))
    return dict(drift_days=drift_days, alpha=alpha, div=div, rng=rng)


# --------------------------------------------------------------------------------------------
# Engine
# --------------------------------------------------------------------------------------------
def run_engine(forcing: Forcing, land: LandMask, lon_start, lat_start, t0_idx, params, rule='strand',
               dt_hours=0.5, grace_km=2.0, Kh=20.0, record=None, noise_seed=7):
    """Backward-in-time Monte Carlo drift.

    lon_start, lat_start, t0_idx : (N,) arrays (start points must be in raster water)
    params : dict from draw_parameters(), arrays shaped (N, M)
    record : iterable of animal row indices (0..N-1) whose full trajectories are stored
    Returns dict with end_lon/end_lat/status/held_steps (N, M) and optional trajectories.
    """
    assert rule in ('strand', 'hold')
    rng = np.random.default_rng(noise_seed)
    drift_days = params['drift_days']
    N, M = drift_days.shape
    dt_days = dt_hours / 24.0
    dt_sec = dt_hours * 3600.0
    n_steps = np.round(drift_days / dt_days).astype(int).ravel()
    alpha = params['alpha'].ravel()
    cdiv, sdiv = np.cos(params['div']).ravel(), np.sin(params['div']).ravel()
    diff_std = np.sqrt(2.0 * Kh * dt_sec)

    P = N * M
    lon = np.repeat(np.asarray(lon_start, float), M)
    lat = np.repeat(np.asarray(lat_start, float), M)
    lon_s, lat_s = lon.copy(), lat.copy()
    t0 = np.repeat(np.asarray(t0_idx), M)
    status = np.zeros(P, dtype=np.int8)
    held = np.zeros(P, dtype=np.int32)
    n_dates = len(forcing.dates)

    rec = None
    if record is not None:
        record = list(record)
        rec_idx = np.concatenate([np.arange(r * M, (r + 1) * M) for r in record])
        rec = dict(animals=record, lon=[lon[rec_idx].copy()], lat=[lat[rec_idx].copy()],
                   alive=[np.ones(len(rec_idx), bool)])

    for step in range(int(n_steps.max())):
        act = np.where((step < n_steps) & (status == STATUS_OK))[0]
        if len(act) == 0:
            break
        t = np.clip(t0[act] - int(step * dt_days), 0, n_dates - 1)
        lo, la = lon[act], lat[act]
        uo, vo = forcing.ocean(t, lo, la)
        uw, vw = forcing.wind(t, lo, la)
        ul = alpha[act] * (uw * cdiv[act] - vw * sdiv[act])
        vl = alpha[act] * (uw * sdiv[act] + vw * cdiv[act])
        dx = -(uo + ul) * dt_sec + rng.normal(0.0, diff_std, len(act))
        dy = -(vo + vl) * dt_sec + rng.normal(0.0, diff_std, len(act))
        nlo = lo + dx / (M_PER_DEG * np.cos(np.radians(la)))
        nla = la + dy / M_PER_DEG

        # ---- full-segment land check (samples every <= 0.5 raster cell) ----
        dlo, dla = nlo - lo, nla - la
        n_sub = np.maximum(1, np.ceil(np.maximum(np.abs(dlo), np.abs(dla)) / (0.5 * RES))).astype(int)
        hit = np.zeros(len(act), bool)
        last_lo, last_la = lo.copy(), la.copy()
        for k in range(1, int(n_sub.max()) + 1):
            sel = np.where((~hit) & (k <= n_sub))[0]
            if len(sel) == 0:
                break
            f = k / n_sub[sel]
            slo, sla = lo[sel] + f * dlo[sel], la[sel] + f * dla[sel]
            on_land = land.is_land(slo, sla)
            hit[sel[on_land]] = True
            w = sel[~on_land]
            last_lo[w], last_la[w] = slo[~on_land], sla[~on_land]

        out = (~hit) & ((nlo < LON_MIN) | (nlo > LON_MAX) | (nla < LAT_MIN) | (nla > LAT_MAX))

        new_lo, new_la = nlo.copy(), nla.copy()
        if rule == 'strand':
            d_start = np.hypot((last_la - lat_s[act]) * KM_PER_DEG,
                               (last_lo - lon_s[act]) * KM_PER_DEG * np.cos(np.radians(last_la)))
            strand = hit & (d_start > grace_km)
            hold = hit & ~strand
            new_lo[strand], new_la[strand] = last_lo[strand], last_la[strand]
            status[act[strand]] = STATUS_STRANDED
        else:
            hold = hit
        new_lo[hold], new_la[hold] = lo[hold], la[hold]
        held[act[hold]] += 1
        status[act[out]] = STATUS_EXITED
        lon[act], lat[act] = new_lo, new_la

        if rec is not None:
            alive = (step < n_steps[rec_idx]) & (status[rec_idx] == STATUS_OK)
            moved_this_step = step < n_steps[rec_idx]
            rec['lon'].append(lon[rec_idx].copy())
            rec['lat'].append(lat[rec_idx].copy())
            rec['alive'].append(alive | (moved_this_step & (status[rec_idx] != STATUS_OK)
                                         & np.isin(rec_idx, act)))

    result = dict(end_lon=lon.reshape(N, M), end_lat=lat.reshape(N, M),
                  status=status.reshape(N, M), held_steps=held.reshape(N, M),
                  drift_days=drift_days, rule=rule, grace_km=grace_km, dt_hours=dt_hours)
    if rec is not None:
        L = np.array(rec['lon'])        # (T+1, len(record)*M)
        A = np.array(rec['lat'])
        traj = {}
        for k, r in enumerate(rec['animals']):
            sl = slice(k * M, (k + 1) * M)
            traj[r] = dict(lon=L[:, sl], lat=A[:, sl], n_steps=n_steps[r * M:(r + 1) * M],
                           status=status[r * M:(r + 1) * M])
        result['traj'] = traj
    return result


# --------------------------------------------------------------------------------------------
# Summaries
# --------------------------------------------------------------------------------------------
def classify_basin(lat, lon):
    """Identical basin definitions to v1."""
    lat, lon = np.asarray(lat), np.asarray(lon)
    conds = [
        (lat >= 41.72) & (lat <= 42.06) & (lon >= -70.75) & (lon <= -70.00) & ~((lat < 41.75) & (lon < -70.58)),
        (lat > 42.06) & (lat <= 42.52) & (lon <= -70.00),
        (lat > 42.52),
        (lat >= 41.55) & (lat <= 42.52) & (lon > -70.00),
        (lat < 41.55) & (lon > -70.20),
        (lat < 41.75) & (lon <= -70.20),
    ]
    return np.select(conds, BASIN_LABELS, default='4. Outer Cape & Nauset Shelf')


def medoid(lon, lat):
    """Particle endpoint minimising total distance to the others (always a real, in-water endpoint)."""
    if len(lon) == 0:
        return np.nan, np.nan
    x = lon * np.cos(np.radians(np.mean(lat)))
    d = np.hypot(x[:, None] - x[None, :], lat[:, None] - lat[None, :]).sum(1)
    k = int(np.argmin(d))
    return lon[k], lat[k]


def path_length_km(lon, lat):
    return np.nansum(np.hypot(np.diff(lat) * KM_PER_DEG,
                              np.diff(lon) * KM_PER_DEG * np.cos(np.radians(lat[:-1]))))
