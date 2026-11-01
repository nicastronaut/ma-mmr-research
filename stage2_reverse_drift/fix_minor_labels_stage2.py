import shutil
from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr
import geopandas as gpd
import shapely
from scipy.ndimage import distance_transform_edt
from scipy.stats import gaussian_kde
from matplotlib.path import Path as MplPath
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['axes.edgecolor'] = '#333333'
plt.rcParams['axes.linewidth'] = 0.85

PROJECT_ROOT = Path('/home/nicastronaut/marine_bio/stage2_reverse_drift')
DATA_DIR = PROJECT_ROOT / 'data'
FIG_DIR = PROJECT_ROOT / 'figures'
ART_DIR = Path('/home/nicastronaut/.gemini/antigravity/brain/f81a9535-0cfe-4288-a167-0b9c32e31954')

shp_path = DATA_DIR / 'cb_2023_us_county_500k.zip'
gdf_counties = gpd.read_file(shp_path)
ne_counties = gdf_counties[gdf_counties['STATEFP'].isin(['25', '44', '09', '33', '23', '36'])].to_crs('EPSG:4326')
ne_land_geom = ne_counties.union_all()
ne_states = ne_counties.dissolve(by='STATEFP')

def draw_real_coast(ax, xlim=(-71.40, -68.90), ylim=(41.00, 42.90), ocean_color='#eef6fa', land_color='#e6e1d6', border_color='#786e5c'):
    ax.set_facecolor(ocean_color)
    ne_counties.plot(ax=ax, facecolor=land_color, edgecolor='#cec7b8', linewidth=0.35, zorder=2.5)
    ne_states.boundary.plot(ax=ax, edgecolor=border_color, linewidth=0.85, zorder=2.6)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect(1.0 / np.cos(np.radians(np.mean(ylim))))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

def mask_kde_on_real_land(xx, yy, zz):
    on_land = shapely.contains_xy(ne_land_geom, xx, yy)
    return np.where(on_land, np.nan, zz)

df_all = pd.read_parquet(DATA_DIR / 'stage2_all_records_clean.parquet')
coastal = pd.read_parquet(DATA_DIR / 'backtracked_drifters_summary.parquet')
ds = xr.open_dataset(DATA_DIR / 'glorys_wide_surface_1999_2025.nc')
glorys_dates = pd.DatetimeIndex(ds.time.values).normalize()
date_to_idx = {d: i for i, d in enumerate(glorys_dates)}
lats = ds.latitude.values.astype(np.float64)
lons = ds.longitude.values.astype(np.float64)
dlat_grid = float(lats[1] - lats[0])
dlon_grid = float(lons[1] - lons[0])
lat0_grid = float(lats[0])
lon0_grid = float(lons[0])
n_lat, n_lon = len(lats), len(lons)

uo_raw = ds['uo'].values
vo_raw = ds['vo'].values
sst_raw = ds['thetao'].values
wet_mask_grid = ~np.isnan(uo_raw[0])
_, nearest_indices = distance_transform_edt(~wet_mask_grid, return_indices=True)
jj_near, ii_near = nearest_indices[0], nearest_indices[1]
uo_filled = np.where(np.isnan(uo_raw), uo_raw[:, jj_near, ii_near], uo_raw)
vo_filled = np.where(np.isnan(vo_raw), vo_raw[:, jj_near, ii_near], vo_raw)
sst_filled = np.where(np.isnan(sst_raw), sst_raw[:, jj_near, ii_near], sst_raw)

cape_cod_poly = MplPath([
    (-70.62, 41.755), (-70.48, 41.750), (-70.35, 41.720), (-70.18, 41.740),
    (-70.05, 41.780), (-70.01, 41.880), (-70.03, 41.980), (-70.08, 42.040),
    (-70.19, 42.050), (-70.14, 42.010), (-70.07, 41.950), (-69.96, 41.880),
    (-69.93, 41.780), (-69.95, 41.670), (-70.02, 41.660), (-70.22, 41.630),
    (-70.45, 41.610), (-70.66, 41.520), (-70.66, 41.650)
])
mv_poly = MplPath([
    (-70.84, 41.34), (-70.75, 41.45), (-70.58, 41.46), (-70.45, 41.36), (-70.55, 41.34)
])
nan_poly = MplPath([
    (-70.24, 41.25), (-70.08, 41.31), (-69.96, 41.33), (-69.96, 41.24), (-70.15, 41.24)
])

def is_on_land_physics(lon_arr, lat_arr):
    j_idx = np.clip(np.round((lat_arr - lat0_grid) / dlat_grid).astype(int), 0, n_lat - 1)
    i_idx = np.clip(np.round((lon_arr - lon0_grid) / dlon_grid).astype(int), 0, n_lon - 1)
    glorys_land = ~wet_mask_grid[j_idx, i_idx]
    in_ccb_water = (lat_arr > 41.755) & (lat_arr < 42.04) & (lon_arr > -70.52) & (lon_arr < -70.03)
    pts = np.column_stack([lon_arr, lat_arr])
    in_cape = cape_cod_poly.contains_points(pts)
    in_mv = mv_poly.contains_points(pts)
    in_nan = nan_poly.contains_points(pts)
    return (glorys_land & ~in_ccb_water) | in_cape | in_mv | in_nan

env_raw = pd.read_parquet('/home/nicastronaut/champ/capstone/data/processed/environmental_daily.parquet')
env_raw['date'] = pd.to_datetime(env_raw['date']).dt.normalize()
env = env_raw.set_index('date').reindex(glorys_dates).ffill().bfill()
w_bay_u, w_bay_v = env['bay_wind_u'].values, env['bay_wind_v'].values
w_out_u, w_out_v = env['outer_wind_u'].values, env['outer_wind_v'].values
w_snd_u, w_snd_v = env['sound_wind_u'].values, env['sound_wind_v'].values

def interp_wind(t_idx, lon_arr, lat_arr):
    d2_bay = (lat_arr - 42.00)**2 + (lon_arr - (-70.25))**2 + 0.02
    d2_out = (lat_arr - 42.00)**2 + (lon_arr - (-69.75))**2 + 0.02
    d2_snd = (lat_arr - 41.50)**2 + (lon_arr - (-70.25))**2 + 0.02
    wb, wo, ws = 1.0 / d2_bay, 1.0 / d2_out, 1.0 / d2_snd
    wsum = wb + wo + ws
    u10 = (wb * w_bay_u[t_idx] + wo * w_out_u[t_idx] + ws * w_snd_u[t_idx]) / wsum
    v10 = (wb * w_bay_v[t_idx] + wo * w_out_v[t_idx] + ws * w_snd_v[t_idx]) / wsum
    return u10, v10

def interp_ocean(t_idx, lon_arr, lat_arr):
    fy = np.clip((lat_arr - lat0_grid) / dlat_grid, 0, n_lat - 1.001)
    fx = np.clip((lon_arr - lon0_grid) / dlon_grid, 0, n_lon - 1.001)
    j0 = fy.astype(int)
    i0 = fx.astype(int)
    wy = fy - j0
    wx = fx - i0
    u = (1 - wy) * ((1 - wx) * uo_filled[t_idx, j0, i0] + wx * uo_filled[t_idx, j0, i0 + 1]) + wy * ((1 - wx) * uo_filled[t_idx, j0 + 1, i0] + wx * uo_filled[t_idx, j0 + 1, i0 + 1])
    v = (1 - wy) * ((1 - wx) * vo_filled[t_idx, j0, i0] + wx * vo_filled[t_idx, j0, i0 + 1]) + wy * ((1 - wx) * vo_filled[t_idx, j0 + 1, i0] + wx * vo_filled[t_idx, j0 + 1, i0 + 1])
    return u, v

angles = np.linspace(0, 2 * np.pi, 36, endpoint=False)
radii = np.linspace(0.01, 0.22, 30)
lat_init = coastal['lat_clean'].values.copy()
lon_init = coastal['lon_clean'].values.copy()
zones = coastal['coastal_zone'].values
on_l = is_on_land_physics(lon_init, lat_init)
for r in radii:
    if not on_l.any():
        break
    idx_bad = np.where(on_l)[0]
    for a in angles:
        cand_lon = lon_init[idx_bad] + r * np.cos(a) / np.cos(np.radians(lat_init[idx_bad]))
        cand_lat = lat_init[idx_bad] + r * np.sin(a)
        still_land = is_on_land_physics(cand_lon, cand_lat)
        is_bay_zone = np.isin(zones[idx_bad], ['1. Inner Cape Hook (Trap Flats)', '2. Bay-Side Upper/Mid Cape (Collector Shore)'])
        wrong_side = is_bay_zone & (cand_lat < 41.755)
        valid_nudge = (~still_land) & (~wrong_side)
        if valid_nudge.any():
            fix_rows = idx_bad[valid_nudge]
            lon_init[fix_rows] = cand_lon[valid_nudge]
            lat_init[fix_rows] = cand_lat[valid_nudge]
            idx_bad = idx_bad[~valid_nudge]
            if len(idx_bad) == 0:
                break
    on_l = is_on_land_physics(lon_init, lat_init)

obs_dates = pd.to_datetime(coastal['Observation Date']).dt.normalize()
t0_indices = np.array([date_to_idx[d] for d in obs_dates])
j_loc = np.clip(np.round((lat_init - lat0_grid) / dlat_grid).astype(int), 0, n_lat - 1)
i_loc = np.clip(np.round((lon_init - lon0_grid) / dlon_grid).astype(int), 0, n_lon - 1)
local_sst = sst_filled[t0_indices, j_loc, i_loc]

def run_reverse_backtracker(M=50, alpha_range=(0.015, 0.035), use_sst_scaling=True, Kh=20.0, seed=2026, record_trajectories_for=None):
    rng = np.random.default_rng(seed)
    N = len(coastal)
    sst_scale = np.clip(np.exp(-0.055 * (local_sst - 10.0)), 0.45, 1.65) if use_sst_scaling else np.ones(N)
    tiers = coastal['drift_tier'].values
    base_days = np.zeros((N, M))
    for i, tier in enumerate(tiers):
        if tier == 'Moderate Decomposition':
            base_days[i] = rng.uniform(3.0, 8.0, size=M)
        elif tier == 'Advanced Decomposition':
            base_days[i] = rng.uniform(8.0, 16.0, size=M)
        else:
            base_days[i] = rng.uniform(0.5, 2.0, size=M)
    drift_days = np.clip(base_days * sst_scale[:, None], 0.375, 21.0)
    dt_days = 0.125
    dt_sec = dt_days * 86400.0
    n_steps_particle = np.round(drift_days / dt_days).astype(int)
    max_steps = int(n_steps_particle.max())
    alpha_leeway = rng.uniform(alpha_range[0], alpha_range[1], size=(N, M))
    div_rad = np.radians(rng.normal(0.0, 8.0, size=(N, M))) if alpha_range[1] > 0 else np.zeros((N, M))
    cos_div, sin_div = np.cos(div_rad), np.sin(div_rad)
    diff_std_m = np.sqrt(2.0 * Kh * dt_sec)
    p_lon = np.repeat(lon_init, M)
    p_lat = np.repeat(lat_init, M)
    p_t0 = np.repeat(t0_indices, M)
    p_steps = n_steps_particle.ravel()
    p_alpha = alpha_leeway.ravel()
    p_cos = cos_div.ravel()
    p_sin = sin_div.ravel()
    traj_history = {}
    if record_trajectories_for is not None:
        for r_idx in record_trajectories_for:
            traj_history[r_idx] = [(lon_init[r_idx] * np.ones(M), lat_init[r_idx] * np.ones(M))]
    for step in range(max_steps):
        active = step < p_steps
        if not active.any():
            break
        idx_act = np.where(active)[0]
        day_offset = int(step * dt_days)
        t_idx = np.clip(p_t0[idx_act] - day_offset, 0, len(glorys_dates) - 1)
        lon_cur = p_lon[idx_act]
        lat_cur = p_lat[idx_act]
        u_oc, v_oc = interp_ocean(t_idx, lon_cur, lat_cur)
        u_w, v_w = interp_wind(t_idx, lon_cur, lat_cur)
        u_leeway = p_alpha[idx_act] * (u_w * p_cos[idx_act] - v_w * p_sin[idx_act])
        v_leeway = p_alpha[idx_act] * (u_w * p_sin[idx_act] + v_w * p_cos[idx_act])
        dx_m = -(u_oc + u_leeway) * dt_sec + rng.normal(0.0, diff_std_m, size=len(idx_act))
        dy_m = -(v_oc + v_leeway) * dt_sec + rng.normal(0.0, diff_std_m, size=len(idx_act))
        dlon = dx_m / (111132.0 * np.cos(np.radians(lat_cur)))
        dlat = dy_m / 111132.0
        new_lon = np.clip(lon_cur + dlon, -71.55, -68.52)
        new_lat = np.clip(lat_cur + dlat, 40.26, 43.15)
        hit_land = is_on_land_physics(new_lon, new_lat)
        if hit_land.any():
            hl_idx = np.where(hit_land)[0]
            lon_only = ~is_on_land_physics(new_lon[hl_idx], lat_cur[hl_idx])
            lat_only = ~is_on_land_physics(lon_cur[hl_idx], new_lat[hl_idx])
            new_lon_hl = np.where(lon_only, new_lon[hl_idx], lon_cur[hl_idx])
            new_lat_hl = np.where((~lon_only) & lat_only, new_lat[hl_idx], lat_cur[hl_idx])
            new_lon[hl_idx] = new_lon_hl
            new_lat[hl_idx] = new_lat_hl
        p_lon[idx_act] = new_lon
        p_lat[idx_act] = new_lat
        if record_trajectories_for is not None:
            for r_idx in record_trajectories_for:
                sl = slice(r_idx * M, (r_idx + 1) * M)
                if (step < n_steps_particle[r_idx]).any():
                    traj_history[r_idx].append((p_lon[sl].copy(), p_lat[sl].copy()))
    return p_lon.reshape(N, M), p_lat.reshape(N, M), drift_days, traj_history

idx_hp, idx_fi, idx_vs, idx_so = 14, 614, 294, 477
showcase_ids = [idx_hp, idx_fi, idx_vs, idx_so]
origin_lon, origin_lat, drift_days, traj_hist = run_reverse_backtracker(
    M=50, alpha_range=(0.015, 0.035), use_sst_scaling=True, Kh=20.0, seed=2026, record_trajectories_for=showcase_ids
)

def save_and_copy(fig, fname):
    out_f = FIG_DIR / fname
    art_f = ART_DIR / fname
    fig.savefig(out_f, bbox_inches='tight', facecolor='white')
    shutil.copy(out_f, art_f)
    plt.close(fig)
    print(f"Updated {fname}")

# ==============================================================================
# FIX FIGURE 1: Move Wellfleet callout to East side (-69.85, 42.02) so zero overlap
# ==============================================================================
fig = plt.figure(figsize=(16.5, 7.6), dpi=180)
gs = gridspec.GridSpec(1, 2, width_ratios=[1.12, 1.0], wspace=0.25)

ax1 = fig.add_subplot(gs[0, 0])
draw_real_coast(ax1, xlim=(-71.40, -68.90), ylim=(41.00, 42.90))
c_df = df_all.loc[df_all['County'] != 'EEZ'].copy()
e_df = df_all.loc[df_all['County'] == 'EEZ'].copy()
decomp_m = c_df['drift_tier'].isin(['Moderate Decomposition', 'Advanced Decomposition', 'Mummified/Skeletal'])
alive_m = c_df['drift_tier'] == 'Alive'
hi_m = c_df['hi_type'].isin(['Fishery / Entanglement', 'Vessel Strike'])

ax1.scatter(c_df.loc[decomp_m, 'lon_clean'], c_df.loc[decomp_m, 'lat_clean'], s=14, c='#d95f02', alpha=0.60, label=f'Decomposed Carcasses (n={decomp_m.sum():,})', zorder=3.5)
ax1.scatter(c_df.loc[alive_m, 'lon_clean'], c_df.loc[alive_m, 'lat_clean'], s=14, c='#1b9e77', alpha=0.50, label=f'Live Strandings (n={alive_m.sum():,})', zorder=3.2)
ax1.scatter(c_df.loc[hi_m, 'lon_clean'], c_df.loc[hi_m, 'lat_clean'], s=38, c='#e7298a', marker='^', edgecolors='k', linewidths=0.45, label=f'Fishery / Vessel Strike HI (n={hi_m.sum():,})', zorder=4.0)
ax1.scatter(e_df['lon_clean'], e_df['lat_clean'], s=54, c='#7570b3', marker='*', edgecolors='k', linewidths=0.5, label=f'Offshore EEZ Floating Sightings (n={len(e_df):,})', zorder=4.5)

ax1.annotate('Wellfleet / Brewster / Dennis\nActive Swimmer Trap\n(47–60% Stranded Alive)', xy=(-70.03, 41.91), xytext=(-69.85, 42.02),
             arrowprops=dict(arrowstyle='->', color='#0f6e52', lw=1.6),
             bbox=dict(boxstyle='round,pad=0.28', fc='white', ec='#1b9e77', lw=1.3, alpha=0.95), fontsize=8.3, fontweight='bold', color='#0f6e52', zorder=6)
ax1.annotate('Sandwich & Barnstable (Bay-Side)\nPassive Collector Shore\n(60–78% Decomposed)', xy=(-70.42, 41.75), xytext=(-71.36, 41.73),
             arrowprops=dict(arrowstyle='->', color='#a64200', lw=1.6),
             bbox=dict(boxstyle='round,pad=0.28', fc='white', ec='#d95f02', lw=1.3, alpha=0.95), fontsize=8.3, fontweight='bold', color='#a64200', zorder=6)
ax1.annotate('South Shore (Hull–Plymouth)\nCarcass Collector Coast\n(60–83% Decomposed)', xy=(-70.68, 42.08), xytext=(-71.36, 42.18),
             arrowprops=dict(arrowstyle='->', color='#a64200', lw=1.6),
             bbox=dict(boxstyle='round,pad=0.28', fc='white', ec='#d95f02', lw=1.3, alpha=0.95), fontsize=8.3, fontweight='bold', color='#a64200', zorder=6)
ax1.annotate('Islands & Buzzards Bay\nSouthern Shelf Collectors\n(58–79% Decomposed)', xy=(-70.55, 41.38), xytext=(-71.36, 41.10),
             arrowprops=dict(arrowstyle='->', color='#a64200', lw=1.6),
             bbox=dict(boxstyle='round,pad=0.28', fc='white', ec='#d95f02', lw=1.3, alpha=0.95), fontsize=8.3, fontweight='bold', color='#a64200', zorder=6)

ax1.set_xlabel('Longitude (°W)', fontsize=10)
ax1.set_ylabel('Latitude (°N)', fontsize=10)
ax1.set_title('A. Active "Trap Beaches" vs. Passive "Collector Beaches" (1:500k Coastline)', fontsize=11, fontweight='bold', loc='left')
ax1.legend(loc='upper right', fontsize=8.3, framealpha=0.95)
ax1.grid(True, linestyle='--', alpha=0.3)

ax2 = fig.add_subplot(gs[0, 1])
def get_town_side(row):
    t = str(row['town']).strip()
    lat = row['lat_clean']
    if t == 'Barnstable':
        return 'Barnstable (Bay-Side)' if lat >= 41.685 else 'Barnstable (Sound-Side)'
    elif t == 'Yarmouth':
        return 'Yarmouth (Bay-Side)' if lat >= 41.685 else 'Yarmouth (Sound-Side)'
    return t

c_df['town_side'] = c_df.apply(get_town_side, axis=1)
t_counts = c_df.loc[c_df['town_side'] != 'Unknown'].groupby('town_side').size()
maj_towns = t_counts[t_counts >= 25].index
sub_t = c_df.loc[c_df['town_side'].isin(maj_towns)].copy()
t_tab = pd.crosstab(sub_t['town_side'], sub_t['drift_tier'], normalize='index') * 100
t_tab['n'] = sub_t.groupby('town_side').size()
t_tab['decomp_pct'] = t_tab.get('Moderate Decomposition', 0) + t_tab.get('Advanced Decomposition', 0) + t_tab.get('Mummified/Skeletal', 0)
t_tab['fresh_single_pct'] = t_tab.get('Fresh Dead (Single Drifter)', 0)
t_tab['fresh_mse_pct'] = t_tab.get('Fresh Dead (MSE)', 0)
t_tab['alive_pct'] = t_tab.get('Alive', 0)
t_tab = t_tab.sort_values('decomp_pct', ascending=True)

y_p = np.arange(len(t_tab))
ax2.barh(y_p, t_tab['decomp_pct'], color='#d95f02', label='Moderate / Advanced Decomp (Drifter)')
ax2.barh(y_p, t_tab['fresh_single_pct'], left=t_tab['decomp_pct'], color='#fc8d62', label='Fresh Dead Singleton (<48h Drifter)')
ax2.barh(y_p, t_tab['fresh_mse_pct'], left=t_tab['decomp_pct'] + t_tab['fresh_single_pct'], color='#8da0cb', label='Fresh Dead (In-MSE)')
ax2.barh(y_p, t_tab['alive_pct'], left=t_tab['decomp_pct'] + t_tab['fresh_single_pct'] + t_tab['fresh_mse_pct'], color='#1b9e77', label='Stranded Alive (Swimmer)')
ax2.axvline(50, color='#333333', linestyle='--', lw=1.1)
ax2.set_yticks(y_p)
ax2.set_yticklabels([f"{idx} (n={int(t_tab.loc[idx, 'n'])})" for idx in t_tab.index], fontsize=8.6)
ax2.set_xlim(0, 100)
ax2.set_xlabel('Percentage of Town Strandings (%)', fontsize=10)
ax2.set_title('B. Town-Level "Collector Index" (Bay-Side vs. Sound-Side Split)', fontsize=11, fontweight='bold', loc='left')
ax2.legend(loc='lower right', fontsize=8.3, framealpha=0.95)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.grid(True, axis='x', linestyle='--', alpha=0.3)

fig.suptitle('Figure 1: Functional Classification of Massachusetts Coastlines — Passive Carcass Collectors vs. Active Swimmer Traps',
             fontsize=13, fontweight='bold', y=0.98)
save_and_copy(fig, 's2_fig01_trap_vs_collector_map_and_bars.png')

# ==============================================================================
# FIX FIGURE 2: Wrap Panel A x-tick labels & move Panel B callout below legend
# ==============================================================================
fig = plt.figure(figsize=(15.8, 6.4), dpi=180)
gs = gridspec.GridSpec(1, 2, width_ratios=[1.0, 1.05], wspace=0.25)

ax1 = fig.add_subplot(gs[0, 0])
guilds = ['Harbor Porpoise', 'Baleen & Large Whales', 'Common Dolphin', 'Other Pelagic Delphinid', 'White-Sided Dolphin']
g_sub = c_df.loc[c_df['species_guild'].isin(guilds)].copy()
g_tab = pd.crosstab(g_sub['species_guild'], g_sub['drift_tier'], normalize='index').reindex(guilds) * 100
g_n = g_sub.groupby('species_guild').size().reindex(guilds)
g_decomp = g_tab.get('Moderate Decomposition', 0) + g_tab.get('Advanced Decomposition', 0) + g_tab.get('Mummified/Skeletal', 0)
g_fsingle = g_tab.get('Fresh Dead (Single Drifter)', 0)
g_alive = g_tab.get('Alive', 0) + g_tab.get('Fresh Dead (MSE)', 0)

x_g = np.arange(len(guilds))
w_g = 0.56
ax1.bar(x_g, g_decomp, width=w_g, color='#d95f02', label='Decomposed Carcass (Mod/Adv/Mum)')
ax1.bar(x_g, g_fsingle, width=w_g, bottom=g_decomp, color='#fc8d62', label='Fresh Dead Singleton (<48h)')
ax1.bar(x_g, g_alive, width=w_g, bottom=g_decomp + g_fsingle, color='#1b9e77', label='Alive / Fresh MSE Swimmer')
for i, g in enumerate(guilds):
    ax1.text(i, g_decomp.iloc[i]/2, f"{g_decomp.iloc[i]:.0f}%", ha='center', va='center', color='white', fontweight='bold', fontsize=10)
    ax1.text(i, g_decomp.iloc[i] + g_fsingle.iloc[i] + g_alive.iloc[i]/2, f"{g_alive.iloc[i]:.0f}%", ha='center', va='center', color='white', fontweight='bold', fontsize=10)

nice_g_labels = [
    f"Harbor\nPorpoise\n(n={g_n['Harbor Porpoise']:,})",
    f"Baleen &\nLarge Whales\n(n={g_n['Baleen & Large Whales']:,})",
    f"Common\nDolphin\n(n={g_n['Common Dolphin']:,})",
    f"Other Pelagic\nDolphins\n(n={g_n['Other Pelagic Delphinid']:,})",
    f"White-Sided\nDolphin\n(n={g_n['White-Sided Dolphin']:,})"
]
ax1.set_xticks(x_g)
ax1.set_xticklabels(nice_g_labels, fontsize=8.8)
ax1.set_ylim(0, 108)
ax1.set_ylabel('Share of Species Strandings (%)', fontsize=10)
ax1.set_title('A. Condition at Discovery by Species Guild (Drifters vs. Swimmers)', fontsize=11, fontweight='bold', loc='left')
ax1.legend(loc='upper right', fontsize=8.5, framealpha=0.95)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.grid(True, axis='y', linestyle='--', alpha=0.3)

ax2 = fig.add_subplot(gs[0, 1])
months = np.arange(1, 13)
m_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
for coh_key, lbl, col, mk in [
    ('3. Harbor Porpoise Carcass', 'Harbor Porpoise Carcass (n=618)', '#d95f02', 'o'),
    ('4. Pelagic Dolphin Carcass', 'Pelagic Dolphin Carcass (n=1,344)', '#1b9e77', 's'),
    ('1. Confirmed Fishery / Entanglement', 'Confirmed Fishery / Entanglement (n=112)', '#e7298a', '^'),
    ('5. Large Whale / Other Carcass', 'Large Whale / Other Carcass (n=185)', '#7570b3', 'D'),
]:
    s_df = coastal.loc[coastal['drifter_cohort'] == coh_key]
    m_pct = s_df['month'].value_counts().reindex(months, fill_value=0) / len(s_df) * 100
    ax2.plot(months, m_pct, marker=mk, lw=2.4, color=col, label=lbl)
ax2.axvspan(2.6, 5.4, color='#d95f02', alpha=0.13, label='Spring Gillnet / Porpoise Peak (Mar–May: 59.2%)')
ax2.annotate('March Peak: 33.7% of all\ndead Harbor Porpoises\nwash ashore in March alone',
             xy=(3, 33.66), xytext=(4.3, 20.8),
             arrowprops=dict(arrowstyle='->', color='#a64200', lw=1.5),
             bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='#d95f02', lw=1.2, alpha=0.96),
             fontsize=8.4, fontweight='bold', color='#a64200')
ax2.set_xticks(months)
ax2.set_xticklabels(m_names, fontsize=9)
ax2.set_ylim(-1, 36.5)
ax2.set_ylabel('Share of Annual Cohort Strandings (%)', fontsize=10)
ax2.set_xlabel('Month of Beach Discovery', fontsize=10)
ax2.set_title('B. Seasonal Timing of Drifter Cohorts (March–May Spring Spike)', fontsize=11, fontweight='bold', loc='left')
ax2.legend(loc='upper right', fontsize=8.3, framealpha=0.95)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.grid(True, linestyle='--', alpha=0.3)

fig.suptitle('Figure 2: Biological & Seasonal Divergence Between Passive Drifter Cohorts and Live Swimmer Guilds',
             fontsize=13, fontweight='bold', y=1.00)
save_and_copy(fig, 's2_fig02_species_guilds_and_seasonality.png')

# ==============================================================================
# FIX FIGURE 4: Separate Case 3 callout box from top-left legend
# ==============================================================================
fig, ax = plt.subplots(figsize=(12.5, 9.4), dpi=180)
draw_real_coast(ax, xlim=(-71.15, -69.25), ylim=(41.15, 42.80))

showcase_meta = [
    (idx_hp, '#d95f02', 'NE-2000-1039099: Harbor Porpoise (Mar 19, 2000 | Barnstable Bay-Side -> W. Cape Cod Bay, 7.6d, 39.6 km)'),
    (idx_fi, '#e7298a', 'NE-2008-1084979: Entangled Pilot Whale (Dec 21, 2008 | Chatham -> Outer Cape Shelf, 6.7d, 41.0 km)'),
    (idx_vs, '#7570b3', 'NE-2004-1040205: Vessel-Struck Minke Whale (May 31, 2004 | Chatham -> Stellwagen Shipping Approach, 5.4d, 80.5 km)'),
    (idx_so, '#1b9e77', 'NE-2006-1061521: Sei Whale Carcass (May 16, 2006 | Edgartown MV -> Great South Channel, 5.4d, 62.9 km)'),
]

for r_idx, col, lbl in showcase_meta:
    hist = traj_hist[r_idx]
    arr_lon = np.array([h[0] for h in hist])
    arr_lat = np.array([h[1] for h in hist])
    for m in range(arr_lon.shape[1]):
        ax.plot(arr_lon[:, m], arr_lat[:, m], color=col, alpha=0.18, linewidth=0.95, zorder=3.0)
    mean_lon = arr_lon.mean(axis=1)
    mean_lat = arr_lat.mean(axis=1)
    ax.plot(mean_lon, mean_lat, color=col, linewidth=3.0, label=lbl, zorder=4.2)
    ax.scatter(arr_lon[0, 0], arr_lat[0, 0], s=115, c='white', edgecolors=col, marker='v', linewidths=2.4, zorder=5.0)
    ax.scatter(arr_lon[-1, :], arr_lat[-1, :], s=22, c=col, edgecolors='white', linewidths=0.35, alpha=0.75, zorder=3.8)
    ax.scatter(mean_lon[-1], mean_lat[-1], s=190, c=col, edgecolors='black', marker='*', linewidths=0.9, zorder=5.5)

ax.annotate('Case 1: NE-2000-1039099 (Harbor Porpoise)\n★ Inferred Origin: W. Cape Cod Bay (Mar 12)\n▽ Beached: Barnstable Bay-Side (Mar 19)\n(V-turn driven by Mar 17 northerly gale)',
            xy=(origin_lon[idx_hp].mean(), origin_lat[idx_hp].mean()), xytext=(-71.10, 42.02),
            arrowprops=dict(arrowstyle='->', color='#a64200', lw=1.6),
            bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='#d95f02', lw=1.4, alpha=0.96),
            fontsize=8.5, fontweight='bold', color='#8c2d04', zorder=6)

ax.annotate('Case 3: NE-2004-1040205 (Minke Whale, Vessel Strike)\n★ Inferred Origin: Stellwagen Bank Approach (May 26)\n▽ Beached: Chatham Outer Cape (May 31, 80.5 km south)',
            xy=(origin_lon[idx_vs].mean(), origin_lat[idx_vs].mean()), xytext=(-69.98, 42.34),
            arrowprops=dict(arrowstyle='->', color='#524b8a', lw=1.6),
            bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='#7570b3', lw=1.4, alpha=0.96),
            fontsize=8.5, fontweight='bold', color='#3f396e', zorder=6)

ax.annotate('Case 2: NE-2008-1084979 (Pilot Whale, Entangled)\n★ Inferred Origin: Outer Cape / Nauset Shelf (Dec 14)\n▽ Beached: Chatham (Dec 21, 41.0 km west)',
            xy=(origin_lon[idx_fi].mean(), origin_lat[idx_fi].mean()), xytext=(-69.95, 41.94),
            arrowprops=dict(arrowstyle='->', color='#b01463', lw=1.6),
            bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='#e7298a', lw=1.4, alpha=0.96),
            fontsize=8.5, fontweight='bold', color='#8f0d4e', zorder=6)

ax.annotate('Case 4: NE-2006-1061521 (Sei Whale Carcass)\n★ Inferred Origin: Great South Channel / Shoals (May 11)\n▽ Beached: Edgartown, Martha\'s Vineyard (May 16, 62.9 km west)',
            xy=(origin_lon[idx_so].mean(), origin_lat[idx_so].mean()), xytext=(-70.35, 41.19),
            arrowprops=dict(arrowstyle='->', color='#0f6e52', lw=1.6),
            bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='#1b9e77', lw=1.4, alpha=0.96),
            fontsize=8.5, fontweight='bold', color='#0b523d', zorder=6)

ax.set_xlabel('Longitude (°W)', fontsize=10.5)
ax.set_ylabel('Latitude (°N)', fontsize=10.5)
ax.set_title('Figure 4: Four Verifiable NOAA Case Studies — 50-Particle Reverse-Drift Plumes (▽ = Beach Discovery, ★ = Inferred Origin)',
             fontsize=12.0, fontweight='bold', loc='left', pad=10)
ax.legend(loc='upper left', fontsize=8.6, framealpha=0.96)
ax.grid(True, linestyle='--', alpha=0.3)
save_and_copy(fig, 's2_fig04_four_case_studies_plume_map.png')

# ==============================================================================
# FIX FIGURE 5: Exact V-turn apex waypoint & zero title/legend overlap
# ==============================================================================
fig = plt.figure(figsize=(16.2, 6.8), dpi=180)
gs = gridspec.GridSpec(1, 2, width_ratios=[1.05, 1.0], wspace=0.28)

ax1 = fig.add_subplot(gs[0, 0])
draw_real_coast(ax1, xlim=(-70.78, -69.90), ylim=(41.64, 42.30))
hist_hp = traj_hist[idx_hp]
arr_lon_hp = np.array([h[0] for h in hist_hp])
arr_lat_hp = np.array([h[1] for h in hist_hp])
for m in range(arr_lon_hp.shape[1]):
    ax1.plot(arr_lon_hp[:, m], arr_lat_hp[:, m], color='#d95f02', alpha=0.22, lw=1.0, zorder=3.0)

ax1.scatter(arr_lon_hp[-1, :], arr_lat_hp[-1, :], s=28, c='#d95f02', edgecolors='white', lw=0.45, alpha=0.88, label='50 Particle Origins (Mar 11–13, 2000)', zorder=4.0)
mean_lon_hp = arr_lon_hp.mean(axis=1)
mean_lat_hp = arr_lat_hp.mean(axis=1)
ax1.plot(mean_lon_hp, mean_lat_hp, color='#8c2d04', lw=3.2, label='Ensemble Centroid Trajectory', zorder=4.5)

ax1.scatter(mean_lon_hp[0], mean_lat_hp[0], marker='v', s=150, facecolor='white', edgecolor='#8c2d04', lw=2.5, label='Beached: Mar 19, 2000 (Barnstable Bay-Side)', zorder=5.5)
ax1.scatter(mean_lon_hp[-1], mean_lat_hp[-1], marker='*', s=220, facecolor='#ffeda0', edgecolor='#8c2d04', lw=1.4, label='Mean Inferred Origin: Mar 12, 2000 (W. Cape Cod Bay)', zorder=5.5)

idx_apex = int(np.argmax(mean_lat_hp))
waypoint_steps = [
    (0,        'Mar 19 (Day 0)\nBeached at Barnstable', (-70.27, 41.665)),
    (idx_apex, 'Mar 17 (-2.5d)\nNortherly Gale V-Turn\noff Provincetown Hook', (-70.14, 42.22)),
    (36,       'Mar 15 (-4.5d)\nDrifting North on\n+6.5 m/s Southerly Wind', (-70.42, 41.82)),
    (len(mean_lon_hp)-1, 'Mar 12 (-7.6d)\nInferred Death Location\n(Western Cape Cod Bay)', (-70.75, 42.16))
]
for st_idx, txt, xy_text in waypoint_steps:
    pt_lon = mean_lon_hp[st_idx]
    pt_lat = mean_lat_hp[st_idx]
    ax1.scatter(pt_lon, pt_lat, s=55, color='#8c2d04', edgecolor='white', lw=1.1, zorder=5.2)
    ax1.annotate(
        txt, xy=(pt_lon, pt_lat), xytext=xy_text,
        arrowprops=dict(arrowstyle='->', color='#333333', lw=1.3),
        bbox=dict(boxstyle='round,pad=0.25', fc='white', ec='#8c2d04', lw=1.1, alpha=0.95),
        fontsize=8.0, fontweight='bold', color='#222222', zorder=6.0
    )

ax1.set_xlabel('Longitude (°W)', fontsize=10)
ax1.set_ylabel('Latitude (°N)', fontsize=10)
ax1.set_title('A. Zoomed Plume: Porpoise NE-2000-1039099 (1:500k Coast)', fontsize=10.6, fontweight='bold', loc='left')
ax1.legend(loc='lower left', fontsize=7.7, framealpha=0.95)
ax1.grid(True, linestyle='--', alpha=0.3)

ax2 = fig.add_subplot(gs[0, 1])
w_sub = env_raw[(env_raw['date'] >= '2000-03-11') & (env_raw['date'] <= '2000-03-19')].copy().sort_values('date')
dates_str = w_sub['date'].dt.strftime('%b %d')
x_d = np.arange(len(w_sub))

ax2.axvspan(2.5, 5.5, color='#1b9e77', alpha=0.12, label='Phase 1 (Mar 14–16): Southerly Wind Pushes Carcass North')
ax2.axvspan(5.5, 8.2, color='#d95f02', alpha=0.15, label='Phase 2 (Mar 17–19): Northerly Gale Slams Carcass South to Barnstable')
ax2.axhline(0, color='#333333', linestyle='--', lw=1.0)

ax2.plot(x_d, w_sub['bay_wind_v'], marker='o', lw=2.8, color='#d95f02', label='N/S Wind Speed: bay_wind_v (+ = Blowing North, - = Blowing South)')
ax2.plot(x_d, w_sub['bay_wind_u'], marker='s', lw=2.0, linestyle=':', color='#2b83ba', label='E/W Wind Speed: bay_wind_u (+ = Blowing East, - = Blowing West)')

for i, v_val in enumerate(w_sub['bay_wind_v']):
    offset = 1.15 if v_val >= 0 else -1.65
    ax2.text(i, v_val + offset, f"{v_val:+.1f} m/s", ha='center', va='center', fontsize=8.1, fontweight='bold', color='#8c2d04')

ax2.annotate(
    'Southerly Peak (+6.5 m/s)\nPushes carcass toward\nProvincetown Hook',
    xy=(5, w_sub['bay_wind_v'].iloc[5]), xytext=(1.6, 6.2),
    arrowprops=dict(arrowstyle='->', color='#0f6e52', lw=1.4),
    bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='#1b9e77', lw=1.1, alpha=0.95),
    fontsize=8.1, fontweight='bold', color='#0f6e52'
)
ax2.annotate(
    'Northerly Gale (-12.1 m/s)\nReverses drift & beaches\ncarcass on Barnstable shore',
    xy=(6, w_sub['bay_wind_v'].iloc[6]), xytext=(3.8, -15.4),
    arrowprops=dict(arrowstyle='->', color='#a64200', lw=1.4),
    bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='#d95f02', lw=1.1, alpha=0.95),
    fontsize=8.1, fontweight='bold', color='#a64200'
)

ax2.set_xticks(x_d)
ax2.set_xticklabels(dates_str, fontsize=8.8)
ax2.set_ylim(-17.8, 16.5)
ax2.set_ylabel('10m Wind Velocity in Cape Cod Bay (m/s)', fontsize=10)
ax2.set_xlabel('Date in March 2000 (Raw Data: environmental_daily.parquet)', fontsize=10)
ax2.set_title('B. Raw Weather Proof: Why NE-2000-1039099 Made a V-Turn', fontsize=10.6, fontweight='bold', loc='left')
ax2.legend(loc='upper left', fontsize=7.8, framealpha=0.95)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.grid(True, linestyle='--', alpha=0.35)

fig.suptitle('Figure 5: Step-by-Step Verification of a Reverse Trajectory Against Raw Historical Wind Records',
             fontsize=13, fontweight='bold', y=0.99)
save_and_copy(fig, 's2_fig05_porpoise_vturn_weather_verification.png')

# ==============================================================================
# FIX FIGURE 8 & FIGURE 9: Clean spacing of callout box & panel titles
# ==============================================================================
xx_h, yy_h = np.meshgrid(np.linspace(-71.45, -68.60, 160), np.linspace(40.40, 43.00, 160))
grid_pts_h = np.vstack([xx_h.ravel(), yy_h.ravel()])

def plot_kde_on_1to500k(ax, mask_bool, cmap_name, line_col, bw=0.16):
    pts = np.vstack([origin_lon[mask_bool].ravel(), origin_lat[mask_bool].ravel()])
    kde = gaussian_kde(pts, bw_method=bw)
    zz = kde(grid_pts_h).reshape(xx_h.shape)
    zz_m = mask_kde_on_real_land(xx_h, yy_h, zz)
    lvls = np.nanpercentile(zz_m, [55, 70, 82, 90, 96, 99.5])
    ax.contourf(xx_h, yy_h, zz_m, levels=lvls, cmap=cmap_name, alpha=0.68, extend='max', zorder=2.0)
    ax.contour(xx_h, yy_h, zz_m, levels=lvls, colors=line_col, linewidths=0.55, alpha=0.55, zorder=2.1)

fig = plt.figure(figsize=(16.4, 7.6), dpi=180)
gs = gridspec.GridSpec(1, 2, wspace=0.25)

ax1 = fig.add_subplot(gs[0, 0])
draw_real_coast(ax1, xlim=(-71.40, -68.80), ylim=(40.80, 42.95))
hp_m = (coastal['drifter_cohort'] == '3. Harbor Porpoise Carcass').values
plot_kde_on_1to500k(ax1, hp_m, 'Oranges', '#8c2d04', bw=0.15)
ax1.scatter(coastal.loc[hp_m, 'lon_clean'], coastal.loc[hp_m, 'lat_clean'],
            s=14, facecolors='none', edgecolors='#2b83ba', linewidths=0.7, alpha=0.70, zorder=3.5,
            label='Beach Discovery Sites (o)')
ax1.scatter(coastal.loc[hp_m, 'origin_lon_mean'], coastal.loc[hp_m, 'origin_lat_mean'],
            s=14, c='#d95f02', alpha=0.60, zorder=4.0,
            label='Inferred At-Sea Mortality Origins (•)')
ax1.annotate('Stellwagen Bank & Mass Bay (43.2%)\n+ North Shore / Jeffreys Ledge (22.8%)\n+ Cape Cod Bay (25.7%)\n= 91.7% Northern Shelf Origin\n(March–May Spring Sink-Gillnet Window)',
             xy=(-70.32, 42.25), xytext=(-71.35, 42.45),
             arrowprops=dict(arrowstyle='->', color='#8c2d04', lw=1.6),
             bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='#d95f02', lw=1.3, alpha=0.95),
             fontsize=8.2, fontweight='bold', color='#8c2d04', zorder=6)
ax1.set_xlabel('Longitude (°W)', fontsize=10)
ax1.set_ylabel('Latitude (°N)', fontsize=10)
ax1.set_title('A. Harbor Porpoise Carcasses (n=618 | Drift: 7.4d, 48.0 km)', fontsize=10.6, fontweight='bold', loc='left')
ax1.legend(loc='lower right', fontsize=8.3, framealpha=0.95)
ax1.grid(True, linestyle='--', alpha=0.3)

ax2 = fig.add_subplot(gs[0, 1])
draw_real_coast(ax2, xlim=(-71.40, -68.80), ylim=(40.80, 42.95))
pd_m = (coastal['drifter_cohort'] == '4. Pelagic Dolphin Carcass').values
plot_kde_on_1to500k(ax2, pd_m, 'YlGnBu', '#0c2c84', bw=0.15)
ax2.scatter(coastal.loc[pd_m, 'lon_clean'], coastal.loc[pd_m, 'lat_clean'],
            s=12, facecolors='none', edgecolors='#d95f02', linewidths=0.65, alpha=0.60, zorder=3.5,
            label='Beach Discovery Sites (o)')
ax2.scatter(coastal.loc[pd_m, 'origin_lon_mean'], coastal.loc[pd_m, 'origin_lat_mean'],
            s=12, c='#1d91c0', alpha=0.50, zorder=4.0,
            label='Inferred At-Sea Mortality Origins (•)')
ax2.annotate('Primary Offshore Source:\nStellwagen Bank & Mass Bay (35.6%)\n& Outer Race Point / CCB Mouth (32.2%)',
             xy=(-70.28, 42.14), xytext=(-71.35, 42.48),
             arrowprops=dict(arrowstyle='->', color='#0c2c84', lw=1.6),
             bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='#1d91c0', lw=1.3, alpha=0.95),
             fontsize=8.2, fontweight='bold', color='#0c2c84', zorder=6)
ax2.annotate('Southern Shelf & Sound\nSource (16.1%): Supplies\nMV & Nantucket',
             xy=(-70.45, 41.22), xytext=(-71.35, 40.90),
             arrowprops=dict(arrowstyle='->', color='#0c2c84', lw=1.5),
             bbox=dict(boxstyle='round,pad=0.28', fc='white', ec='#1d91c0', lw=1.2, alpha=0.95),
             fontsize=8.0, fontweight='bold', color='#0c2c84', zorder=6)
ax2.set_xlabel('Longitude (°W)', fontsize=10)
ax2.set_ylabel('Latitude (°N)', fontsize=10)
ax2.set_title('B. Pelagic Dolphin Carcasses (n=1,344 | Drift: 6.9d, 48.7 km)', fontsize=10.6, fontweight='bold', loc='left')
ax2.legend(loc='lower right', fontsize=8.3, framealpha=0.95)
ax2.grid(True, linestyle='--', alpha=0.3)

fig.suptitle('Figure 8: Inferred At-Sea Mortality Hotspots for Harbor Porpoises and Decomposed Pelagic Dolphins (1:500k Coastline)',
             fontsize=13, fontweight='bold', y=0.98)
save_and_copy(fig, 's2_fig08_porpoise_and_dolphin_hotspots.png')

# ==============================================================================
# FIX FIGURE 9: Clean title separation (wspace=0.26, concise titles)
# ==============================================================================
fig = plt.figure(figsize=(16.5, 7.8), dpi=180)
gs = gridspec.GridSpec(1, 2, wspace=0.26)

ax1 = fig.add_subplot(gs[0, 0])
draw_real_coast(ax1, xlim=(-71.40, -68.80), ylim=(40.80, 42.95))
fi_m = (coastal['drifter_cohort'] == '1. Confirmed Fishery / Entanglement').values
plot_kde_on_1to500k(ax1, fi_m, 'RdPu', '#7a0177', bw=0.18)

ax1.scatter(coastal.loc[fi_m, 'lon_clean'], coastal.loc[fi_m, 'lat_clean'],
            s=28, facecolors='white', edgecolors='#7a0177', marker='o', linewidths=1.0, alpha=0.85, zorder=3.8,
            label='Beach Discovery Sites (o, n=112)')
ax1.scatter(coastal.loc[fi_m, 'origin_lon_mean'], coastal.loc[fi_m, 'origin_lat_mean'],
            s=42, c='#e7298a', edgecolors='black', marker='^', linewidths=0.45, alpha=0.92, zorder=4.5,
            label='Inferred Entanglement Origins (▲, n=112)')

ax1.annotate('Cape Cod Bay (31.2%) &\nStellwagen / Mass Bay (24.1%)\n+ North Shore / Jeffreys (11.6%)\nFixed-Gear & Sink-Gillnet Grounds',
             xy=(-70.30, 42.02), xytext=(-71.35, 42.45),
             arrowprops=dict(arrowstyle='->', color='#7a0177', lw=1.6),
             bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='#e7298a', lw=1.3, alpha=0.95),
             fontsize=8.2, fontweight='bold', color='#7a0177', zorder=6)
ax1.annotate('Southern Shelf & Sounds (18.8%)\n& Outer Cape / Nauset (9.8%)',
             xy=(-70.45, 41.32), xytext=(-71.35, 40.92),
             arrowprops=dict(arrowstyle='->', color='#7a0177', lw=1.5),
             bbox=dict(boxstyle='round,pad=0.28', fc='white', ec='#e7298a', lw=1.2, alpha=0.95),
             fontsize=8.0, fontweight='bold', color='#7a0177', zorder=6)

ax1.set_xlabel('Longitude (°W)', fontsize=10)
ax1.set_ylabel('Latitude (°N)', fontsize=10)
ax1.set_title('A. Confirmed Fishery Entanglements (n=112 | Drift: 3.7d, 34.2 km)', fontsize=10.6, fontweight='bold', loc='left')
ax1.legend(loc='lower right', fontsize=8.2, framealpha=0.95)
ax1.grid(True, linestyle='--', alpha=0.3)

ax2 = fig.add_subplot(gs[0, 1])
draw_real_coast(ax2, xlim=(-71.40, -68.80), ylim=(40.80, 42.95))
vs_m = (coastal['drifter_cohort'] == '2. Confirmed Vessel Strike').values
vs_df = coastal.loc[vs_m].copy()

plot_kde_on_1to500k(ax2, vs_m, 'YlOrBr', '#8c2d04', bw=0.22)

for _, r in vs_df.iterrows():
    dlon = r['origin_lon_mean'] - r['lon_clean']
    dlat = r['origin_lat_mean'] - r['lat_clean']
    if np.hypot(dlon, dlat) > 0.03:
        ax2.annotate('', xy=(r['origin_lon_mean'], r['origin_lat_mean']),
                     xytext=(r['lon_clean'], r['lat_clean']),
                     arrowprops=dict(arrowstyle='->', color='#555555', lw=1.0, alpha=0.65, shrinkA=2, shrinkB=3),
                     zorder=3.8)

ax2.scatter(vs_df['lon_clean'], vs_df['lat_clean'],
            s=42, facecolors='white', edgecolors='#333333', marker='o', linewidths=1.2, alpha=0.9, zorder=4.2,
            label='Beach Discovery Site (o, n=23)')
ax2.scatter(vs_df['origin_lon_mean'], vs_df['origin_lat_mean'],
            s=85, c='#ffd92f', edgecolors='black', marker='D', linewidths=0.9, zorder=5.0,
            label='Backtracked Vessel Strike Origin (♦, n=23)')

ax2.annotate('Cluster 1 (n=8, 34.8%):\nBoston Harbor Approaches,\nNorth Shore (Gloucester/Marblehead)\n& N. Stellwagen / Jeffreys Ledge',
             xy=(-70.65, 42.45), xytext=(-71.36, 42.50),
             arrowprops=dict(arrowstyle='->', color='#8c2d04', lw=1.6),
             bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='#d95f02', lw=1.3, alpha=0.96),
             fontsize=8.0, fontweight='bold', color='#8c2d04', zorder=6)

ax2.annotate('Cluster 2 (n=5, 21.7%):\nWestern Mass Bay (Plymouth) &\nCape Cod Canal East Entrance\n(Plus n=2 off Chatham Outer Cape)',
             xy=(-70.48, 41.84), xytext=(-71.36, 41.88),
             arrowprops=dict(arrowstyle='->', color='#8c2d04', lw=1.6),
             bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='#d95f02', lw=1.3, alpha=0.96),
             fontsize=8.0, fontweight='bold', color='#8c2d04', zorder=6)

ax2.annotate('Cluster 3 (n=8, 34.8%):\nBuzzards Bay (Canal West Approach),\nVineyard/Nantucket Ferry Corridors\n& Southern Shelf',
             xy=(-70.58, 41.42), xytext=(-71.36, 40.92),
             arrowprops=dict(arrowstyle='->', color='#8c2d04', lw=1.6),
             bbox=dict(boxstyle='round,pad=0.28', fc='white', ec='#d95f02', lw=1.3, alpha=0.96),
             fontsize=7.9, fontweight='bold', color='#8c2d04', zorder=6)

ax2.set_xlabel('Longitude (°W)', fontsize=10)
ax2.set_ylabel('Latitude (°N)', fontsize=10)
ax2.set_title('B. Confirmed Vessel Strikes & Rewind Vectors (n=23 | Drift: 4.3d, 36.2 km)', fontsize=10.6, fontweight='bold', loc='left')
ax2.legend(loc='lower right', fontsize=8.2, framealpha=0.95)
ax2.grid(True, linestyle='--', alpha=0.3)

fig.suptitle('Figure 9: Separating Anthropogenic Mortality Hotspots — Fishery Entanglements vs. Vessel Strikes (1:500k Coastline)',
             fontsize=13, fontweight='bold', y=0.98)
save_and_copy(fig, 's2_fig09_hi_fishery_and_vessel_strike_hotspots.png')
