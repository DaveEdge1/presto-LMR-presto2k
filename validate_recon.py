"""
Instrumental validation for LMR reconstruction results.

Compares reconstruction GMST and spatial fields against multiple instrumental
datasets (GISTEMP, HadCRUT5) and the published LMRv2.1, computing both
correlation (R) and coefficient of efficiency (CE).

Modeled after the PReSto2k validation notebook:
  LinkedEarth/presto2k_cfr_pb  notebooks/validation/C03_a_validating_PReSto2k.ipynb

Run inside davidedge/lmr2:latest Docker container.
"""

import os
import csv
import json
import numpy as np
import xarray as xr

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import cartopy.crs as ccrs
import cartopy.feature as cfeature

import cfr

# ── Configuration ────────────────────────────────────────────────────────────
RECON_DIR    = os.environ.get('RECON_DIR', '/recons')
OUT_DIR      = os.environ.get('VALIDATION_DIR', '/validation')
LMR_V21_PATH = os.environ.get(
    'LMR_V21_PATH', '/reference_data/gmt_MCruns_ensemble_full_LMRv2.1.nc')
VALID_START  = 1880
VALID_END    = 2000
ANOM_PERIOD  = [1951, 1980]

os.makedirs(OUT_DIR, exist_ok=True)


# ═══════════════════════════════════════════════════════════════════════════
# Utility functions
# ═══════════════════════════════════════════════════════════════════════════

def area_weighted_mean(da):
    """Area-weighted spatial mean of a DataArray with lat/lon dims."""
    wgts = np.cos(np.deg2rad(da['lat']))
    return float(da.weighted(wgts).mean(('lat', 'lon')).values)


def ensts_to_1d(ensts):
    """Extract a 1D time series from an EnsTS (uses median across ensemble)."""
    time = np.asarray(ensts.time)
    val = np.asarray(ensts.value)
    if val.ndim == 2:
        val_1d = np.nanmedian(val, axis=1)
    else:
        val_1d = val
    return time, val_1d


def coefficient_of_efficiency(obs, pred):
    """Nash-Sutcliffe coefficient of efficiency (CE).
    CE = 1 - sum((obs-pred)^2) / sum((obs-mean(obs))^2)
    Perfect reconstruction = 1, climatology = 0, worse than climatology < 0.
    """
    mask = np.isfinite(obs) & np.isfinite(pred)
    if mask.sum() < 5:
        return float('nan')
    o, p = obs[mask], pred[mask]
    ss_res = np.sum((o - p) ** 2)
    ss_tot = np.sum((o - np.mean(o)) ** 2)
    if ss_tot == 0:
        return float('nan')
    return float(1.0 - ss_res / ss_tot)


def pearson_r(a, b):
    """Pearson correlation between two arrays over their valid (finite) entries."""
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 5:
        return float('nan')
    return float(np.corrcoef(a[mask], b[mask])[0, 1])


def align_series(time_a, val_a, time_b, val_b, ymin, ymax):
    """Align two time series to common integer years within [ymin, ymax].
    Returns (common_years, vals_a_aligned, vals_b_aligned)."""
    years_a = np.asarray(time_a, dtype=int)
    years_b = np.asarray(time_b, dtype=int)
    common = np.intersect1d(years_a, years_b)
    common = common[(common >= ymin) & (common <= ymax)]
    if len(common) == 0:
        return common, np.array([]), np.array([])
    idx_a = np.searchsorted(years_a, common)
    idx_b = np.searchsorted(years_b, common)
    return common, val_a[idx_a], val_b[idx_b]


def fetch_hadcrut5_gmst():
    """Download HadCRUT5 global annual mean temperature anomaly.
    Returns (years, values) as numpy arrays."""
    import urllib.request
    url = ('https://www.metoffice.gov.uk/hadobs/hadcrut5/data/'
           'HadCRUT.5.0.2.0/analysis/diagnostics/'
           'HadCRUT.5.0.2.0.analysis.summary_series.global.annual.csv')
    print(f'  Downloading HadCRUT5 from {url} ...')
    try:
        response = urllib.request.urlopen(url, timeout=60)
        lines = response.read().decode('utf-8').strip().split('\n')
    except Exception as e:
        print(f'  WARNING: Failed to download HadCRUT5: {e}')
        return None, None

    # Parse CSV: columns are Time, Anomaly (deg C), ...
    years, vals = [], []
    for line in lines[1:]:  # skip header
        parts = line.split(',')
        try:
            years.append(int(float(parts[0])))
            vals.append(float(parts[1]))
        except (ValueError, IndexError):
            continue
    return np.array(years), np.array(vals)


# ═══════════════════════════════════════════════════════════════════════════
# 1. Load reconstruction
# ═══════════════════════════════════════════════════════════════════════════
print(f'Loading reconstruction from {RECON_DIR} ...')
res = cfr.ReconRes(RECON_DIR)
res.load(['tas', 'tas_gm'], verbose=True)

recon_tas = res.recons['tas']      # ClimateField (ensemble-mean spatial field)
recon_gm  = res.recons['tas_gm']   # EnsTS (global mean, full ensemble)

recon_time = np.asarray(recon_gm.time)
recon_val  = np.asarray(recon_gm.value)
if recon_val.ndim == 1:
    recon_val = recon_val.reshape(-1, 1)
recon_median = np.nanmedian(recon_val, axis=1)

# ═══════════════════════════════════════════════════════════════════════════
# 2. Fetch instrumental observations
# ═══════════════════════════════════════════════════════════════════════════
print('Fetching GISTEMP observations ...')
obs = cfr.ClimateField().fetch('gistemp1200_ERSSTv4', vn='tempanomaly')
obs = obs.get_anom(ref_period=ANOM_PERIOD)
obs = obs.annualize(months=list(range(1, 13)))
obs_gm = obs.geo_mean()
obs_time, obs_1d = ensts_to_1d(obs_gm)

# Build GISTEMP EnsTS for cfr.compare()
gis_values = obs_1d[:, np.newaxis] if obs_1d.ndim == 1 else obs_1d
gis_ensts = cfr.EnsTS(time=obs_time, value=gis_values,
                       value_name='Temperature Anomaly')

print('Fetching HadCRUT5 observations ...')
had_time, had_vals = fetch_hadcrut5_gmst()
has_hadcrut = had_time is not None and len(had_time) > 0
if has_hadcrut:
    had_values = had_vals[:, np.newaxis]
    had_ensts = cfr.EnsTS(time=had_time, value=had_values,
                          value_name='Temperature Anomaly')
    print(f'  HadCRUT5: {len(had_time)} years ({had_time.min()}-{had_time.max()})')

# ═══════════════════════════════════════════════════════════════════════════
# 3. Load LMRv2.1 reference
# ═══════════════════════════════════════════════════════════════════════════
lmr_v21_time = None
lmr_v21_median = None
lmr_v21_ensts = None
if os.path.exists(LMR_V21_PATH):
    print(f'Loading published LMRv2.1 GMST from {LMR_V21_PATH} ...')
    lmr_v21 = xr.open_dataset(LMR_V21_PATH)
    gmt = lmr_v21['gmt']
    ens_dims = [d for d in ('MCrun', 'members') if d in gmt.dims]
    if ens_dims:
        gmt_ens = gmt.stack(ensemble=ens_dims)
    else:
        gmt_ens = gmt
    ens_arr = np.asarray(gmt_ens.values)

    raw_time = lmr_v21['time'].values
    try:
        lmr_v21_time = np.array([int(t.year) for t in raw_time])
    except AttributeError:
        lmr_v21_time = np.asarray(raw_time, dtype=float).astype(int)

    lmr_v21_median = np.nanmedian(ens_arr, axis=1)
    lmr_v21_q05    = np.nanquantile(ens_arr, 0.05, axis=1)
    lmr_v21_q95    = np.nanquantile(ens_arr, 0.95, axis=1)
    lmr_v21_ensts  = cfr.EnsTS(time=lmr_v21_time, value=ens_arr,
                                value_name='GMSTa')
    print(f'  LMRv2.1 GMST: {len(lmr_v21_time)} years, '
          f'{ens_arr.shape[1]} ensemble members')
else:
    print(f'WARNING: LMRv2.1 reference not found at {LMR_V21_PATH}')


# ═══════════════════════════════════════════════════════════════════════════
# 4. GMST Instrumental Validation (R and CE)
# ═══════════════════════════════════════════════════════════════════════════
print(f'\nComputing GMST validation metrics ({VALID_START}-{VALID_END}) ...')

# Collect all results: {dataset_name: {recon_name: {R, CE}}}
gmst_results = {}


def compute_gmst_stats(recon_name, recon_t, recon_v, ref_name, ref_t, ref_v):
    """Compute R and CE between two GMST time series."""
    _, ra, rb = align_series(recon_t, recon_v, ref_t, ref_v,
                             VALID_START, VALID_END)
    r_val = pearson_r(ra, rb)
    ce_val = coefficient_of_efficiency(rb, ra)  # obs=ref, pred=recon
    print(f'  {recon_name} vs {ref_name}: R={r_val:.4f}, CE={ce_val:.4f}')
    if ref_name not in gmst_results:
        gmst_results[ref_name] = {}
    gmst_results[ref_name][recon_name] = {'R': r_val, 'CE': ce_val}
    return r_val, ce_val


recon_years = recon_time.astype(int)

# vs GISTEMP
compute_gmst_stats('Custom Recon', recon_years, recon_median,
                   'GISTEMP', obs_time.astype(int), obs_1d)
if lmr_v21_time is not None:
    compute_gmst_stats('LMRv2.1', lmr_v21_time, lmr_v21_median,
                       'GISTEMP', obs_time.astype(int), obs_1d)

# vs HadCRUT5
if has_hadcrut:
    compute_gmst_stats('Custom Recon', recon_years, recon_median,
                       'HadCRUT5', had_time, had_vals)
    if lmr_v21_time is not None:
        compute_gmst_stats('LMRv2.1', lmr_v21_time, lmr_v21_median,
                           'HadCRUT5', had_time, had_vals)

# Consensus (mean of available instrumental datasets)
consensus_refs = [('GISTEMP', obs_time.astype(int), obs_1d)]
if has_hadcrut:
    consensus_refs.append(('HadCRUT5', had_time, had_vals))

if len(consensus_refs) > 1:
    # Align all instrumental datasets to common years
    all_years = consensus_refs[0][1]
    for _, t, _ in consensus_refs[1:]:
        all_years = np.intersect1d(all_years, t)
    all_years = all_years[(all_years >= VALID_START) & (all_years <= VALID_END)]

    if len(all_years) > 10:
        consensus_vals = []
        for _, t, v in consensus_refs:
            idx = np.searchsorted(t.astype(int), all_years)
            consensus_vals.append(v[idx])
        consensus_mean = np.mean(consensus_vals, axis=0)

        compute_gmst_stats('Custom Recon', recon_years, recon_median,
                           'Consensus', all_years, consensus_mean)
        if lmr_v21_time is not None:
            compute_gmst_stats('LMRv2.1', lmr_v21_time, lmr_v21_median,
                               'Consensus', all_years, consensus_mean)

# vs LMRv2.1 (direct recon-to-recon comparison over full overlap)
if lmr_v21_time is not None:
    overlap_start = int(max(recon_years.min(), lmr_v21_time.min()))
    overlap_end   = int(min(recon_years.max(), lmr_v21_time.max()))
    if overlap_end > overlap_start:
        _, ra, rb = align_series(recon_years, recon_median,
                                 lmr_v21_time, lmr_v21_median,
                                 overlap_start, overlap_end)
        lmr_r = pearson_r(ra, rb)
        lmr_ce = coefficient_of_efficiency(rb, ra)
        gmst_results['LMRv2.1 (direct)'] = {
            'Custom Recon': {'R': lmr_r, 'CE': lmr_ce,
                             'period': f'{overlap_start}-{overlap_end}'}
        }
        print(f'  Custom Recon vs LMRv2.1 (full overlap {overlap_start}-{overlap_end}): '
              f'R={lmr_r:.4f}, CE={lmr_ce:.4f}')


# ═══════════════════════════════════════════════════════════════════════════
# 5. Spatial Validation Maps (Correlation + CE)
# ═══════════════════════════════════════════════════════════════════════════
print(f'\nComputing spatial validation maps ({VALID_START}-{VALID_END}) ...')

# Spatial correlation
corr_field = recon_tas.compare(obs, stat='corr', timespan=[VALID_START, VALID_END])
corr_da = corr_field.da
geo_mean_corr = area_weighted_mean(corr_da)
print(f'  Geographic mean correlation: {geo_mean_corr:.4f}')

# Spatial CE
ce_field = recon_tas.compare(obs, stat='CE', timespan=[VALID_START, VALID_END])
ce_da = ce_field.da
geo_mean_ce = area_weighted_mean(ce_da)
print(f'  Geographic mean CE: {geo_mean_ce:.4f}')

# Plot spatial correlation
fig, ax = plt.subplots(1, 1, figsize=(12, 6),
                       subplot_kw={'projection': ccrs.Robinson()})
corr_da.plot(ax=ax, transform=ccrs.PlateCarree(),
             cmap='RdYlBu_r', vmin=-1, vmax=1,
             cbar_kwargs={'label': 'Correlation (r)',
                          'orientation': 'horizontal',
                          'shrink': 0.7, 'pad': 0.08})
ax.coastlines(linewidth=0.5)
ax.add_feature(cfeature.BORDERS, linewidth=0.3, alpha=0.5)
ax.set_global()
ax.set_title(f'Reconstruction vs GISTEMP Correlation ({VALID_START}-{VALID_END})\n'
             f'Geographic Mean r = {geo_mean_corr:.3f}', fontsize=13)
fig.savefig(os.path.join(OUT_DIR, 'spatial_corr_map.png'),
            dpi=150, bbox_inches='tight')
plt.close(fig)

# Plot spatial CE
fig, ax = plt.subplots(1, 1, figsize=(12, 6),
                       subplot_kw={'projection': ccrs.Robinson()})
ce_da.plot(ax=ax, transform=ccrs.PlateCarree(),
           cmap='RdYlBu_r', vmin=-1, vmax=1,
           cbar_kwargs={'label': 'Coefficient of Efficiency (CE)',
                        'orientation': 'horizontal',
                        'shrink': 0.7, 'pad': 0.08})
ax.coastlines(linewidth=0.5)
ax.add_feature(cfeature.BORDERS, linewidth=0.3, alpha=0.5)
ax.set_global()
ax.set_title(f'Reconstruction vs GISTEMP CE ({VALID_START}-{VALID_END})\n'
             f'Geographic Mean CE = {geo_mean_ce:.3f}', fontsize=13)
fig.savefig(os.path.join(OUT_DIR, 'spatial_ce_map.png'),
            dpi=150, bbox_inches='tight')
plt.close(fig)

# Combined side-by-side spatial maps
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 6),
                               subplot_kw={'projection': ccrs.Robinson()})
corr_da.plot(ax=ax1, transform=ccrs.PlateCarree(),
             cmap='RdYlBu_r', vmin=-1, vmax=1,
             cbar_kwargs={'label': 'r', 'orientation': 'horizontal',
                          'shrink': 0.8, 'pad': 0.08})
ax1.coastlines(linewidth=0.5)
ax1.set_global()
ax1.set_title(f'Correlation (mean r = {geo_mean_corr:.3f})')

ce_da.plot(ax=ax2, transform=ccrs.PlateCarree(),
           cmap='RdYlBu_r', vmin=-1, vmax=1,
           cbar_kwargs={'label': 'CE', 'orientation': 'horizontal',
                        'shrink': 0.8, 'pad': 0.08})
ax2.coastlines(linewidth=0.5)
ax2.set_global()
ax2.set_title(f'Coefficient of Efficiency (mean CE = {geo_mean_ce:.3f})')

fig.suptitle(f'Spatial Validation vs GISTEMP ({VALID_START}-{VALID_END})', fontsize=14)
fig.savefig(os.path.join(OUT_DIR, 'spatial_corr_ce_combined.png'),
            dpi=150, bbox_inches='tight')
plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════
# 6. GMST Time Series Plot (with ensemble spread)
# ═══════════════════════════════════════════════════════════════════════════
print('Generating GMST time series plot ...')

recon_q05 = np.nanquantile(recon_val, 0.05, axis=1)
recon_q95 = np.nanquantile(recon_val, 0.95, axis=1)

fig, ax = plt.subplots(figsize=(14, 5))
ax.fill_between(recon_time, recon_q05, recon_q95,
                alpha=0.3, color='steelblue',
                label='Custom recon (5-95% range)')
ax.plot(recon_time, recon_median, color='steelblue', lw=1.5,
        label='Custom recon (median)')

if lmr_v21_time is not None:
    ax.fill_between(lmr_v21_time, lmr_v21_q05, lmr_v21_q95,
                    alpha=0.25, color='darkorange',
                    label='LMRv2.1 (5-95% range)')
    ax.plot(lmr_v21_time, lmr_v21_median, color='darkorange', lw=1.5,
            label='LMRv2.1 (median)')

ax.plot(obs_time, obs_1d, color='red', lw=1.5, label='GISTEMP', alpha=0.85)

if has_hadcrut:
    ax.plot(had_time, had_vals, color='green', lw=1.5, ls='--',
            label='HadCRUT5', alpha=0.85)

ax.set_xlabel('Year CE')
ax.set_ylabel('Temperature Anomaly (\u00b0C)')
ax.set_title('Global Mean Surface Temperature')
ax.legend(loc='upper left')
t_min = recon_time.min()
if lmr_v21_time is not None:
    t_min = min(t_min, lmr_v21_time.min())
ax.set_xlim(t_min, 2000)
ax.axhline(0, color='gray', lw=0.5, alpha=0.5)
fig.savefig(os.path.join(OUT_DIR, 'gmst_timeseries.png'),
            dpi=150, bbox_inches='tight')
plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════
# 6b. GMST Ensemble Members Plot (all iterations)
# ═══════════════════════════════════════════════════════════════════════════
print('Generating GMST ensemble members plot ...')
n_ens = recon_val.shape[1]

fig, ax = plt.subplots(figsize=(14, 6))

# Plot every ensemble member as a thin translucent line
# Cap at 200 members for readability; if more, subsample evenly
max_lines = 200
if n_ens <= max_lines:
    plot_indices = range(n_ens)
else:
    plot_indices = np.linspace(0, n_ens - 1, max_lines, dtype=int)

for i in plot_indices:
    ax.plot(recon_time, recon_val[:, i], color='steelblue',
            alpha=max(0.03, 3.0 / n_ens), lw=0.4)

# Overlay median and quantiles
ax.fill_between(recon_time, recon_q05, recon_q95,
                alpha=0.15, color='navy', label='5-95% range')
ax.plot(recon_time, recon_median, color='navy', lw=2,
        label='Ensemble median')

# Overlay instrumental
ax.plot(obs_time, obs_1d, color='red', lw=1.5, label='GISTEMP', alpha=0.85)

ax.set_xlabel('Year CE')
ax.set_ylabel('Temperature Anomaly (\u00b0C)')
ax.set_title(f'GMST: All {n_ens} Ensemble Members')
ax.legend(loc='upper left')
t_min = recon_time.min()
ax.set_xlim(t_min, 2000)
ax.axhline(0, color='gray', lw=0.5, alpha=0.5)
fig.savefig(os.path.join(OUT_DIR, 'gmst_ensemble_members.png'),
            dpi=150, bbox_inches='tight')
plt.close(fig)

# Zoomed instrumental-period version
fig, ax = plt.subplots(figsize=(14, 6))
mask_t = (recon_time >= VALID_START) & (recon_time <= VALID_END)
for i in plot_indices:
    ax.plot(recon_time[mask_t], recon_val[mask_t, i], color='steelblue',
            alpha=max(0.05, 5.0 / n_ens), lw=0.5)
ax.fill_between(recon_time[mask_t], recon_q05[mask_t], recon_q95[mask_t],
                alpha=0.15, color='navy', label='5-95% range')
ax.plot(recon_time[mask_t], recon_median[mask_t], color='navy', lw=2,
        label='Ensemble median')
omask = (obs_time >= VALID_START) & (obs_time <= VALID_END)
ax.plot(obs_time[omask], obs_1d[omask], color='red', lw=2,
        label='GISTEMP', alpha=0.85)
if has_hadcrut:
    hmask = (had_time >= VALID_START) & (had_time <= VALID_END)
    ax.plot(had_time[hmask], had_vals[hmask], color='green', lw=2,
            ls='--', label='HadCRUT5', alpha=0.85)
ax.set_xlabel('Year CE')
ax.set_ylabel('Temperature Anomaly (\u00b0C)')
ax.set_title(f'GMST Ensemble Members: Instrumental Period ({VALID_START}-{VALID_END})')
ax.legend(loc='upper left')
ax.grid(True, alpha=0.3)
fig.savefig(os.path.join(OUT_DIR, 'gmst_ensemble_members_instrumental.png'),
            dpi=150, bbox_inches='tight')
plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════
# 7. GMST Difference Plot (recon - LMRv2.1)
# ═══════════════════════════════════════════════════════════════════════════
if lmr_v21_time is not None:
    print('Generating GMST difference plot ...')
    _, recon_aligned, lmr_aligned = align_series(
        recon_years, recon_median, lmr_v21_time, lmr_v21_median,
        int(max(recon_years.min(), lmr_v21_time.min())),
        int(min(recon_years.max(), lmr_v21_time.max())))
    diff_years = np.arange(
        int(max(recon_years.min(), lmr_v21_time.min())),
        int(min(recon_years.max(), lmr_v21_time.max())) + 1)
    diff_years = diff_years[:len(recon_aligned)]
    difference = recon_aligned - lmr_aligned

    fig, ax = plt.subplots(figsize=(14, 4))
    ax.fill_between(diff_years, 0, difference,
                    where=difference >= 0, color='firebrick', alpha=0.5)
    ax.fill_between(diff_years, 0, difference,
                    where=difference < 0, color='steelblue', alpha=0.5)
    ax.plot(diff_years, difference, color='black', lw=0.5, alpha=0.7)
    ax.axhline(0, color='k', ls='--', alpha=0.5)
    ax.set_xlabel('Year CE')
    ax.set_ylabel('Difference (\u00b0C)')
    ax.set_title('GMST Difference: Custom Reconstruction - LMRv2.1\n'
                 '(Red = warmer than LMRv2.1, Blue = cooler)')
    ax.grid(True, alpha=0.3)
    fig.savefig(os.path.join(OUT_DIR, 'gmst_difference.png'),
                dpi=150, bbox_inches='tight')
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════
# 8. Instrumental Period Detail (1880-2000)
# ═══════════════════════════════════════════════════════════════════════════
print('Generating instrumental period detail plot ...')

fig, ax = plt.subplots(figsize=(14, 5))
mask = (recon_time >= VALID_START) & (recon_time <= VALID_END)
ax.fill_between(recon_time[mask], recon_q05[mask], recon_q95[mask],
                alpha=0.3, color='steelblue',
                label='Custom recon (5-95%)')
ax.plot(recon_time[mask], recon_median[mask], color='steelblue', lw=2,
        label='Custom recon (median)')

if lmr_v21_time is not None:
    lmask = (lmr_v21_time >= VALID_START) & (lmr_v21_time <= VALID_END)
    ax.fill_between(lmr_v21_time[lmask], lmr_v21_q05[lmask],
                    lmr_v21_q95[lmask], alpha=0.2, color='darkorange',
                    label='LMRv2.1 (5-95%)')
    ax.plot(lmr_v21_time[lmask], lmr_v21_median[lmask],
            color='darkorange', lw=2, label='LMRv2.1 (median)')

omask = (obs_time >= VALID_START) & (obs_time <= VALID_END)
ax.plot(obs_time[omask], obs_1d[omask], color='red', lw=2,
        label='GISTEMP', alpha=0.85)

if has_hadcrut:
    hmask = (had_time >= VALID_START) & (had_time <= VALID_END)
    ax.plot(had_time[hmask], had_vals[hmask], color='green', lw=2,
            ls='--', label='HadCRUT5', alpha=0.85)

ax.set_xlabel('Year CE')
ax.set_ylabel('Temperature Anomaly (\u00b0C)')
ax.set_title(f'Instrumental Validation Period ({VALID_START}-{VALID_END})')
ax.legend(loc='upper left')
ax.axhline(0, color='gray', lw=0.5, alpha=0.5)
ax.grid(True, alpha=0.3)
fig.savefig(os.path.join(OUT_DIR, 'gmst_instrumental_detail.png'),
            dpi=150, bbox_inches='tight')
plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════
# 9. Save metrics CSV
# ═══════════════════════════════════════════════════════════════════════════
metrics_path = os.path.join(OUT_DIR, 'validation_metrics.csv')
with open(metrics_path, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['metric', 'value'])
    writer.writerow(['geo_mean_spatial_corr', f'{geo_mean_corr:.4f}'])
    writer.writerow(['geo_mean_spatial_CE', f'{geo_mean_ce:.4f}'])
    for ref_name, recons in gmst_results.items():
        for recon_name, stats in recons.items():
            prefix = f'{recon_name}_vs_{ref_name}'
            writer.writerow([f'{prefix}_R', f'{stats["R"]:.4f}'])
            writer.writerow([f'{prefix}_CE', f'{stats["CE"]:.4f}'])
    writer.writerow(['validation_period', f'{VALID_START}-{VALID_END}'])
    writer.writerow(['anom_ref_period', f'{ANOM_PERIOD[0]}-{ANOM_PERIOD[1]}'])
    writer.writerow(['n_ensemble_members', int(recon_val.shape[1])])

# Also save as JSON for programmatic access
json_metrics = {
    'spatial': {'corr_geo_mean': geo_mean_corr, 'CE_geo_mean': geo_mean_ce},
    'gmst': gmst_results,
    'config': {
        'validation_period': [VALID_START, VALID_END],
        'anom_ref_period': ANOM_PERIOD,
        'n_ensemble_members': int(recon_val.shape[1]),
    }
}
with open(os.path.join(OUT_DIR, 'validation_metrics.json'), 'w') as f:
    json.dump(json_metrics, f, indent=2, default=str)


# ═══════════════════════════════════════════════════════════════════════════
# 10. Generate HTML report
# ═══════════════════════════════════════════════════════════════════════════
print('Generating HTML report ...')

# Build GMST results table rows
table_rows = ''
for ref_name in ['GISTEMP', 'HadCRUT5', 'Consensus']:
    if ref_name not in gmst_results:
        continue
    for recon_name in ['Custom Recon', 'LMRv2.1']:
        if recon_name not in gmst_results[ref_name]:
            continue
        stats = gmst_results[ref_name][recon_name]
        r_val = stats['R']
        ce_val = stats['CE']
        # Color CE: green if > 0.5, orange if > 0, red if negative
        if ce_val > 0.5:
            ce_color = '#16a34a'
        elif ce_val > 0:
            ce_color = '#d97706'
        else:
            ce_color = '#dc2626'
        chip_class = 'chip-custom' if recon_name == 'Custom Recon' else 'chip-lmrv21'
        label_class = 'label-custom' if recon_name == 'Custom Recon' else 'label-lmrv21'
        table_rows += f'''    <tr>
      <td><span class="chip {chip_class}"></span><span class="{label_class}">{recon_name}</span></td>
      <td>{ref_name}</td>
      <td>{r_val:.4f}</td>
      <td style="color: {ce_color}; font-weight: 600;">{ce_val:.4f}</td>
    </tr>\n'''

# Direct vs LMRv2.1 row
lmr_direct_row = ''
if 'LMRv2.1 (direct)' in gmst_results:
    d = gmst_results['LMRv2.1 (direct)']['Custom Recon']
    period = d.get('period', '')
    lmr_direct_row = f'''    <tr>
      <td><span class="chip chip-custom"></span><span class="label-custom">Custom Recon</span></td>
      <td><span class="chip chip-lmrv21"></span><span class="label-lmrv21">LMRv2.1 ({period})</span></td>
      <td>{d["R"]:.4f}</td>
      <td>{d["CE"]:.4f}</td>
    </tr>'''

# Determine if we have the difference plot
has_diff_plot = lmr_v21_time is not None

html = f"""<!DOCTYPE html>
<html>
<head>
  <title>LMR Instrumental Validation</title>
  <style>
    :root {{
      --custom: #4682b4;
      --lmrv21: #ff8c00;
      --gistemp: #dc2626;
      --hadcrut: #16a34a;
      --bg: #f7f8fa;
    }}
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
           max-width: 1100px; margin: 0 auto; padding: 24px; color: #1a1a1a;
           background: var(--bg); }}
    h1 {{ border-bottom: 3px solid var(--custom); padding-bottom: 12px; font-size: 1.8rem; }}
    h2 {{ color: #374151; margin-top: 36px; font-size: 1.3rem;
          border-left: 4px solid var(--custom); padding-left: 12px; }}
    p {{ line-height: 1.6; color: #4b5563; }}
    table {{ border-collapse: collapse; margin: 16px 0; width: 100%;
             background: white; border-radius: 8px; overflow: hidden;
             box-shadow: 0 1px 3px rgba(0,0,0,0.08); }}
    th, td {{ border: 1px solid #e5e7eb; padding: 10px 16px; text-align: left; }}
    th {{ background: #f3f4f6; font-weight: 600; font-size: 0.9rem;
          text-transform: uppercase; letter-spacing: 0.03em; color: #6b7280; }}
    img {{ max-width: 100%; margin: 12px 0; border: 1px solid #e5e7eb;
           border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.06); }}
    .back {{ margin-top: 32px; }}
    .label-custom  {{ color: var(--custom);  font-weight: 600; }}
    .label-lmrv21  {{ color: var(--lmrv21);  font-weight: 600; }}
    .label-gistemp {{ color: var(--gistemp); font-weight: 600; }}
    .label-hadcrut {{ color: var(--hadcrut); font-weight: 600; }}
    .chip {{
      display: inline-block; width: 0.75em; height: 0.75em;
      border-radius: 2px; margin-right: 6px; vertical-align: baseline;
    }}
    .chip-custom  {{ background: var(--custom); }}
    .chip-lmrv21  {{ background: var(--lmrv21); }}
    .chip-gistemp {{ background: var(--gistemp); }}
    .chip-hadcrut {{ background: var(--hadcrut); }}
    .section {{ background: white; padding: 24px; border-radius: 8px;
                box-shadow: 0 1px 3px rgba(0,0,0,0.08); margin: 20px 0; }}
    .metric-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                    gap: 16px; margin: 16px 0; }}
    .metric-card {{ background: white; padding: 20px; border-radius: 8px;
                    box-shadow: 0 1px 3px rgba(0,0,0,0.08); text-align: center; }}
    .metric-card .value {{ font-size: 2rem; font-weight: 700; color: var(--custom); }}
    .metric-card .label {{ font-size: 0.85rem; color: #6b7280; margin-top: 4px; }}
  </style>
</head>
<body>
  <h1>Instrumental Validation Report</h1>
  <p>Validation of the custom LMR reconstruction against instrumental observations
     and the published LMRv2.1, following the methodology of
     <a href="https://github.com/LinkedEarth/presto2k_cfr_pb/blob/main/notebooks/validation/C03_a_validating_PReSto2k.ipynb">PReSto2k validation</a>.</p>

  <div class="metric-grid">
    <div class="metric-card">
      <div class="value">{geo_mean_corr:.3f}</div>
      <div class="label">Spatial Correlation (geo. mean)</div>
    </div>
    <div class="metric-card">
      <div class="value">{geo_mean_ce:.3f}</div>
      <div class="label">Spatial CE (geo. mean)</div>
    </div>
    <div class="metric-card">
      <div class="value">{gmst_results.get('GISTEMP', {}).get('Custom Recon', {}).get('R', float('nan')):.3f}</div>
      <div class="label">GMST R vs GISTEMP</div>
    </div>
    <div class="metric-card">
      <div class="value">{gmst_results.get('GISTEMP', {}).get('Custom Recon', {}).get('CE', float('nan')):.3f}</div>
      <div class="label">GMST CE vs GISTEMP</div>
    </div>
  </div>

  <h2>GMST Validation Metrics ({VALID_START}-{VALID_END})</h2>
  <p>Correlation (R) and Coefficient of Efficiency (CE) of the
     ensemble-median GMST against instrumental datasets.
     CE = 1 is perfect; CE = 0 equals climatology; CE &lt; 0 is worse
     than climatology.</p>
  <table>
    <tr><th>Reconstruction</th><th>Reference</th><th>R</th><th>CE</th></tr>
{table_rows}{lmr_direct_row}
  </table>

  <h2>Spatial Validation vs GISTEMP</h2>
  <p>Grid-point correlation and coefficient of efficiency between the
     <span class="label-custom">custom reconstruction</span> and
     <span class="label-gistemp">GISTEMP</span> over {VALID_START}-{VALID_END}.</p>
  <img src="spatial_corr_ce_combined.png" alt="Spatial correlation and CE maps">

  <details>
    <summary>Individual maps</summary>
    <img src="spatial_corr_map.png" alt="Spatial correlation map">
    <img src="spatial_ce_map.png" alt="Spatial CE map">
  </details>

  <h2>GMST Time Series</h2>
  <p><span class="label-custom">Custom reconstruction</span> ensemble spread
     compared against <span class="label-lmrv21">LMRv2.1</span>,
     <span class="label-gistemp">GISTEMP</span>{', and <span class="label-hadcrut">HadCRUT5</span>' if has_hadcrut else ''}.</p>
  <img src="gmst_timeseries.png" alt="GMST time series">

  <h2>GMST Ensemble Members ({n_ens} total)</h2>
  <p>Every ensemble member plotted individually, showing the full spread
     of the reconstruction across all iterations and seeds.</p>
  <img src="gmst_ensemble_members.png" alt="GMST all ensemble members">

  <h3>Instrumental Period ({VALID_START}-{VALID_END})</h3>
  <p>Zoomed view of ensemble members during the instrumental overlap period,
     with <span class="label-gistemp">GISTEMP</span>{' and <span class="label-hadcrut">HadCRUT5</span>' if has_hadcrut else ''}
     overlaid.</p>
  <img src="gmst_ensemble_members_instrumental.png" alt="Ensemble members instrumental period">

  <h2>Instrumental Period Detail</h2>
  <p>Zoomed view of the validation period ({VALID_START}-{VALID_END}) with
     ensemble spread and all reference datasets.</p>
  <img src="gmst_instrumental_detail.png" alt="Instrumental detail">

  {'<h2>GMST Difference (Custom - LMRv2.1)</h2>' if has_diff_plot else ''}
  {'<p>Year-by-year difference between the custom reconstruction and LMRv2.1 ensemble medians. Red = warmer, Blue = cooler.</p>' if has_diff_plot else ''}
  {'<img src="gmst_difference.png" alt="GMST difference plot">' if has_diff_plot else ''}

  <p class="back"><a href="../index.html">&larr; Back to results</a></p>
</body>
</html>"""

with open(os.path.join(OUT_DIR, 'index.html'), 'w') as f:
    f.write(html)

print(f'\nValidation complete. Outputs in {OUT_DIR}/')
print(f'  Plots: spatial_corr_map.png, spatial_ce_map.png, '
      f'spatial_corr_ce_combined.png')
print(f'  Plots: gmst_timeseries.png, gmst_instrumental_detail.png'
      f'{", gmst_difference.png" if has_diff_plot else ""}')
print(f'  Data:  validation_metrics.csv, validation_metrics.json')
print(f'  HTML:  index.html')
