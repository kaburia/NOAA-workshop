# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.3
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# # Focus Area 5: Sub-Seasonal to Seasonal (S2S) — Dry Spells & Probabilities
#
# **Objective**: Compute ensemble probability of a dry spell (≥5 consecutive days <1 mm) in a 20-day window, and assess reliability with Brier Score.
#
# Expected datasets: Salient S2S ensemble daily precip NetCDF with dims member,time,lat,lon and var 'precip'.

# ## CONFIG
REGION = "HoA"
S2S_POINT = {"EA": (2.0,45.5), "HoA": (7.0,40.5), "SAF": (-14.5,34.3)}[REGION]
S2S_WINDOW = ("2019-01-01","2019-03-31")
DRY_LEN = 5
DRY_THRESH = 1.0

PATHS = {
    "s2s_salient": "data/s2s/salient_precip_daily_ens.nc",
    "ground_rain": f"data/ground/tahmo_daily_rain_{REGION}.csv",
}

# ## Imports & Utils
import os, warnings
import numpy as np, pandas as pd, xarray as xr, matplotlib.pyplot as plt

warnings.filterwarnings("ignore"); plt.rcParams["figure.figsize"]=(8,4)

def dry_spell_bool(series, k=5, thresh=1.0):
    x = (pd.Series(series).fillna(0) < thresh).astype(int).values
    if len(x) < k: return False
    run = np.convolve(x, np.ones(k, dtype=int), 'valid')
    return bool((run == k).any())

def brier_score(prob, obs):
    prob = np.asarray(prob); obs = np.asarray(obs).astype(float)
    return float(np.mean((prob-obs)**2))

def safe_open_dataset(p):
    if not os.path.exists(p): print(f"[WARN] Missing: {p}"); return None
    try: return xr.open_dataset(p)
    except Exception as e: print(f"[WARN] Open fail {p}: {e}"); return None

def nearest_station_series(ground_df, lat, lon, value_col):
    df = ground_df.copy()
    df['dist'] = (df['lat']-lat).abs() + (df['lon']-lon).abs()
    sid = df.loc[df['dist'].idxmin(),'station_id']
    s = df[df['station_id']==sid].set_index('date').sort_index()[value_col].asfreq('D')
    return s, sid

# ## Load
s2s = safe_open_dataset(PATHS['s2s_salient'])
if s2s is None:
    raise SystemExit("[FA5] Salient S2S dataset missing.")

lat, lon = S2S_POINT
ens_pt = s2s['precip'].sel(lat=lat, lon=lon, method='nearest').to_pandas()
members = ens_pt.index.get_level_values('member').unique()
times   = ens_pt.index.get_level_values('time').unique()

# Example 20-day block probability from start of window
t0 = pd.to_datetime(S2S_WINDOW[0])
block = pd.date_range(t0, t0+pd.Timedelta(days=19), freq='D')
flags=[]
for m in members:
    s = ens_pt.xs(m, level='member').reindex(block).interpolate().fillna(0)
    flags.append(dry_spell_bool(s, k=DRY_LEN, thresh=DRY_THRESH))
p_block = np.mean(flags)
print(f"[FA5] P(dry spell in next 20d) at ({lat},{lon}): {p_block*100:.0f}%")

# Reliability vs observed point
obs = pd.read_csv(PATHS['ground_rain'], parse_dates=['date'])
obs = obs.set_index('date').sort_index()
gpt, sid = nearest_station_series(obs.reset_index(), lat, lon, 'precip')

starts = pd.date_range(S2S_WINDOW[0], S2S_WINDOW[1], freq='7D')
P, O = [], []
for t0 in starts:
    block = pd.date_range(t0, t0+pd.Timedelta(days=19))
    flags=[]
    for m in members:
        s = ens_pt.xs(m, level='member').reindex(block).interpolate().fillna(0)
        flags.append(dry_spell_bool(s, DRY_LEN, DRY_THRESH))
    P.append(np.mean(flags))
    O.append(1 if dry_spell_bool(gpt.reindex(block).interpolate().fillna(0), DRY_LEN, DRY_THRESH) else 0)

BS = brier_score(np.array(P), np.array(O))
print(f"[FA5] Brier Score: {BS:.3f}")
plt.scatter(P, O); plt.xlabel('Forecast probability'); plt.ylabel('Observed (0/1)')
plt.title('Dry Spell Reliability (toy)'); plt.grid(True); plt.show()
