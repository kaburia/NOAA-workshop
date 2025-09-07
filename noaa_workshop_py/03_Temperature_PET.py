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

# # Focus Area 3: Temperature Quality, Microclimates & PET
#
# **Objective**: Show resolution impact (CBAM 4 km vs ERA5 0.25°), compute PET (Hargreaves) for agri/health stress, validate vs ground station.
#
# Expected columns:
#   Ground temp CSV: date,station_id,lat,lon,tmean,tmax,tmin
#   ERA5/CBAM NetCDF vars: t2m,tmax,tmin (K)

# ## CONFIG
LOC = dict(lat=0.15, lon=37.30)  # Mt. Kenya flank
TEMP_WINDOW = ("2021-01-01","2021-12-31")
REGION = "EA"

PATHS = {
    "ground_temp": f"data/ground/tahmo_temp_daily_{REGION}.csv",
    "era5_temp":   "data/reanalysis/era5_t2m_daily.nc",
    "cbam_temp":   "data/reanalysis/cbam_t2m_daily.nc",
}

# ## Imports & Utils
import os, warnings
import numpy as np, pandas as pd, xarray as xr, matplotlib.pyplot as plt
from scipy.stats import pearsonr

warnings.filterwarnings("ignore"); plt.rcParams["figure.figsize"]=(8,4)

def ensure_cols(df, cols):
    miss=[c for c in cols if c not in df.columns]
    if miss: raise KeyError(f"Missing columns: {miss}")

def to_C(K): return K - 273.15

def rmse(a,b): return float(np.sqrt(np.nanmean((np.asarray(a)-np.asarray(b))**2)))

def safe_open_dataset(p):
    if not os.path.exists(p): print(f"[WARN] Missing: {p}"); return None
    try: return xr.open_dataset(p)
    except Exception as e: print(f"[WARN] Open fail {p}: {e}"); return None

def pet_hargreaves(tmin, tmax, tmean, Ra=15.0):
    dtr = np.maximum(tmax - tmin, 0)
    return 0.0023 * Ra * (tmean + 17.8) * np.sqrt(dtr)

# ## Load
era = safe_open_dataset(PATHS['era5_temp'])
cbm = safe_open_dataset(PATHS['cbam_temp'])

if (era is not None) and (cbm is not None):
    e = {k: to_C(era[k].sel(lat=LOC['lat'], lon=LOC['lon'], method='nearest').to_pandas())
         for k in ['t2m','tmax','tmin'] if k in era}
    c = {k: to_C(cbm[k].sel(lat=LOC['lat'], lon=LOC['lon'], method='nearest').to_pandas())
         for k in ['t2m','tmax','tmin'] if k in cbm}
else:
    e, c = {}, {}; print("[FA3] Missing ERA5/CBAM temperature datasets.")

gtemp = pd.read_csv(PATHS['ground_temp'], parse_dates=['date']).set_index('date').sort_index()
ensure_cols(gtemp.reset_index(), ['date','station_id','lat','lon','tmean','tmax','tmin'])

merged = pd.DataFrame({
    'g': gtemp['tmean'].asfreq('D'),
    'cbam': c.get('t2m'),
    'era5': e.get('t2m')
}).dropna().loc[TEMP_WINDOW[0]:TEMP_WINDOW[1]]

for name in ['cbam','era5']:
    if name in merged:
        R = pearsonr(merged['g'], merged[name])[0]
        print(f"[FA3] {name.upper()} r={R:.2f} RMSE={rmse(merged[name], merged['g']):.2f} °C")

# Plots
plt.figure(); plt.plot(merged.index, merged['g'], label='Ground')
if 'cbam' in merged: plt.plot(merged.index, merged['cbam'], label='CBAM')
if 'era5' in merged: plt.plot(merged.index, merged['era5'], label='ERA5')
plt.legend(); plt.title('Tmean validation'); plt.ylabel('°C'); plt.grid(True); plt.show()

# PET + stress
pet_c = pet_hargreaves(c['tmin'], c['tmax'], c['t2m'])
pet_e = pet_hargreaves(e['tmin'], e['tmax'], e['t2m'])
plt.figure(); plt.plot(pet_c.loc[TEMP_WINDOW[0]:TEMP_WINDOW[1]], label='PET CBAM')
plt.plot(pet_e.loc[TEMP_WINDOW[0]:TEMP_WINDOW[1]], label='PET ERA5')
plt.legend(); plt.title('PET comparison'); plt.ylabel('mm/day'); plt.grid(True); plt.show()

stress = (pet_c>5) & (c['tmax']>32)
print("[FA3] Heat/Agri stress days (CBAM):", int(stress.loc[TEMP_WINDOW[0]:TEMP_WINDOW[1]].sum()))
