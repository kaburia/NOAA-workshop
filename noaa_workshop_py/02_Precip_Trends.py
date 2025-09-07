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

# # Focus Area 2: Next-Gen Precipitation — Trends & Extremes
#
# **Objective**: Build long-term normals (LTN), analyze trends (Kendall tau), and compare multi-product response during extreme events vs nearest ground station.
# **Case Windows**: Region-configured; edit REGION/CFG/paths as needed.
#
# Expected datasets: CHIRPS/TAMSAT/IMERG/ERA5/CBAM daily precip NetCDF with var 'precip'.

# ## CONFIG
REGION = "EA"
REGION_DEFAULTS = {
    "EA": dict(point=(-1.29,36.82), event=("2025-03-01","2025-03-31")),
    "HoA": dict(point=(8.98,38.80),  event=("2025-08-01","2025-08-31")),
    "SAF": dict(point=(-15.8,35.0),  event=("2023-02-15","2023-03-31")),
}
CFG = REGION_DEFAULTS[REGION]
LTN_RANGE = ("2014-01-01","2024-12-31")

PATHS = {
    "ground_rain":  f"data/ground/tahmo_daily_rain_{REGION}.csv",
    "chirps_daily": "data/satellite/chirps_daily.nc",
    "tamsat_daily": "data/satellite/tamsat_daily.nc",
    "imerg_daily":  "data/satellite/gpm_imerg_daily.nc",
    "era5_precip":  "data/reanalysis/era5_precip_daily.nc",
    "cbam_precip":  "data/reanalysis/cbam_precip_daily.nc",
}

# ## Imports & Utils
import os, warnings
import numpy as np, pandas as pd, xarray as xr, matplotlib.pyplot as plt
from scipy.stats import kendalltau, pearsonr

warnings.filterwarnings("ignore"); plt.rcParams["figure.figsize"]=(8,4)

def ensure_cols(df, cols):
    miss=[c for c in cols if c not in df.columns]
    if miss: raise KeyError(f"Missing columns: {miss}")
    return True

def rmse(a,b): return float(np.sqrt(np.nanmean((np.asarray(a)-np.asarray(b))**2)))
def bias(a,b): return float(np.nanmean(np.asarray(a)-np.asarray(b)))

def safe_open_dataset(p):
    if not os.path.exists(p):
        print(f"[WARN] Missing: {p}"); return None
    try: return xr.open_dataset(p)
    except Exception as e: print(f"[WARN] Open fail {p}: {e}"); return None

def extract_point(ds, var, lat, lon):
    return ds[var].sel(lat=lat, lon=lon, method='nearest').to_pandas()

def nearest_station_series(ground_df, lat, lon, value_col):
    df = ground_df.copy()
    df['dist'] = (df['lat']-lat).abs() + (df['lon']-lon).abs()
    sid = df.loc[df['dist'].idxmin(),'station_id']
    s = df[df['station_id']==sid].set_index('date').sort_index()[value_col].asfreq('D')
    return s, sid

# ## Load
POINT_LAT, POINT_LON = CFG['point']
EVENT = CFG['event']

series={}
for name, path in PATHS.items():
    if name == "ground_rain": continue
    ds = safe_open_dataset(path)
    if ds is None: continue
    var = 'precip' if 'precip' in ds.data_vars else [v for v in ds.data_vars if 'precip' in v.lower()][0]
    series[name] = extract_point(ds, var, POINT_LAT, POINT_LON)

ground = pd.read_csv(PATHS['ground_rain'], parse_dates=['date'])
ensure_cols(ground, ['date','station_id','lat','lon','precip'])
gts, sid = nearest_station_series(ground, POINT_LAT, POINT_LON, 'precip')

# ## LTN & Trends
for name, s in series.items():
    sel = s.loc[LTN_RANGE[0]:LTN_RANGE[1]]
    ann = sel.resample('A').sum(min_count=1)
    ltn = float(ann.mean())
    tau,p = kendalltau(ann.index.year, ann.values)
    print(f"[FA2] {name:6s} | LTN={ltn:.1f} mm/yr | Kendall τ={tau:.2f} (p={p:.2f})")

# ## Event comparison
sl = slice(EVENT[0], EVENT[1])
plt.figure()
plt.plot(gts.loc[sl], label='Ground', lw=2)
for name,s in series.items():
    plt.plot(s.loc[sl], label=name, alpha=0.8)
plt.title('Event rainfall response'); plt.ylabel('mm/day'); plt.legend(); plt.grid(True); plt.show()

for name, s in series.items():
    merged = pd.DataFrame({'g':gts.loc[sl], 'x':s.loc[sl]}).dropna()
    if len(merged) > 10:
        R = pearsonr(merged['g'], merged['x'])[0]
        print(f"[FA2] {name:6s} | RMSE={rmse(merged['x'],merged['g']):.2f} | Bias={bias(merged['x'],merged['g']):.2f} | r={R:.2f}")
