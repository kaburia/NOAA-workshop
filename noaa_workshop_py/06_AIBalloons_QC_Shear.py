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

# # Focus Area 6: AI Balloons (Windborne) — Profiles, QC, Shear
#
# **Objective**: QC vertical profiles, compare with ERA5, and compute simple wind shear diagnostics for operations.
#
# Expected columns:
#   Balloons CSV: time,lat,lon,pressure_hPa,alt_m,temp_C,wind_u,wind_v
#   ERA5 profile NetCDF: temp(time,level), u(time,level), v(time,level), coords level (hPa)

# ## CONFIG
REGION = "EA"
SITE = {"EA": (-1.29,36.82), "HoA": (8.98,38.80), "SAF": (-15.8,35.0)}[REGION]
TIMEBOX = ("2025-02-01","2025-03-31")

PATHS = {
    "balloons": "data/balloons/windborne_profiles.csv",
    "era5_prof":"data/reanalysis/era5_profile.nc",
}

# ## Imports & Utils
import os, warnings
import numpy as np, pandas as pd, xarray as xr, matplotlib.pyplot as plt

warnings.filterwarnings("ignore"); plt.rcParams["figure.figsize"]=(8,4)

def ensure_cols(df, cols):
    miss=[c for c in cols if c not in df.columns]
    if miss: raise KeyError(f"Missing columns: {miss}")

def wind_speed(u,v): return np.sqrt(u**2 + v**2)

def layer_mean(df, pmin, pmax, cols=('wind_u','wind_v')):
    sel = df[(df['pressure_hPa']<=pmax)&(df['pressure_hPa']>=pmin)]
    return sel[list(cols)].mean()

def safe_open_dataset(p):
    if not os.path.exists(p): print(f"[WARN] Missing: {p}"); return None
    try: return xr.open_dataset(p)
    except Exception as e: print(f"[WARN] Open fail {p}: {e}"); return None

# ## Load & QC
b = pd.read_csv(PATHS['balloons'], parse_dates=['time'])
ensure_cols(b, ['time','lat','lon','pressure_hPa','alt_m','temp_C','wind_u','wind_v'])
b = b[(b['time']>=TIMEBOX[0])&(b['time']<=TIMEBOX[1])]

lat0, lon0 = SITE
dist_km = np.hypot((b['lat']-lat0)*111, (b['lon']-lon0)*111*np.cos(np.radians(lat0)))
bp = b[dist_km<=100].copy()

# Temperature QC: 4*IQR per pressure bin
bp['qc_ok']=True
bins = [100,300,500,700,900,1100]
bp['plev_bin'] = pd.cut(bp['pressure_hPa'], bins=bins)
for pbin, g in bp.groupby('plev_bin'):
    if len(g)<10: continue
    q1, q3 = g['temp_C'].quantile([0.25,0.75])
    iqr = q3-q1
    ok = (g['temp_C']>=q1-4*iqr) & (g['temp_C']<=q3+4*iqr)
    bp.loc[g.index, 'qc_ok'] = ok
bp = bp[bp['qc_ok']].drop(columns=['plev_bin'])

# Compare with ERA5 profile at a representative time
era_prof = safe_open_dataset(PATHS['era5_prof'])
if era_prof is not None and len(bp):
    t0 = bp['time'].iloc[len(bp)//2]
    bln = bp[bp['time']==t0].sort_values('pressure_hPa')
    e_t = era_prof.sel(time=t0, method='nearest')
    e_p = e_t['level'].values if 'level' in e_t.coords else np.linspace(1000,100, len(e_t['temp']))
    plt.figure()
    plt.plot(bln['temp_C'], bln['pressure_hPa'], label='Balloon')
    plt.plot(e_t['temp'].values, e_p, label='ERA5')
    plt.gca().invert_yaxis(); plt.xlabel('Temp (°C)'); plt.ylabel('Pressure (hPa)')
    plt.title('Vertical Temperature Profile'); plt.legend(); plt.show()

    low = layer_mean(bln, 800, 925); mid = layer_mean(bln, 600, 700)
    shear = wind_speed(mid['wind_u']-low['wind_u'], mid['wind_v']-low['wind_v'])
    print(f"[FA6] Layer shear (700–925 hPa): {shear:.1f} m/s")
else:
    print("[FA6] ERA5 profile dataset missing or no balloon profiles within range.")
