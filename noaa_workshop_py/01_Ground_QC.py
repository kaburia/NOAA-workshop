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

# # Focus Area 1: Ground Observations Monitoring & QC
#
# **Objective**: Quality-control daily station rainfall and validate against CHIRPS pentads.
# **Outputs**: Station scorecard (confidence class), scatter diagnostics, simple map of station confidence.
# **Region Case**: Auto-configured; edit REGION/CFG/paths below.
#
# Expected columns:
#   Ground rain CSV: date,station_id,lat,lon,precip
#   CHIRPS pentad NetCDF: dims time,lat,lon with var 'precip' (mm/5d)

# ## CONFIG
REGION = "EA"  # "EA" (East Africa), "HoA" (Horn), "SAF" (Southern Africa)
REGION_DEFAULTS = {
    "EA": dict(country="Kenya",   flood_window=("2025-04-01", "2025-05-31")),
    "HoA": dict(country="Ethiopia", flood_window=("2025-01-01", "2025-03-31")),
    "SAF": dict(country="Malawi", flood_window=("2025-06-01", "2025-06-30")),
}
CFG = REGION_DEFAULTS[REGION]

PATHS = {
    "ground_rain":   f"data/ground/tahmo_daily_rain_{REGION}.csv",
    "chirps_pentad": "data/satellite/chirps_pentad.nc",
}

# QC parameters
PENTAD_FREQ = "5D"
RAIN_MAX_D = 500.0   # mm/day
FLATLINE_DAYS = 5
EVENT_WINDOW = CFG['flood_window']

# ## Imports
import os, warnings
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, ttest_rel

try:
    import ipywidgets as widgets
    from IPython.display import display
except Exception:
    widgets = None

warnings.filterwarnings("ignore")
plt.rcParams["figure.figsize"] = (8,4)

# ## Utils
def ensure_cols(df: pd.DataFrame, cols):
    miss = [c for c in cols if c not in df.columns]
    if miss: raise KeyError(f"Missing columns: {miss}")
    return True

def rmse(a,b):
    a = np.asarray(a); b = np.asarray(b)
    return float(np.sqrt(np.nanmean((a-b)**2)))

def safe_open_dataset(path):
    if not os.path.exists(path):
        print(f"[WARN] Missing: {path}"); return None
    try:
        return xr.open_dataset(path)
    except Exception as e:
        print(f"[WARN] Could not open {path}: {e}"); return None

# ## Core functions
def qc_daily_rain(df: pd.DataFrame, v='precip'):
    df = df.copy()
    ensure_cols(df, ['date','station_id',v])
    df['qc_range_ok'] = (df[v] >= 0) & (df[v] <= RAIN_MAX_D)
    df['qc_missing']  = df[v].isna()
    df['flatline']    = False
    for sid, g in df.groupby('station_id', sort=False):
        run = (g[v].diff()==0) & g[v].notna()
        runlen = run.groupby((~run).cumsum()).transform('size')
        flat = run & (runlen >= FLATLINE_DAYS)
        df.loc[flat.index, 'flatline'] = flat.values
    df['qc_ok'] = df['qc_range_ok'] & (~df['qc_missing']) & (~df['flatline'])
    return df

def to_pentad_with_coords(df: pd.DataFrame, date='date', v='precip'):
    d = df.copy()
    d[date] = pd.to_datetime(d[date])
    xy = d[['station_id','lat','lon']].drop_duplicates('station_id')
    pent = (d.set_index(date)
              .groupby('station_id')[v].resample(PENTAD_FREQ).sum()
              .rename('station_pentad')
              .reset_index())
    pent = pent.merge(xy, on='station_id', how='left')
    return pent

def match_satellite_pentad(pent_df: pd.DataFrame, sat_path: str):
    ds = safe_open_dataset(sat_path)
    if ds is None: return pent_df
    out = []
    for (sid, lat, lon), g in pent_df.groupby(['station_id','lat','lon']):
        sat_point = ds['precip'].sel(lat=lat, lon=lon, method='nearest').to_pandas()
        gg = g.set_index('date').join(sat_point.rename('sat_pentad'), how='inner').reset_index()
        gg['station_id']=sid; gg['lat']=lat; gg['lon']=lon
        out.append(gg)
    return pd.concat(out, ignore_index=True) if out else pent_df

def station_scores(df: pd.DataFrame):
    rows=[]
    for sid, g in df.groupby('station_id'):
        valid = g[['station_pentad','sat_pentad']].dropna()
        if len(valid) < 6:
            rows.append((sid, np.nan, np.nan, np.nan, 'NA')); continue
        r, _ = pearsonr(valid['station_pentad'], valid['sat_pentad'])
        Rmse  = rmse(valid['station_pentad'], valid['sat_pentad'])
        try:
            _, p_val = ttest_rel(valid['station_pentad'], valid['sat_pentad'], nan_policy='omit')
            bias_sig = 'Significant' if p_val < 0.05 else 'Not Significant'
        except Exception:
            bias_sig = 'NA'
        completeness = 1 - g['station_pentad'].isna().mean()
        outlier_rate = (g['station_pentad']<0).mean()
        score = 100*(0.4*completeness + 0.4*np.nan_to_num(r, nan=0) + 0.2*(1-outlier_rate))
        rows.append((sid, score, r, Rmse, bias_sig))
    out = pd.DataFrame(rows, columns=['station_id','confidence_score','pearson_r','rmse','bias_signif'])
    out['class'] = pd.cut(out['confidence_score'], [0,60,80,100], labels=['Low','Medium','High'])
    return out

# ## Load & Run
try:
    g = pd.read_csv(PATHS['ground_rain'])
    ensure_cols(g, ['date','station_id','lat','lon','precip'])
    g['date']=pd.to_datetime(g['date'])
    g = qc_daily_rain(g)

    pent = to_pentad_with_coords(g[['date','station_id','lat','lon','precip']])
    m = match_satellite_pentad(pent, PATHS['chirps_pentad'])
    m_event = m[(m['date']>=EVENT_WINDOW[0])&(m['date']<=EVENT_WINDOW[1])]

    score = station_scores(m)
    score = score.merge(g[['station_id','lat','lon']].drop_duplicates(), on='station_id', how='left')
    print(score.sort_values('confidence_score', ascending=False).head(8))

    # Visuals
    plt.figure(); plt.scatter(m['station_pentad'], m['sat_pentad'], s=10, alpha=0.5)
    plt.xlabel('Station pentad (mm)'); plt.ylabel('CHIRPS pentad (mm)')
    plt.title(f"{CFG['country']}: Station vs CHIRPS (pentad)"); plt.grid(True); plt.show()

    for c in ['High','Medium','Low']:
        sel = score[score['class']==c]; plt.scatter(sel['lon'], sel['lat'], label=c, s=28)
    plt.legend(); plt.title(f"{CFG['country']} Station Confidence Classes")
    plt.xlabel('Lon'); plt.ylabel('Lat'); plt.show()

except FileNotFoundError:
    print("[FA1] Ground or satellite file missing; update PATHS.")

# ## Interactive QC threshold (optional)
def _plot_valid_hist(max_d=500.0):
    global RAIN_MAX_D; RAIN_MAX_D = max_d
    if 'g' in globals():
        g2 = qc_daily_rain(g)
        plt.figure(); plt.hist(g2['precip'][g2['qc_ok']], bins=25)
        plt.title(f'Valid precip histogram (max={max_d} mm)'); plt.show()

if widgets: display(widgets.interact(_plot_valid_hist, max_d=widgets.FloatSlider(min=100,max=1000,step=50,value=500)))
