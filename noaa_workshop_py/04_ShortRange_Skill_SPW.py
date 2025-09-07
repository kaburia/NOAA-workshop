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

# # Focus Area 4: AI-Driven Now & Short-Term Forecasting (0–10 days)
#
# **Objective**: Compute HSS & Frequency Bias vs ground across thresholds; evaluate probabilistic ROC/AUC; run Suitable Planting Window (SPW) logic.
#
# Expected datasets: GraphCast (daily precip), GenCast (ensemble daily precip), CBAM forecast (daily precip), ground CSV.

# ## CONFIG
REGION = "EA"
CFG = {
    "EA": dict(point=(-1.29,36.82), event=("2025-03-01","2025-03-31")),
    "HoA": dict(point=(8.98,38.80),  event=("2025-08-01","2025-08-31")),
    "SAF": dict(point=(-15.8,35.0),  event=("2023-02-15","2023-03-31")),
}[REGION]

THRESH_MM = [1,10,25,50]
SPW_THRESHOLDS = [20,40,60]
SPW_WINDOW = 20

PATHS = {
    "ground_rain":  f"data/ground/tahmo_daily_rain_{REGION}.csv",
    "graphcast":    "data/forecasts/graphcast_daily.nc",
    "gencast_ens":  "data/forecasts/gencast_daily_ens.nc",
    "cbam_forecast":"data/forecasts/cbam_daily.nc",
}

# ## Imports & Utils
import os, warnings
import numpy as np, pandas as pd, xarray as xr, matplotlib.pyplot as plt

warnings.filterwarnings("ignore"); plt.rcParams["figure.figsize"]=(8,4)

def ensure_cols(df, cols):
    miss=[c for c in cols if c not in df.columns]
    if miss: raise KeyError(f"Missing columns: {miss}")

def contingency(obs, fc, thr):
    o = np.asarray(obs) >= thr; f = np.asarray(fc) >= thr
    H = int(np.sum(o & f)); M = int(np.sum(o & ~f)); FA = int(np.sum(~o & f)); CN = int(np.sum(~o & ~f))
    return H,M,FA,CN

def HSS(H,M,FA,CN):
    num = 2*(H*CN - FA*M); den = (H+M)*(M+CN) + (H+FA)*(FA+CN)
    return num/den if den>0 else np.nan

def FB(H,M,FA,CN):
    return (H+FA)/(H+M) if (H+M)>0 else np.nan

def roc_auc_from_probs(prob, obs, thr):
    prob = np.asarray(prob); obs = (np.asarray(obs)>=thr).astype(int)
    grid = np.linspace(0,1,101); TPR=[]; FPR=[]
    for cut in grid:
        f = (prob>=cut).astype(int)
        TP = np.sum((f==1)&(obs==1)); FP = np.sum((f==1)&(obs==0))
        FN = np.sum((f==0)&(obs==1)); TN = np.sum((f==0)&(obs==0))
        tpr = TP/(TP+FN) if (TP+FN)>0 else 0
        fpr = FP/(FP+TN) if (FP+TN)>0 else 0
        TPR.append(tpr); FPR.append(fpr)
    order = np.argsort(FPR)
    return float(np.trapz(np.array(TPR)[order], np.array(FPR)[order]))

def earliest_spw(start_date, series, thresh, window=20):
    s = series.loc[start_date:start_date+pd.Timedelta(days=window-1)].fillna(0)
    cum = s.cumsum(); hit = cum[cum>=thresh]
    return hit.index.min() if len(hit) else pd.NaT

def safe_open_dataset(p):
    if not os.path.exists(p): print(f"[WARN] Missing: {p}"); return None
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
obs = pd.read_csv(PATHS['ground_rain'], parse_dates=['date'])
ensure_cols(obs, ['date','station_id','lat','lon','precip'])
pt_lat, pt_lon = CFG['point']
obs_pt, sid = nearest_station_series(obs, pt_lat, pt_lon, 'precip')

graph = safe_open_dataset(PATHS['graphcast'])
genc  = safe_open_dataset(PATHS['gencast_ens'])
cbaf  = safe_open_dataset(PATHS['cbam_forecast'])

if (graph is not None) and (genc is not None) and (cbaf is not None):
    fc_graph = extract_point(graph, 'precip', pt_lat, pt_lon)
    fc_cbam  = extract_point(cbaf,  'precip', pt_lat, pt_lon)
    ens = genc['precip'].sel(lat=pt_lat, lon=pt_lon, method='nearest').to_pandas()
    prob25 = ens.groupby(level='time').apply(lambda x: (x>=25).mean())
    sl = slice(CFG['event'][0], CFG['event'][1])
    align = pd.concat({'obs':obs_pt.loc[sl],'graph':fc_graph.loc[sl],'cbam':fc_cbam.loc[sl],'p25':prob25.loc[sl]}, axis=1).dropna()

    # Skill metrics
    for thr in THRESH_MM:
        H,M,FA,CN = contingency(align['obs'], align['graph'], thr)
        print(f"[FA4] GraphCast thr{thr}mm: HSS={HSS(H,M,FA,CN):.2f} FB={FB(H,M,FA,CN):.2f}")
        H,M,FA,CN = contingency(align['obs'], align['cbam'], thr)
        print(f"[FA4] CBAM      thr{thr}mm: HSS={HSS(H,M,FA,CN):.2f} FB={FB(H,M,FA,CN):.2f}")

    auc = roc_auc_from_probs(align['p25'], align['obs'], 25)
    print(f"[FA4] GenCast P(≥25mm) ROC AUC: {auc:.2f}")

    start = align.index.min()
    for T in SPW_THRESHOLDS:
        d_graph = earliest_spw(start, fc_graph, T, SPW_WINDOW)
        d_cbam  = earliest_spw(start, fc_cbam,  T, SPW_WINDOW)
        print(f"[FA4] SPW ≥{T}mm/{SPW_WINDOW}d: GraphCast {d_graph}, CBAM {d_cbam}")
else:
    print("[FA4] One or more forecast datasets missing.")
