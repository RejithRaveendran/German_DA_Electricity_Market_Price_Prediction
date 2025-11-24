
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Day-Ahead Electricity Price Prediction for German Market (DE-LU)
Generates two fixed CSV files in the GitHub project folder:
1. artifacts/forecast.csv (tomorrow's forecast)
2. artifacts/accuracy.csv (previous day's actual vs predicted)
"""

import subprocess
import sys
import os
import warnings
warnings.filterwarnings("ignore")

# -------------------------- Auto-install required packages --------------------------
required_packages = ["pandas", "numpy", "requests", "pytz", "python-dateutil", "scikit-learn", "lightgbm", "joblib", "entsoe-py"]
for pkg in required_packages:
    try:
        __import__(pkg.split('-')[0])
    except ImportError:
        print(f"Installing missing package: {pkg}")
        subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])

# -------------------------- Imports --------------------------
import pandas as pd
import numpy as np
from datetime import date
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import lightgbm as lgb
import joblib
from entsoe import EntsoePandasClient

# -------------------------- Config --------------------------
ENTSOE_TOKEN = "c75a0111-6841-4300-a9d8-e3746d32ac5e"  # Hardcoded token
ZONE_CODE_ENTSOE = "DE_LU"
TZ_LOCAL = "Europe/Berlin"
TZ_ENTSOE = "Europe/Brussels"

FEATURE_STORE_DIR = "feature_store"; os.makedirs(FEATURE_STORE_DIR, exist_ok=True)
MODEL_DIR = "models"; os.makedirs(MODEL_DIR, exist_ok=True)
ARTIFACTS_DIR = "artifacts"; os.makedirs(ARTIFACTS_DIR, exist_ok=True)

TRAIN_START = pd.Timestamp("2024-10-01", tz=TZ_LOCAL)
TRAIN_END = pd.Timestamp.today(tz=TZ_LOCAL) - pd.Timedelta(days=5)
TOMORROW_LOCAL = (pd.Timestamp.today(tz=TZ_LOCAL) + pd.Timedelta(days=1)).normalize()
TODAY_LOCAL = pd.Timestamp.today(tz=TZ_LOCAL).normalize()
PREV_LOCAL_DATE = TODAY_LOCAL - pd.Timedelta(days=1)

print("✅ Config loaded.")

# -------------------------- Helpers --------------------------
def make_qh_index(start_ts, end_ts, tzname=TZ_LOCAL):
    return pd.date_range(start=start_ts, end=end_ts, freq="15min", tz=tzname)

def to_brussels(ts):
    ts = pd.Timestamp(ts)
    return ts.tz_localize(TZ_ENTSOE) if ts.tz is None else ts.tz_convert(TZ_ENTSOE)

# -------------------------- ENTSO-E client --------------------------
pc = EntsoePandasClient(api_key=ENTSOE_TOKEN)

# -------------------------- Data fetch --------------------------
def get_da_series(start_local, end_local):
    idx_qh = make_qh_index(start_local, end_local)
    try:
        s_br, e_br = to_brussels(start_local), to_brussels(end_local)
        da = pc.query_day_ahead_prices(country_code=ZONE_CODE_ENTSOE, start=s_br, end=e_br)
        da = da.rename("da_eur_per_mwh").to_frame()
    except Exception as e:
        print("ENTSO-E DA failed:", repr(e))
        da = pd.DataFrame(columns=["da_eur_per_mwh"])
    if da.empty or not isinstance(da.index, pd.DatetimeIndex):
        return pd.DataFrame(index=idx_qh, columns=["da_eur_per_mwh"])
    return da.resample("15min").mean().ffill().bfill().reindex(idx_qh)

def get_load_forecast(start_local, end_local):
    try:
        s_br, e_br = to_brussels(start_local), to_brussels(end_local)
        ser = pc.query_load_forecast(country_code=ZONE_CODE_ENTSOE, start=s_br, end=e_br)
        return ser.rename("load_fcst_mw")
    except:
        return pd.Series(dtype=float, name="load_fcst_mw")

def get_wind_solar_forecast(start_local, end_local):
    try:
        s_br, e_br = to_brussels(start_local), to_brussels(end_local)
        df = pc.query_wind_and_solar_forecast(country_code=ZONE_CODE_ENTSOE, start=s_br, end=e_br)
        if isinstance(df, pd.DataFrame) and not df.empty:
            ser = df.select_dtypes(include=[np.number]).sum(axis=1)
            ser.name = "res_fcst_mw"
            return ser
    except:
        pass
    return pd.Series(dtype=float, name="res_fcst_mw")

# -------------------------- Feature store --------------------------
def build_feature_store(start_local=TRAIN_START, end_local=TRAIN_END):
    idx_qh = make_qh_index(start_local, end_local)
    da = get_da_series(start_local, end_local)
    load_fcst = get_load_forecast(start_local, end_local)
    res_fcst = get_wind_solar_forecast(start_local, end_local)

    def to_qh(df_or_ser):
        if df_or_ser is None or len(df_or_ser)==0:
            return pd.DataFrame(index=idx_qh)
        df = df_or_ser if isinstance(df_or_ser, pd.DataFrame) else df_or_ser.to_frame()
        return df.resample("15min").mean().ffill().bfill().reindex(idx_qh)

    X = to_qh(da)
    X = X.join(to_qh(load_fcst))
    X = X.join(to_qh(res_fcst))

    X["hour"] = X.index.hour
    X["quarter"] = (X.index.minute // 15)
    X["dow"] = X.index.dayofweek
    X["month"] = X.index.month
    X["is_weekend"] = X["dow"].isin([5,6]).astype(int)

    for lag_qh in [1, 4, 96, 96*7]:
        X[f"da_lag_{lag_qh}"] = X["da_eur_per_mwh"].shift(lag_qh)

    expected_cols = ["load_fcst_mw","res_fcst_mw","hour","quarter","dow","month","is_weekend","da_lag_1","da_lag_4","da_lag_96","da_lag_672"]
    for c in expected_cols:
        if c not in X.columns:
            X[c] = 0.0

    path = f"{FEATURE_STORE_DIR}/da_features_qh.parquet"
    X.to_parquet(path)
    print(f"✅ Feature store saved: {path} (shape={X.shape})")
    return X

# -------------------------- Train or load model --------------------------
if os.path.exists(f"{MODEL_DIR}/da_models_qh.joblib"):
    print("✅ Found existing model. Loading...")
    models = joblib.load(f"{MODEL_DIR}/da_models_qh.joblib")
else:
    print("⚠ No model found. Training new model...")
    features = build_feature_store()
    X_da = features.drop(columns=["da_eur_per_mwh"])
    y_da = features["da_eur_per_mwh"].dropna()
    X_da = X_da.loc[y_da.index]

    cut = int(0.85 * len(X_da))
    X_tr, X_va = X_da.iloc[:cut].values, X_da.iloc[cut:].values
    y_tr, y_va = y_da.iloc[:cut], y_da.iloc[cut:]

    def train_q_model(Xtr, ytr, alpha):
        params = dict(objective="quantile", alpha=alpha, n_estimators=500, learning_rate=0.05, num_leaves=63)
        model = lgb.LGBMRegressor(**params)
        pipe = Pipeline([("scaler", StandardScaler(with_mean=False)), ("lgb", model)])
        pipe.fit(Xtr, ytr)
        return pipe

    mdl_p50 = train_q_model(X_tr, y_tr, 0.5)
    mdl_p10 = train_q_model(X_tr, y_tr, 0.1)
    mdl_p90 = train_q_model(X_tr, y_tr, 0.9)

    joblib.dump({"p50": mdl_p50, "p10": mdl_p10, "p90": mdl_p90, "cols": X_da.columns.tolist()}, f"{MODEL_DIR}/da_models_qh.joblib")
    models = joblib.load(f"{MODEL_DIR}/da_models_qh.joblib")
    print("✅ Model trained and saved.")

# -------------------------- Forecast --------------------------
def build_features_for_day(delivery_local_date):
    start_t = delivery_local_date
    end_t = delivery_local_date + pd.Timedelta(days=1)
    idx_qh = make_qh_index(start_t, end_t)

    load_fcst_t = get_load_forecast(start_t, end_t)
    res_fcst_t = get_wind_solar_forecast(start_t, end_t)

    y_start = start_t - pd.Timedelta(days=1)
    y_end = end_t - pd.Timedelta(days=1)
    da_yday = get_da_series(y_start, y_end)
    yday_shifted = da_yday["da_eur_per_mwh"].copy()
    yday_shifted.index = yday_shifted.index + pd.Timedelta(days=1)

    X = pd.DataFrame(index=idx_qh)
    X = X.join(load_fcst_t.to_frame(name="load_fcst_mw"), how="left")
    X = X.join(res_fcst_t.to_frame(name="res_fcst_mw"), how="left")

    X["hour"] = X.index.hour
    X["quarter"] = (X.index.minute // 15)
    X["dow"] = X.index.dayofweek
    X["month"] = X.index.month
    X["is_weekend"] = X["dow"].isin([5,6]).astype(int)

    lag_base = yday_shifted.reindex(idx_qh)
    X["da_lag_96"] = lag_base
    X["da_lag_1"] = lag_base.shift(1)
    X["da_lag_4"] = lag_base.shift(4)
    X["da_lag_672"] = lag_base.shift(96*7)

    for c in models["cols"]:
        if c not in X.columns:
            X[c] = 0.0
    X = X[models["cols"]].ffill().bfill().fillna(0.0)
    return X

def forecast_for_day(delivery_local_date):
    X_day = build_features_for_day(delivery_local_date)
    p50 = models["p50"].predict(X_day.values)
    p10 = models["p10"].predict(X_day.values)
    p90 = models["p90"].predict(X_day.values)

    out = pd.DataFrame({
        "timestamp_local": X_day.index,
        "price_p10_eur_per_mwh": np.round(p10, 2),
        "price_p50_eur_per_mwh": np.round(p50, 2),
        "price_p90_eur_per_mwh": np.round(p90, 2)
    }).set_index("timestamp_local")
    return out

def prev_day_actual_vs_pred(prev_local_date):
    actual_df = get_da_series(prev_local_date, prev_local_date + pd.Timedelta(days=1))
    actual_df = actual_df.rename(columns={"da_eur_per_mwh":"actual_da_eur_per_mwh"})
    pred_df = forecast_for_day(prev_local_date).rename(columns={
        "price_p50_eur_per_mwh":"pred_p50_eur_per_mwh",
        "price_p10_eur_per_mwh":"pred_p10_eur_per_mwh",
        "price_p90_eur_per_mwh":"pred_p90_eur_per_mwh"
    })
    comp = pred_df.join(actual_df, how="left")
    comp["abs_error"] = np.abs(comp["actual_da_eur_per_mwh"] - comp["pred_p50_eur_per_mwh"])
    comp["sq_error"] = (comp["actual_da_eur_per_mwh"] - comp["pred_p50_eur_per_mwh"])**2
    comp["in_band"] = ((comp["actual_da_eur_per_mwh"] >= comp["pred_p10_eur_per_mwh"]) & (comp["actual_da_eur_per_mwh"] <= comp["pred_p90_eur_per_mwh"])).astype(int)
    mae = float(comp["abs_error"].mean())
    rmse = float(np.sqrt(comp["sq_error"].mean()))
    coverage = float(100.0 * comp["in_band"].mean())
    print(f"✅ Prev-day metrics → MAE: {mae:.3f} EUR/MWh | RMSE: {rmse:.3f} EUR/MWh | Coverage: {coverage:.2f}%")
    return comp

# -------------------------- Save outputs to GitHub project folder --------------------------
forecast_tomorrow_df = forecast_for_day(TOMORROW_LOCAL)
csv_forecast_path = os.path.join(ARTIFACTS_DIR, "forecast.csv")
forecast_tomorrow_df.to_csv(csv_forecast_path)
print(f"✅ Saved forecast CSV to {csv_forecast_path}")

prev_comp_df = prev_day_actual_vs_pred(PREV_LOCAL_DATE)
csv_accuracy_path = os.path.join(ARTIFACTS_DIR, "accuracy.csv")
prev_comp_df.to_csv(csv_accuracy_path)
print(f"✅ Saved accuracy CSV to {csv_accuracy_path}")

# --- ADD THIS BLOCK NEAR THE END, right after saving the CSVs ---

import hashlib
import json

def file_sha256(path: str) -> str:
    """Return SHA-256 hash (hex) of the file bytes."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()

def write_hash_meta(csv_path: str, update_on_change_only: bool = True) -> None:
    """
    Create/update a metadata JSON next to the CSV that records its SHA-256.
    If update_on_change_only=True, we only rewrite the meta file when the hash changed,
    avoiding commits for identical content.
    """
    sha_now = file_sha256(csv_path)
    meta_path = csv_path + ".meta.json"

    prev_sha = None
    if os.path.exists(meta_path):
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                prev = json.load(f)
                prev_sha = prev.get("sha256")
        except Exception:
            # If previous meta is unreadable, proceed to write a fresh one.
            pass

    if update_on_change_only and prev_sha == sha_now:
        print(f"⛔ No CSV content change detected for {csv_path} (sha256={sha_now}). Meta not updated.")
        return

    meta = {
        "file": os.path.basename(csv_path),
        "sha256": sha_now,
        "generated_at": pd.Timestamp.now(tz=TZ_LOCAL).isoformat(),
        "previous_sha256": prev_sha,
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"✅ Wrote/updated hash metadata: {meta_path} (sha256={sha_now})")

# After saving the two CSVs:
write_hash_meta(csv_forecast_path)  # artifacts/forecast.csv.meta.json
write_hash_meta(csv_accuracy_path)  # artifacts/accuracy.csv.meta.json

