import streamlit as st
import pandas as pd
import numpy as np
import os

# --- 1. Load data ---

# Use absolute path - hardcoded for reliability
path = r"C:\Users\Admin\OneDrive\Documents\ken-rainfall-subnat-full.csv"

try:
    df_all = pd.read_csv(path, parse_dates=["date"])
except FileNotFoundError:
    st.error(f"❌ File not found at: {path}")
    st.info("Please verify the file path exists.")
    st.stop()
except Exception as e:
    st.error(f"❌ Error loading file: {str(e)}")
    st.stop()

# --- 2. Core functions (no external modules) ---

def add_rain_anomalies_all(df, col="rfh", window=14, z_thresh=3):
    """Add rain_mm, rolling mean/std, z_score and is_anomaly for all PCODEs."""
    df = df.sort_values(["PCODE", "date"]).copy()
    df["rain_mm"] = df[col]

    df["rain_mean"] = (
        df.groupby("PCODE")["rain_mm"]
          .rolling(window=window, min_periods=window//2)
          .mean()
          .reset_index(level=0, drop=True)
    )
    df["rain_std"] = (
        df.groupby("PCODE")["rain_mm"]
          .rolling(window=window, min_periods=window//2)
          .std()
          .reset_index(level=0, drop=True)
    )

    eps = 1e-3
    df["z_score"] = (df["rain_mm"] - df["rain_mean"]) / (df["rain_std"] + eps)
    df["is_anomaly"] = df["z_score"].abs() > z_thresh
    return df


def label_risk_v3(row):
    """Rule-based risk: 0=low, 1=medium, 2=high."""
    r1 = row["rain_mm"]
    r3 = row["rain_3d"]
    r7 = row["rain_7d"]
    is_anom = bool(row["is_anomaly"])

    # High risk
    if (is_anom and r1 >= 50) or (r3 >= 130) or (r7 >= 200):
        return 2

    # Medium risk
    if is_anom or (r1 >= 30) or (r3 >= 80) or (r7 >= 120):
        return 1

    # Low risk
    return 0


def build_features(df_all):
    """Return df_feat with anomalies, rolling sums, and risk_level_v3."""
    df_feat = add_rain_anomalies_all(df_all)
    df_feat = df_feat.sort_values(["PCODE", "date"]).copy()

    df_feat["rain_3d"] = (
        df_feat.groupby("PCODE")["rain_mm"]
               .rolling(window=3, min_periods=1).sum()
               .reset_index(level=0, drop=True)
    )

    df_feat["rain_7d"] = (
        df_feat.groupby("PCODE")["rain_mm"]
               .rolling(window=7, min_periods=1).sum()
               .reset_index(level=0, drop=True)
    )

    df_feat["risk_level_v3"] = df_feat.apply(label_risk_v3, axis=1)
    return df_feat


def get_recent_risk_for_pcode(df_feat, pcode, n_last=60):
    sub = df_feat[df_feat["PCODE"] == pcode].copy()
    sub = sub.sort_values("date").dropna(subset=["rain_mm", "rain_3d", "rain_7d"])
    return sub.tail(n_last).copy()


# --- 3. Build features once ---

df_feat = build_features(df_all)

# --- 4. Streamlit UI ---

st.title("Kenya rainfall risk (rule-based)")

pcodes = sorted(df_feat["PCODE"].unique())
pcode = st.selectbox("Select PCODE:", pcodes)
n_last = st.slider("Days to show:", 30, 365, 120)

sub = get_recent_risk_for_pcode(df_feat, pcode, n_last=n_last)

if sub.empty:
    st.warning("No data available for this PCODE / period after filtering.")
else:
    st.subheader("Rainfall (mm)")
    st.line_chart(sub.set_index("date")["rain_mm"])

    st.subheader("Recent risk (last 60 rows shown)")
    st.dataframe(
        sub[["date", "rain_mm", "rain_3d", "rain_7d", "risk_level_v3"]].tail(60)
    )
