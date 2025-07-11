import pandas as pd
import numpy as np
from pathlib import Path


def normalize_dataframe(df):
    numeric_cols = df.select_dtypes(include=np.number).columns
    for col in numeric_cols:
        col_range = df[col].max() - df[col].min()
        if col_range != 0:
            df[col] = (df[col] - df[col].min()) / col_range
        else:
            df[col] = 0.0
    return df


def create_shading_features(differences_dict, out_dir, verbose_output=False, store_results=True):
    feature_vectors = {}

    if store_results:
        shading_dir = Path(out_dir) / "shading"
        shading_dir.mkdir(parents=True, exist_ok=True)

    for plant_id, df in differences_dict.items():
        if verbose_output:
            print(f"Processing shading features for plant: {plant_id}")
        
        required_cols = ['Timestamp', 'PV(W)', 'MPP1(A)', 'MPP2(A)', 'MPP1(V)', 'MPP2(V)']
        if not all(col in df.columns for col in required_cols) or df.empty:
            continue
        
        df = df.copy()
        df["date"] = pd.to_datetime(df["Timestamp"]).dt.date
        daily_groups = df.groupby("date")

        features = []
        for day, group in daily_groups:
            if group.empty:
                continue
                
            pv = group["PV(W)"]
            mpp1_a = group["MPP1(A)"]
            mpp2_a = group["MPP2(A)"]
            mpp1_v = group["MPP1(V)"]
            mpp2_v = group["MPP2(V)"]

            energy_yield_kWh = pv.sum() / 1000.0
            peak_pv_power_W = pv.max()
            
            delta = pv.diff().fillna(0)
            delta_std = delta.std()
            
            max_delta_idx = delta.abs().argmax()
            max_delta_hour = group.iloc[max_delta_idx]["Timestamp"].hour if len(group) > max_delta_idx else 12
            
            morning_mask = group["Timestamp"].dt.hour < 12
            morning = pv[morning_mask].sum()
            afternoon = pv[~morning_mask].sum()
            morning_afternoon_ratio = morning / afternoon if afternoon > 0 else np.nan

            mmp1_a_std = mpp1_a.std()
            mpp2_a_std = mpp2_a.std()
            mpp1_v_std = mpp1_v.std()
            mpp2_v_std = mpp2_v.std()

            features.append({
                "date": day,
                "energy_yield_kWh": energy_yield_kWh,
                "peak_pv_power_W": peak_pv_power_W,
                "delta_std": delta_std,
                "max_delta_hour": max_delta_hour,
                "morning_afternoon_ratio": morning_afternoon_ratio,
                "mpp1_a_std": mmp1_a_std,
                "mpp2_a_std": mpp2_a_std,
                "mpp1_v_std": mpp1_v_std,
                "mpp2_v_std": mpp2_v_std
            })

        if not features:
            continue

        feature_df = pd.DataFrame(features)
        feature_df.set_index("date", inplace=True)
        feature_df = normalize_dataframe(feature_df)
        feature_vectors[plant_id] = feature_df

        if store_results:
            out_file = Path(out_dir) / "shading" / f"feature_vectors_{plant_id}.csv"
            feature_df.to_csv(out_file)
            if verbose_output:
                print(f"  Shading features saved to: {out_file}")

    return feature_vectors


def create_pollution_features(differences_dict, out_dir, verbose_output=False, store_results=True):
    feature_vectors = {}

    if store_results:
        pollution_dir = Path(out_dir) / "pollution_rawlike"
        pollution_dir.mkdir(parents=True, exist_ok=True)

    for plant_id, df in differences_dict.items():
        if verbose_output:
            print(f"Processing pollution features for plant: {plant_id}")
        
        required_cols = ['Timestamp', 'PV(W)', 'Battery(W)', 'SOC(%)', 'Load(W)', 'Difference']
        if not all(col in df.columns for col in required_cols) or df.empty:
            continue
        
        df = df.copy()
        df["date"] = pd.to_datetime(df["Timestamp"]).dt.date

        daily_df = df.groupby("date").agg({
            "PV(W)": ["sum", "mean", "std", "max"],
            "Battery(W)": ["mean"],
            "SOC(%)": ["mean"],
            "Load(W)": ["mean"],
            "Difference": ["mean", "std", "min"]
        })

        if daily_df.empty:
            continue

        daily_df.columns = ['_'.join(col).strip() for col in daily_df.columns.values]
        daily_df.index = pd.to_datetime(daily_df.index)
        daily_df.index.name = "date"

        daily_df["diff_mean_30d_avg"] = daily_df["Difference_mean"].rolling(
            window=30, min_periods=15
        ).mean()

        daily_df_clean = daily_df.dropna()
        
        if daily_df_clean.empty:
            continue

        daily_df_clean = normalize_dataframe(daily_df_clean)
        feature_vectors[plant_id] = daily_df_clean

        if store_results:
            out_file = Path(out_dir) / "pollution_rawlike" / f"feature_vectors_{plant_id}.csv"
            daily_df_clean.to_csv(out_file)
            if verbose_output:
                print(f"  Pollution features saved to: {out_file}")

    return feature_vectors
