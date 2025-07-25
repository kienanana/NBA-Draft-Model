import os
import pandas as pd
from sklearn.cluster import KMeans
import joblib
from features import college_features, noncollege_features

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
YEARS = [2020, 2021, 2022, 2023, 2024]

def collect_feature_data(years, target_classes, features):
    all_rows = []
    for year in years:
        path = os.path.join(PROJECT_ROOT, "data", "processed", str(year), f"draftpool_stats_{year}.csv")
        if not os.path.exists(path):
            print(f"⚠️ File not found: {path}")
            continue
        df = pd.read_csv(path)

        # Log available classifications
        print(f"📊 {year} classifications: {df['classification'].unique()}")

        # Filter by one or more target classifications
        df = df[df["classification"].isin(target_classes)]
        df = df[features].dropna()

        if not df.empty:
            all_rows.append(df)
        else:
            print(f"⚠️ No matching rows for {target_classes} in year {year}")
    return pd.concat(all_rows, ignore_index=True)

def main():
    os.makedirs("../models", exist_ok=True)

    # --- Fit college clusters ---
    print("🎓 Fitting College Clusters...")
    college_data = collect_feature_data(YEARS, ["College"], college_features)
    college_kmeans = KMeans(n_clusters=8, random_state=42).fit(college_data)
    college_model_path = os.path.join(PROJECT_ROOT, "models", "kmeans_college.pkl")
    joblib.dump(college_kmeans, college_model_path)
    print(f"✅ Saved: {college_model_path}")

    # --- Fit non-college clusters ---
    print("🌍 Fitting Non-College Clusters...")
    noncollege_data = collect_feature_data(YEARS, ["G League", "Overtime Elite", "International"], noncollege_features)
    noncollege_kmeans = KMeans(n_clusters=8, random_state=42).fit(noncollege_data)
    noncollege_model_path = os.path.join(PROJECT_ROOT, "models", "kmeans_noncollege.pkl")
    joblib.dump(noncollege_kmeans, noncollege_model_path)
    print(f"✅ Saved: {noncollege_model_path}")

if __name__ == "__main__":
    main()
