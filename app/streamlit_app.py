import os
import io
import numpy as np
import pandas as pd
import streamlit as st
import altair as alt
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Local imports
import sys
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)

from ml.profiles import get_prospect_profile, build_profiles
from ml.simulation import simulate_draft, load_default_weights
from ml.simulation import load_clusters  
from ml.features import college_features, noncollege_features

st.set_page_config(page_title="Draft Lab", layout="wide")
st.title("🏀 NBA Draft Lab — Profiles & Simulation")

tab1, tab2, tab3 = st.tabs(["Prospect Profiles", "Clustering", "Draft Simulation"])

# -------------------- TAB 1: PROFILES --------------------
with tab1:
    st.subheader("Prospect Profiles")
    year = st.selectbox("Draft Class Year", [2020, 2021, 2022, 2023, 2024], index=4)

    # load list of names for the dropdown
    prof_df, _, _ = build_profiles(year)
    names = sorted(prof_df["Name"].unique().tolist())
    name = st.selectbox("Select Prospect", names)

    if name:
        row, fig_pg, fig_adv, cols_pg, cols_adv = get_prospect_profile(year, name)
        if row is None:
            st.warning("Prospect not found in this class.")
        else:
            left, right = st.columns([1,1])
            with left:
                st.pyplot(fig_pg)
            with right:
                st.pyplot(fig_adv)

            # Scores
            st.markdown("### Scores")
            colA, colB, colC = st.columns(3)
            with colA:
                st.metric("Overall (/10)", f"{row.get('UpdatedOverall', row.get('OverallScore')):.2f}")
            with colB:
                if row.get("UpsideScore") is not None and not pd.isna(row.get("UpsideScore")):
                    st.metric("Upside (/10)", f"{row['UpsideScore']:.2f}")
                else:
                    st.metric("Upside (/10)", "—")
            with colC:
                st.metric("Age", f"{row.get('Age', '—')}")

            # Show comps if the CSV exists
            comps_path = os.path.join(PROJECT_ROOT, "data", "processed", str(year), f"prospect_comps_{year}.csv")
            if os.path.exists(comps_path):
                comps_df = pd.read_csv(comps_path)
                c_row = comps_df[comps_df["Name"].str.lower() == name.lower()]
                if not c_row.empty:
                    c_row = c_row.iloc[0]
                    st.markdown("### NBA Comparisons")
                    st.write(f"**Archetype**: {int(c_row['Archetype']) if pd.notna(c_row['Archetype']) else '—'}")
                    st.write(f"- {c_row['Comp1']}  (dist={c_row['Comp1_dist']:.2f})")
                    st.write(f"- {c_row['Comp2']}  (dist={c_row['Comp2_dist']:.2f})")
                    st.write(f"- {c_row['Comp3']}  (dist={c_row['Comp3_dist']:.2f})")
                else:
                    st.info("No comps found for this player.")
            else:
                st.info("Comps file not found. Run the comps step in your notebook.")

            def row_to_csv_bytes(row_dict):
                df = pd.DataFrame([row_dict])
                b = io.BytesIO()
                df.to_csv(b, index=False)
                return b.getvalue()
            
            # Download profile row as CSV
            st.download_button(
                "Download profile row (CSV)",
                data=row_to_csv_bytes(row),
                file_name=f"{name.replace(' ','_')}_profile_{year}.csv",
                mime="text/csv"
            )
            
# -------------------- TAB 2: CLUSTERING (INTERACTIVE) --------------------
with tab2:
    st.subheader("Global Playstyle Clusters")
    st.caption(
        "Select a year (2020–2024). Draft Class is then projected into 2D using PCA for visualisation, "
        "colors by cluster, and shows hover tooltips. Clusters are computed via the global models."
    )

    vis_year = st.selectbox("Draft year (clusters)", [2020, 2021, 2022, 2023, 2024], index=4, key="cluster_year")
    subset = st.multiselect(
        "Which classifications to show?",
        options=["College", "Non-College"],
        default=["College", "Non-College"]
    )

    # Load draft pool for the year
    pool_path = os.path.join(PROJECT_ROOT, "data", "processed", str(vis_year), f"draftpool_stats_{vis_year}.csv")
    if not os.path.exists(pool_path):
        st.error(f"Missing: {pool_path}")
    else:
        df = pd.read_csv(pool_path).copy()

        # make sure cluster models are present (raises if not)
        try:
            _ = load_clusters()
        except FileNotFoundError as e:
            st.error(str(e))
            st.stop()

        # Helper: compute clusters using our existing function
        def _cluster_by_group(df_in, classification: str, feature_cols):
            from ml.playstyles import compute_playstyle_clusters  # local import to avoid circulars
            sub = df_in[df_in["classification"].eq(classification)].copy()
            if sub.empty:
                return sub
            try:
                sub = compute_playstyle_clusters(sub, feature_cols=feature_cols, classification=classification)
            except Exception as e:
                st.warning(f"Clustering failed for {classification}: {e}")
            return sub

        # build the clustered frame(s)
        frames = []
        if "College" in subset:
            frames.append(_cluster_by_group(df, "College", college_features))
        if "Non-College" in subset:
            frames.append(_cluster_by_group(df, "Non-College", noncollege_features))
        dfc = pd.concat([f for f in frames if f is not None], ignore_index=True) if frames else pd.DataFrame()

        if dfc.empty:
            st.info("No rows to show for the selected filters.")
            st.stop()

        # Derive a readable cluster label column (many pipelines output cluster_0..k probs)
        cluster_cols = [c for c in dfc.columns if c.startswith("cluster_") and dfc[c].dtype != "O"]
        if "cluster_label" in dfc.columns:
            dfc["Cluster"] = dfc["cluster_label"].astype(int)
        elif cluster_cols:
            # Argmax over cluster probability/indicator columns
            dfc["Cluster"] = dfc[cluster_cols].astype(float).values.argmax(axis=1).astype(int)
        elif "cluster" in dfc.columns:
            dfc["Cluster"] = dfc["cluster"].astype(int)
        else:
            # Fallback
            dfc["Cluster"] = -1

        # 2D projection (PCA) per cohort, then concatenate (scales may differ slightly between cohorts)
        def _pca_embed(frame: pd.DataFrame, feature_cols):
            X = frame[feature_cols].copy()
            X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)
            X = np.asarray(X, dtype=float)
            Xs = StandardScaler().fit_transform(X)
            comp = PCA(n_components=2, random_state=0).fit_transform(Xs)
            out = frame.copy()
            out["pc1"] = comp[:, 0]
            out["pc2"] = comp[:, 1]
            return out

        parts = []
        if "College" in subset and not dfc[dfc["classification"] == "College"].empty:
            parts.append(_pca_embed(dfc[dfc["classification"] == "College"], college_features))
        if "Non-College" in subset and not dfc[dfc["classification"] != "College"].empty:
            parts.append(_pca_embed(dfc[dfc["classification"] != "College"], noncollege_features))

        df_plot = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()

        if df_plot.empty:
            st.info("Nothing to visualize after PCA.")
            st.stop()

        # Interactive scatter with hover tooltips
        tooltips = [
            alt.Tooltip("Name:N", title="Player"),
            alt.Tooltip("classification:N", title="Cohort"),
            alt.Tooltip("Cluster:N"),
        ]
        # Add a few common stat columns to tooltip if present
        for col in ["Age", "PTS", "AST", "TRB", "TS%", "OBPM", "DBPM", "BPM"]:
            if col in df_plot.columns:
                tooltips.append(alt.Tooltip(f"{col}:Q"))

        chart = (
            alt.Chart(df_plot)
            .mark_circle(size=70, opacity=0.85)
            .encode(
                x=alt.X("pc1:Q", title="PC1"),
                y=alt.Y("pc2:Q", title="PC2"),
                color=alt.Color("Cluster:N", legend=alt.Legend(title="Cluster")),
                shape=alt.Shape("classification:N", legend=alt.Legend(title="Cohort")),
                tooltip=tooltips,
            )
            .interactive()
            .properties(height=520)
        )
        st.altair_chart(chart, use_container_width=True)

        # Cluster summary table
        st.markdown("#### Cluster summary")
        summary = (
            df_plot.groupby(["classification", "Cluster"])["Name"]
            .count()
            .reset_index()
            .rename(columns={"Name": "Players"})
            .sort_values(["classification", "Cluster"])
        )
        st.dataframe(summary, use_container_width=True)

        csv_bytes = df_plot[["Name", "classification", "Cluster", "pc1", "pc2"]].to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download clustered points (CSV)",
            data=csv_bytes,
            file_name=f"clusters_{vis_year}.csv",
            mime="text/csv",
        )
        
        # --- Players by cluster (filterable table) ---
        st.markdown("#### Players in each cluster")

        # Cluster multi-select (defaults to all present)
        cluster_options = sorted([int(x) for x in df_plot["Cluster"].dropna().unique()])
        chosen_clusters = st.multiselect(
            "Show clusters", cluster_options, default=cluster_options, key="clusters_to_list"
        )

        name_query = st.text_input("Filter by player name (optional)", "")

        # Build a tidy table with useful columns
        base_cols = ["Name", "classification", "Cluster"]
        stat_pref = ["Age", "PTS", "AST", "TRB", "TS%", "OBPM", "DBPM", "BPM",
                     "OffenseScore", "DefenseScore", "GeneralScore"]
        extra_cols = [c for c in stat_pref if c in df_plot.columns]
        cols_to_show = base_cols + extra_cols

        table = df_plot.loc[:, [c for c in cols_to_show if c in df_plot.columns]].copy()

        # Apply filters
        if chosen_clusters:
            table = table[table["Cluster"].isin(chosen_clusters)]
        if name_query:
            table = table[table["Name"].str.contains(name_query, case=False, na=False)]

        table = table.sort_values(["Cluster", "classification", "Name"]).reset_index(drop=True)

        st.dataframe(table, use_container_width=True, height=360)

        # Optional: split view per cluster in expanders (handy for long lists)
        with st.expander("View by cluster (expandable)", expanded=False):
            for c in chosen_clusters:
                sub = table[table["Cluster"] == c]
                st.markdown(f"**Cluster {c}** — {len(sub)} players")
                st.dataframe(sub.drop(columns=["Cluster"]), use_container_width=True)

        # Download filtered table
        st.download_button(
            "Download filtered table (CSV)",
            data=table.to_csv(index=False).encode("utf-8"),
            file_name=f"clusters_{vis_year}_filtered.csv",
            mime="text/csv",
        )

        st.caption(
            "Note: PC1/PC2 are computed separately per cohort (College vs Non-College) for better local structure; "
            "axes aren’t directly comparable across cohorts."
        )

# -------------------- TAB 3: DRAFT SIM --------------------
with tab3:
    st.subheader("Run Draft Simulation")
    st.caption(
        "Select a year (2020–2024). Uses current weight files if present; "
        "otherwise falls back to defaults inside ml.simulation.load_default_weights()."
    )

    st.markdown(r"""
    **What does the _Composite weight_ do?**  
    Each pick scores player–team pairs by combining **team fit** and a **BPA** pull.

    $$
    \text{Score} =
    \underbrace{\sum_{s\in \text{needs}} w_s\,\text{stat}_{s,\text{norm}}}_{\text{team fit}}
    + \underbrace{\text{composite\_weight}\times \text{mean}(\text{Offense},\text{Defense},\text{General})}_{\text{BPA pull}}
    + \text{cluster adjustment}
    $$

    - Lower values → more **team need** driven.  
    - Higher values → more **BPA** pull.
    """)
    
    sim_year = st.selectbox("Draft year", [2020, 2021, 2022, 2023, 2024], index=4)
    composite_weight = st.slider("Composite weight", 0.0, 1.0, 0.20, 0.05)

    run = st.button(f"Run Simulation ({sim_year})")
    if run:
        try:
            college_w, noncollege_w = load_default_weights()  # checks weights/ as you implemented

            sim_df, acc, lottery, mrr, ndcg = simulate_draft(
                year=sim_year,
                composite_weight=composite_weight,
                plot_distribution=False,
                college_weights=college_w,
                noncollege_weights=noncollege_w
            )

            st.success(
                f"Done for {sim_year}! "
                f"Team-match accuracy: {acc:.2%} | Lottery hit rate: {lottery:.2%} | "
                f"MRR: {mrr:.3f} | nDCG@14: {ndcg:.3f}"
            )

            # Show picks starting at 1
            sim_df.index = sim_df.index + 1
            st.dataframe(sim_df)

            # Export
            csv = sim_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download Simulated Draft (CSV)",
                data=csv,
                file_name=f"simulated_draft_{sim_year}.csv",
                mime="text/csv"
            )

        except FileNotFoundError as e:
            st.error(str(e))
        except Exception as e:
            st.exception(e)