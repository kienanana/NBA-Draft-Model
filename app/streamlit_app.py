import os
import io
import numpy as np
import pandas as pd
import streamlit as st
import altair as alt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Local imports
import sys
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)

from ml.profiles import get_prospect_profile, build_profiles
from ml.simulation import simulate_draft
from ml.simulation import load_clusters  
from ml.features import college_features, noncollege_features
from ml.lottery_inclusion import BACKTEST_PATH, MODEL_PATH, assign_teams, forecast, load_class
from app.api_client import RankingAPIError, get_rankings, get_teams

st.set_page_config(page_title="Draft Lab", layout="wide")
st.title("🏀 NBA Draft Lab — Profiles & Simulation")

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Prospect Profiles", "Clustering", "Draft Simulation", "Team Rankings", "Lottery Forecast"
])

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
                st.metric("Profile score (/10)", f"{row['OverallScore']:.2f}")
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
        st.dataframe(summary, width="stretch")

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

        st.dataframe(table, width="stretch", height=360)

        # Optional: split view per cluster in expanders (handy for long lists)
        with st.expander("View by cluster (expandable)", expanded=False):
            for c in chosen_clusters:
                sub = table[table["Cluster"] == c]
                st.markdown(f"**Cluster {c}** — {len(sub)} players")
                st.dataframe(sub.drop(columns=["Cluster"]), width="stretch")

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
        "Select a year (2020–2024). Uses the explainable baseline-v1 scorer. "
        "Historical optimized weights require retraining for this scoring contract."
    )

    st.markdown(r"""
    **What does the _Composite weight_ do?**  
    Each pick combines direction-aware team needs with a baseline prospect-quality pull.

    $$
    \text{Score} =
    \underbrace{\sum_{s\in \text{needs}} w_s\,\text{stat}_{s,\text{norm}}}_{\text{team fit}}
    + \underbrace{\text{composite\_weight}\times \text{mean}(\text{TS\%},\text{PER},\text{AST/TO},\text{Age})}_{\text{prospect quality}}
    $$

    Lower Age, TOV% and DRtg are better. Missing player values receive a neutral
    value of 0.5 and appear in ranking warnings.
    """)
    
    sim_year = st.selectbox("Draft year", [2020, 2021, 2022, 2023, 2024], index=4)
    composite_weight = st.slider("Composite weight", 0.0, 1.0, 0.20, 0.05)

    run = st.button(f"Run Simulation ({sim_year})")
    if run:
        try:
            sim_df, acc, lottery, first_hit_rr, ndcg = simulate_draft(
                year=sim_year,
                composite_weight=composite_weight,
                plot_distribution=False,
            )

            st.success(
                f"Done for {sim_year} using baseline-v1! "
                f"Team-match accuracy: {acc:.2%} | Lottery hit rate: {lottery:.2%} | "
                f"First-hit reciprocal rank: {first_hit_rr:.3f} | nDCG@14: {ndcg:.3f}"
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

# -------------------- TAB 4: EXPLAINABLE TEAM RANKINGS --------------------
with tab4:
    st.subheader("Team candidate rankings")
    st.caption(
        "Rank a historical draft class for one team's needs. Scores come from the "
        "FastAPI service and show the contribution of every scored stat."
    )
    ranking_year = st.selectbox("Draft year (rankings)", [2020, 2021, 2022, 2023, 2024], index=4)
    try:
        ranking_teams = get_teams(ranking_year)
    except RankingAPIError as exc:
        st.error(str(exc))
        ranking_teams = []

    if ranking_teams:
        ranking_team = st.selectbox("Team", ranking_teams)
        ranking_weight = st.slider("Prospect quality weight", 0.0, 1.0, 0.2, 0.05)
        try:
            ranking = get_rankings(ranking_year, ranking_team, ranking_weight)
        except RankingAPIError as exc:
            st.error(str(exc))
        else:
            st.caption(
                f"Model: {ranking['model_version']} · {ranking['total_candidates']} candidates · "
                "stats scaled within college and non-college groups"
            )
            if ranking["warnings"]:
                st.warning("; ".join(ranking["warnings"]))
            with st.expander("Team needs used for this ranking"):
                if ranking["team_needs"]:
                    needs_table = pd.DataFrame([
                        {"Stat": stat, "Need weight": weight}
                        for stat, weight in ranking["team_needs"].items()
                    ])
                    st.dataframe(needs_table, hide_index=True, width="stretch")
                else:
                    st.info("No below-average team stats were identified; only prospect quality contributes.")

            candidates = ranking["candidates"]
            candidate_names = [candidate["player"] for candidate in candidates]
            table = pd.DataFrame([{
                "Rank": candidate["rank"],
                "Prospect": candidate["player"],
                "Cohort": candidate["classification"],
                "Score": candidate["score"],
                "Warnings": len(candidate["warnings"]),
            } for candidate in candidates])
            st.dataframe(table, hide_index=True, width="stretch")

            if candidates:
                st.markdown("#### Explain a score")
                selected_name = st.selectbox("Prospect to explain", candidate_names)
                selected = next(candidate for candidate in candidates if candidate["player"] == selected_name)
                st.metric("Total score", f"{selected['score']:.3f}")
                contribution_table = pd.DataFrame([
                    {"Contribution": stat.replace("team_fit:", "Team fit · ").replace("quality:", "Quality · "),
                     "Points": points}
                    for stat, points in selected["contributions"].items()
                ])
                st.bar_chart(contribution_table.set_index("Contribution"))
                if selected["warnings"]:
                    st.warning("; ".join(selected["warnings"]))

            if len(candidates) >= 2:
                st.markdown("#### Compare two candidates")
                left, right = st.columns(2)
                with left:
                    first_name = st.selectbox("First candidate", candidate_names, index=0)
                with right:
                    second_options = [name for name in candidate_names if name != first_name]
                    second_name = st.selectbox("Second candidate", second_options)
                first = next(candidate for candidate in candidates if candidate["player"] == first_name)
                second = next(candidate for candidate in candidates if candidate["player"] == second_name)
                comparison = pd.DataFrame({
                    "Contribution": sorted(set(first["contributions"]) | set(second["contributions"])),
                })
                comparison[first_name] = comparison["Contribution"].map(first["contributions"]).fillna(0.0)
                comparison[second_name] = comparison["Contribution"].map(second["contributions"]).fillna(0.0)
                comparison["Difference"] = comparison[first_name] - comparison[second_name]
                st.dataframe(comparison, hide_index=True, width="stretch")
                st.caption(f"Total: {first_name} {first['score']:.3f} · {second_name} {second['score']:.3f}")
                for candidate in (first, second):
                    if candidate["warnings"]:
                        st.warning(f"{candidate['player']}: {'; '.join(candidate['warnings'])}")

# -------------------- TAB 5: LOTTERY INCLUSION --------------------
with tab5:
    st.subheader("Lottery Forecast")
    st.caption(
        "Prospect-only ranking for the 2024 lottery. The model was trained on "
        "2020–2023 and selects its top 14 prospects; team matches do not affect training."
    )
    st.info(
        "This is a historical replay. Prospect lists came from Tankathon's past-drafts "
        "pages, so these results are not prospective pre-draft accuracy."
    )
    if not MODEL_PATH.exists() or not BACKTEST_PATH.exists():
        st.warning("Model files are missing. Run `python -m ml.lottery_inclusion` from the project root.")
    else:
        import joblib

        artifact = joblib.load(MODEL_PATH)
        ranked = forecast(2024, artifact)
        top_14 = ranked.head(14)
        actual = load_class(2024).set_index("Name")["lottery"]
        st.metric("2024 lottery players in model top 14", f"{int(actual.loc[top_14['Name']].sum())} / 14")
        st.dataframe(
            top_14[["rank", "Name", "classification", "model_score"]].rename(columns={
                "rank": "Rank", "Name": "Prospect", "classification": "Cohort",
                "model_score": "Model score",
            }),
            hide_index=True, width="stretch",
        )
        st.caption("Model scores order candidates; they are not calibrated probabilities.")
        with st.expander("Optional team assignment"):
            assigned, assignment_metrics = assign_teams(2024, ranked)
            st.caption(
                "The existing team-fit scorer assigns only the model's selected 14 players. "
                "This step does not affect model training or lottery inclusion."
            )
            st.metric("Exact team matches", f"{int(assignment_metrics['team_match'] * 14)} / 14")
            st.dataframe(assigned, hide_index=True, width="stretch")
        with st.expander("All candidates"):
            st.dataframe(ranked, hide_index=True, width="stretch")
        with st.expander("Year-by-year backtest"):
            st.dataframe(pd.read_csv(BACKTEST_PATH), hide_index=True, width="stretch")
