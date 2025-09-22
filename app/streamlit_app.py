import os
import io
import numpy as np
import pandas as pd
import streamlit as st

# Local imports
import sys
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
#sys.path.append(PROJECT_ROOT)

from ml.profiles import get_prospect_profile, build_profiles
from ml.simulation import simulate_draft, load_default_weights

st.set_page_config(page_title="Draft Lab", layout="wide")
st.title("🏀 NBA Draft Lab — Profiles & Simulation")

tab1, tab2 = st.tabs(["Prospect Profiles", "Draft Simulation (2024)"])

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

# -------------------- TAB 2: DRAFT SIM --------------------
with tab2:
    st.subheader("Run 2024 Draft Simulation")
    st.caption("Uses the current weight files if present; otherwise falls back to defaults inside ml.simulation.load_default_weights().")

    composite_weight = st.slider("Composite weight", 0.0, 1.0, 0.20, 0.05)

    run = st.button("Run Simulation")
    if run:
        try:
            college_w, noncollege_w = load_default_weights()  # your helper that checks ../weights
            # NOTE: expects ../data/processed/2024/... to exist
            sim_df, acc, lottery, mrr, ndcg = simulate_draft(
                year=2024,
                composite_weight=composite_weight,
                plot_distribution=False,
                college_weights=college_w,
                noncollege_weights=noncollege_w
            )

            st.success(f"Done! Team-match accuracy: {acc:.2%} | Lottery hit rate: {lottery:.2%} | MRR: {mrr:.3f} | nDCG@14: {ndcg:.3f}")
            st.dataframe(sim_df)

            # Export
            csv = sim_df.to_csv(index=False).encode("utf-8")
            st.download_button("Download Simulated Draft (CSV)", data=csv, file_name="simulated_draft_2024.csv", mime="text/csv")

        except FileNotFoundError as e:
            st.error(str(e))
        except Exception as e:
            st.exception(e)