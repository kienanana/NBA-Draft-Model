# ml/profiles.py
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#plt.use("Agg")

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# ----- radar helpers -----
def _to_numeric(df, cols):
    for c in cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def _percent_to_0_1(series):
    s = series.astype(float)
    if s.max(skipna=True) is not None and s.max(skipna=True) > 1.5:
        return s / 100.0
    return s

def _coerce_percent(df, cols):
    for c in cols:
        if c in df.columns:
            df[c] = _percent_to_0_1(df[c])
    return df

def zscore(s):
    m, sd = s.mean(), s.std()
    if sd == 0 or np.isnan(sd):
        return s * 0
    return (s - m) / sd

def minmax01(s):
    mn, mx = s.min(), s.max()
    if mx == mn:
        return s * 0 + 0.5
    return (s - mn) / (mx - mn)

def scale_for_radar(df, cols):
    tmp = df[cols].apply(zscore, axis=0)
    tmp = tmp.apply(minmax01, axis=0)
    return tmp

def radar_plot(values, labels, title="", ax=None):
    N = len(labels)
    angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
    values = values.tolist()
    values += values[:1]
    angles += angles[:1]

    if ax is None:
        ax = plt.subplot(111, polar=True)

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=9)

    ax.set_rlabel_position(0)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels([".25",".5",".75","1.0"], fontsize=8)
    ax.set_ylim(0, 1)

    ax.plot(angles, values, linewidth=2)
    ax.fill(angles, values, alpha=0.15)
    ax.set_title(title, y=1.1, fontsize=12)
    return ax

# ----- public API the app will use -----

RADAR_PERGAME_COLS = ["PTS","AST","TRB","STL","BLK","TOV","FG%","3P%","FT%"]
RADAR_ADV_COLS     = ["PER","TS%","OBPM","DBPM","BPM","OWS","DWS","WS/40"]  # will fallback to WS

def _ensure_cols(df, cols):
    return [c for c in cols if c in df.columns]

def build_profiles(year: int):
    """Return (profiles_df, per_game_cols, advanced_cols) for the given draft year."""
    draft_path = os.path.join(PROJECT_ROOT, "data", "processed", str(year), f"draftpool_stats_{year}.csv")
    if not os.path.exists(draft_path):
        raise FileNotFoundError(f"Missing: {draft_path}")

    draft = pd.read_csv(draft_path).copy()
    draft["Name"] = draft["Name"].str.strip()

    cols_pg  = _ensure_cols(draft, RADAR_PERGAME_COLS)
    cols_adv = _ensure_cols(draft, RADAR_ADV_COLS)
    if "WS/40" not in cols_adv and "WS" in draft.columns:
        cols_adv = [c for c in cols_adv if c != "WS/40"] + ["WS"]

    _to_numeric(draft, cols_pg + cols_adv + ["Age"])
    draft = _coerce_percent(draft, [c for c in cols_pg + cols_adv if "%" in c])

    radar_pg_scaled  = scale_for_radar(draft, cols_pg)
    radar_adv_scaled = scale_for_radar(draft, cols_adv)

    # overall score (without Upside for now; UI can show either)
    z_pg  = draft[cols_pg].apply(zscore, axis=0)
    z_adv = draft[cols_adv].apply(zscore, axis=0)

    pergame_weights  = {c:1.0 for c in cols_pg}
    advanced_weights = {c:1.0 for c in cols_adv}

    def weighted_mean(row, weights):
        if not weights: return 0.0
        tot_w = sum(weights.values())
        if tot_w == 0: return 0.0
        s = 0.0
        for k, w in weights.items():
            if k in row and not pd.isna(row[k]):
                s += row[k] * w
        return s / tot_w

    pg_index  = z_pg.apply(lambda r: weighted_mean(r, pergame_weights), axis=1)
    adv_index = z_adv.apply(lambda r: weighted_mean(r, advanced_weights), axis=1)

    age_min, age_max = draft["Age"].min(), draft["Age"].max()
    if pd.isna(age_min) or pd.isna(age_max) or age_min == age_max:
        age_bonus = 0.0
    else:
        age01 = (draft["Age"] - age_min) / (age_max - age_min)
        age_bonus = (1 - age01)

    pg01  = minmax01(pg_index)
    adv01 = minmax01(adv_index)

    overall01 = 0.45*pg01 + 0.45*adv01 + 0.10*age_bonus
    overall10 = (overall01 * 10).round(2)

    radar_pg_scaled.columns  = [f"pg::{c}"  for c in cols_pg]
    radar_adv_scaled.columns = [f"adv::{c}" for c in cols_adv]

    out = pd.concat([draft[["Name","classification","Age"]], radar_pg_scaled, radar_adv_scaled], axis=1)
    out["OverallScore"] = overall10

    # Try to join Upside/UpdatedOverall if it exists
    up_path = os.path.join(PROJECT_ROOT, "data", "processed", str(year), f"prospect_profiles_with_upside_{year}.csv")
    if os.path.exists(up_path):
        up = pd.read_csv(up_path)
        out = out.merge(up[["Name","UpsideScore","UpdatedOverall"]], on="Name", how="left")

    return out, cols_pg, cols_adv

def get_prospect_profile(year: int, name: str):
    """Return row dict and two matplotlib figures for the radars."""
    prof, cols_pg, cols_adv = build_profiles(year)
    row = prof[prof["Name"].str.lower() == name.strip().lower()]
    if row.empty:
        return None, None, None, None, None
    row = row.iloc[0]

    vals_pg  = row[[f"pg::{c}" for c in cols_pg]].values
    vals_adv = row[[f"adv::{c}" for c in cols_adv]].values

    fig_pg, ax_pg = plt.subplots(subplot_kw=dict(polar=True), figsize=(5, 5))
    radar_plot(vals_pg, cols_pg, title=f"{row['Name']} – Per-Game", ax=ax_pg)

    fig_adv, ax_adv = plt.subplots(subplot_kw=dict(polar=True), figsize=(5, 5))
    radar_plot(vals_adv, cols_adv, title=f"{row['Name']} – Advanced", ax=ax_adv)

    return row.to_dict(), fig_pg, fig_adv, cols_pg, cols_adv

