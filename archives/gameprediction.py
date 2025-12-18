import pandas as pd
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from auth import get_user_shots
from typing import Optional

def main():
    st.title("Golf Game Prediction & Shot Simulation")
    st.markdown("---")
    st.header("Course distances & shot distributions")

    # course distances input
    units = st.radio("Units", options=["yards", "feet"], index=0, horizontal=True)
    def to_yards(x):
        return x if units == "yards" else x / 3.0
    def from_yards(x):
        return x if units == "yards" else x * 3.0

    col_a, col_b = st.columns(2)
    with col_a:
        d_tee_fairway = st.number_input("Distance Tee -> Fairway", min_value=0.0, value=250.0)
        d_fairway_green = st.number_input("Distance Fairway -> Green", min_value=0.0, value=150.0)
        d_rough_green = st.number_input("Distance Rough -> Green", min_value=0.0, value=120.0)
    with col_b:
        d_bunker_green = st.number_input("Distance Bunker -> Green", min_value=0.0, value=60.0)
        d_green_hole = st.number_input("Distance on green (putt) average length", min_value=0.0, value=20.0)
    
    # Precompute states and default positions (yards)
    states = ["Tee", "Fairway", "Rough", "Bunker", "Green", "Hole"]
    pos = {}
    pos["Tee"] = 0.0
    pos["Fairway"] = to_yards(d_tee_fairway)
    pos["Green"] = to_yards(d_tee_fairway + d_fairway_green)
    pos["Hole"] = pos["Green"] + to_yards(d_green_hole)
    pos["Rough"] = pos["Green"] + to_yards(d_rough_green)
    pos["Bunker"] = pos["Green"] + to_yards(d_bunker_green)

    custom_distances = {}
    with st.expander("Optional: override distances for any origin -> target pair"):
        st.caption("Leave blank to use defaults derived from the linear tee->fairway->green->hole model.")
        for o in states:
            cols = st.columns(len(states))
            for j, t in enumerate(states):
                if o == t:
                    cols[j].markdown(f"**{o} → {t}**")
                    continue
                default_val = from_yards(abs(pos[t] - pos[o]))
                # unique key per pair
                key = f"custom_{o}_{t}"
                val = cols[j].number_input(f"{o}→{t}", value=float(default_val), key=key)
                # store as yards
                custom_distances[(o, t)] = to_yards(val)
                
    tol_pct = st.slider("Tolerance window (% of target)", 0.1, 20.0, 2.0, help="Window is ± this percent of the target distance when counting an empirical hit")

    st.write("\nProvide sample distances for the player's shot carries (comma-separated). For example: 70,50,30")

    use_saved = False
    if 'user_id' in st.session_state and st.session_state.user_id:
        use_saved = st.checkbox("Load my saved shots from profile", value=False)
    else:
        st.info("Log in to load saved shot data from your profile (optional).")

    tee_shots = st.text_input("Player tee/drive carries", value="250,240,260")
    approach_shots = st.text_input("Player approach shot carries", value="150,140,160")
    bunker_shots = st.text_input("Player bunker shot carries", value="60,55,65")
    putts = st.text_input("Player putt distances", value="30,20,15,40")

    def parse_list(s: str):
        try:
            parts = [float(x.strip()) for x in s.split(",") if x.strip() != ""]
            return np.array(parts)
        except Exception:
            return np.array([])

    # Default arrays from text input
    tee_arr = parse_list(tee_shots)
    app_arr = parse_list(approach_shots)
    bunk_arr = parse_list(bunker_shots)
    putt_arr = parse_list(putts)

    # If user requests, override with saved shots
    if use_saved:
        try:
            shots = get_user_shots(st.session_state.user_id)
            if shots:
                # shots is list of dicts or list of tuples depending on implementation
                sdf = pd.DataFrame(shots)
                # If the saved format matches playerdata.py, column names include 'Shot Type' and 'Carry (yards)'
                if 'Shot Type' in sdf.columns and 'Carry (yards)' in sdf.columns:
                    # map shot types to arrays
                    drives = sdf[sdf['Shot Type'] == 'Drive']['Carry (yards)'].dropna().astype(float).values
                    approaches = sdf[sdf['Shot Type'].isin(['Approach', 'Iron Shot'])]['Carry (yards)'].dropna().astype(float).values
                    chips = sdf[sdf['Shot Type'] == 'Chip']['Carry (yards)'].dropna().astype(float).values
                    putts_saved = sdf[sdf['Shot Type'] == 'Putt']['Carry (yards)'].dropna().astype(float).values
                    # override arrays where we have saved data
                    if drives.size > 0:
                        tee_arr = drives
                    if approaches.size > 0:
                        app_arr = approaches
                    if chips.size > 0:
                        bunk_arr = chips
                    if putts_saved.size > 0:
                        putt_arr = putts_saved
                else:
                    st.warning("Saved shot data found but in unexpected format; falling back to manual inputs.")
            else:
                st.info("No saved shots found in your profile; using manual inputs.")
        except Exception as e:
            st.error(f"Could not load saved shots: {e}")

    st.subheader("Empirical hit probabilities")

    def empirical_prob(arr: np.ndarray, target_distance: float, tol_pct_local=None):
        if arr.size == 0:
            return None
        if tol_pct_local is None:
            tol_pct_local = tol_pct
        window = max(0.5, (tol_pct_local / 100.0) * target_distance)
        hits = np.abs(arr - target_distance) <= window
        return hits.sum() / arr.size, window

    # convert targets to yards for internal consistency if needed
    t_tee = to_yards(d_tee_fairway)
    t_app = to_yards(d_fairway_green)
    t_bunk = to_yards(d_bunker_green)
    t_putt = to_yards(d_green_hole)

    # convert arrays if input units were feet
    def ensure_yards(arr):
        if arr.size == 0:
            return arr
        return arr if units == "yards" else arr / 3.0

    tee_arr_y = ensure_yards(tee_arr)
    app_arr_y = ensure_yards(app_arr)
    bunk_arr_y = ensure_yards(bunk_arr)
    putt_arr_y = ensure_yards(putt_arr)

    p_tee_hit, w_tee = empirical_prob(tee_arr_y, t_tee)
    p_app_hit, w_app = empirical_prob(app_arr_y, t_app)
    p_bunk_hit, w_bunk = empirical_prob(bunk_arr_y, t_bunk)
    p_putt_hit, w_putt = empirical_prob(putt_arr_y, t_putt)

    st.write("\nFit a smooth probability density (Gaussian kernel or normal approx) from the player's sample and show probability of hitting the course distance.")

    import math

    # safe KDE builder (imports scipy only if available)
    def build_kde_safe_local(arr):
        try:
            from scipy.stats import gaussian_kde
            if arr.size > 1:
                return gaussian_kde(arr)
        except Exception:
            return None
        return None

    # precompute kdes for use in per-pair summary
    tee_kde = build_kde_safe_local(tee_arr_y)
    app_kde = build_kde_safe_local(app_arr_y)
    bunk_kde = build_kde_safe_local(bunk_arr_y)
    putt_kde = build_kde_safe_local(putt_arr_y)

    # normal pdf/cdf helpers (no scipy required)
    def normal_pdf(xs, mu, sd):
        sd = float(sd) if sd > 0 else 1e-6
        xs = np.array(xs, dtype=float)
        return (1.0 / (sd * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((xs - mu) / sd) ** 2)

    def normal_cdf(x, mu, sd):
        sd = float(sd) if sd > 0 else 1e-6
        return 0.5 * (1.0 + math.erf((x - mu) / (sd * math.sqrt(2.0))))

    def fit_and_plot(arr: np.ndarray, target: float, label: str):
        if arr.size == 0:
            st.write(f"No sample data for {label}")
            return
        # KDE
        try:
            kde = build_kde_safe_local(arr)
            xs = np.linspace(max(0, arr.min() - arr.std()), arr.max() + arr.std(), 300)
            ys = kde(xs) if kde is not None else None
            # approximate normal fit
            mu, sd = arr.mean(), arr.std(ddof=0) if arr.size > 1 else 1.0
            normal_pdf_vals = normal_pdf(xs, mu, sd)

            fig, ax = plt.subplots()
            if ys is not None:
                ax.plot(xs, ys, label='KDE')
            ax.plot(xs, normal_pdf_vals, label='Normal fit', linestyle='--')
            ax.axvline(target, color='red', linestyle=':', label=f'Target {target}')
            # compute probability mass near target: integrate pdf in small window
            # window in yards
            window = max(0.5, (tol_pct / 100.0) * target)
            if ys is not None:
                idx = (xs >= target - window) & (xs <= target + window)
                prob_kde = ys[idx].sum() * (xs[1] - xs[0])
            else:
                prob_kde = None
            prob_norm = normal_cdf(target + window, mu, sd) - normal_cdf(target - window, mu, sd)

            ax.set_title(f"{label} distribution (n={arr.size})")
            ax.legend()
            st.pyplot(fig)
            st.write(f"Estimated probability within ±{from_yards(window):.1f} {units}: KDE={prob_kde if prob_kde is not None else 'N/A'}, Normal={prob_norm:.3f}")
        except Exception as e:
            st.write(f"Could not fit distribution for {label}: {e}")

    with st.expander("Show fitted distributions"):
        fit_and_plot(tee_arr, d_tee_fairway, "Tee/drive carry")
        fit_and_plot(app_arr, d_fairway_green, "Approach carry")
        fit_and_plot(bunk_arr, d_bunker_green, "Bunker escape")
        fit_and_plot(putt_arr, d_green_hole, "Putt distance")

    # Personalize transition matrix based on hit probabilities
    st.markdown("---")
    st.header("Personalize transition probabilities")
    st.write("If your historical shots are close to the course distances, increase the probability of transitioning to the desired next state; otherwise reduce it.")
    personalize = st.checkbox("Personalize transition matrix using my shot data", value=False)
    influence = st.slider("Influence of my data on transition probs", 0.0, 1.0, 0.5)

    def adjust_transition_matrix(T_orig: pd.DataFrame, hits: dict, influence: float) -> pd.DataFrame:
        T_adj = T_orig.copy().astype(float)
        # mapping: state -> (target_column, hit_key)
        mapping = {
            "Tee": ("Fairway", "tee"),
            "Fairway": ("Green", "approach"),
            "Rough": ("Green", "approach"),
            "Bunker": ("Green", "bunker"),
            "Green": ("Hole", "putt"),
        }
        for state, (target, hitkey) in mapping.items():
            if state not in T_adj.index or target not in T_adj.columns:
                continue
            hitp = hits.get(hitkey, None)
            if hitp is None:
                continue
            # scale factor between (1-influence) and (1+influence) based on hitp in [0,1]
            # center hitp=0.5 -> factor=1.0
            factor = 1.0 + influence * ((hitp - 0.5) * 2.0)
            # apply to target prob
            T_adj.loc[state, target] = max(0.0, T_adj.loc[state, target] * factor)
            # renormalize row to sum to 1
            row_sum = T_adj.loc[state].sum()
            if row_sum > 0:
                T_adj.loc[state] = T_adj.loc[state] / row_sum
        return T_adj

    hits = {
        "tee": p_tee_hit if p_tee_hit is not None else 0.5,
        "approach": p_app_hit if p_app_hit is not None else 0.5,
        "bunker": p_bunk_hit if p_bunk_hit is not None else 0.5,
        "putt": p_putt_hit if p_putt_hit is not None else 0.5,
    }

    T_personalized = adjust_transition_matrix(T, hits, influence) if personalize else T

    if personalize:
        st.markdown("**Original vs Personalize**")
        c1, c2 = st.columns(2)
        with c1:
            st.write("Original transition matrix")
            st.dataframe(T.style.format("{:.2f}"))
        with c2:
            st.write("Personalized transition matrix")
            st.dataframe(T_personalized.style.format("{:.2f}"))

    # Use T_personalized for analytic and Monte Carlo if personalization is on
    T_used = T_personalized

    # --- Compute probabilities for every combination of shot locations/types ---
    st.markdown("---")
    st.header("Per-pair probabilities (all origin -> target combinations)")
    states = ["Tee", "Fairway", "Rough", "Bunker", "Green", "Hole"]

    # positions in yards along a notional line (Tee -> Fairway -> Green -> Hole)
    pos = {}
    pos["Tee"] = 0.0
    pos["Fairway"] = to_yards(d_tee_fairway)
    pos["Green"] = to_yards(d_tee_fairway + d_fairway_green)
    pos["Hole"] = pos["Green"] + to_yards(d_green_hole)
    pos["Rough"] = pos["Green"] + to_yards(d_rough_green)
    pos["Bunker"] = pos["Green"] + to_yards(d_bunker_green)

    

    # helper to pick shot distribution based on origin
    shot_map = {
        'Tee': ('drive', tee_arr_y, tee_kde),
        'Fairway': ('approach', app_arr_y, app_kde),
        'Rough': ('approach', app_arr_y, app_kde),
        'Bunker': ('bunker', bunk_arr_y, bunk_kde),
        'Green': ('putt', putt_arr_y, putt_kde),
    }

    summary_rows = []

    # function to compute probs for a given arr and target (yards)
    def compute_probs_for_arr(arr_y, kde, target_y):
        if arr_y is None or arr_y.size == 0:
            return None, None
        # empirical
        emp_hits = np.abs(arr_y - target_y) <= max(0.5, (tol_pct / 100.0) * target_y)
        emp_p = emp_hits.sum() / arr_y.size
        # kde and normal
        prob_kde = None
        prob_norm = None
        try:
            if kde is not None:
                xs = np.linspace(max(0, arr_y.min() - arr_y.std()), arr_y.max() + arr_y.std(), 300)
                ys = kde(xs)
                window = max(0.5, (tol_pct / 100.0) * target_y)
                idx = (xs >= target_y - window) & (xs <= target_y + window)
                prob_kde = ys[idx].sum() * (xs[1] - xs[0])
            # normal approx
            mu = float(arr_y.mean())
            sd = float(arr_y.std(ddof=0) if arr_y.size > 1 else max(1.0, 0.1 * mu))
            window = max(0.5, (tol_pct / 100.0) * target_y)
            prob_norm = normal_cdf(target_y + window, mu, sd) - normal_cdf(target_y - window, mu, sd)
        except Exception:
            pass
        return emp_p, prob_kde, prob_norm

    # iterate pairs
    for o in states:
        for t in states:
            if o == t:
                continue
            # use custom override if present
            target_y = custom_distances.get((o, t), abs(pos[t] - pos[o]))
            # pick shot distribution based on origin
            if o in shot_map:
                shot_name, arr_y, kde = shot_map[o]
            else:
                shot_name, arr_y, kde = ('approach', app_arr_y, app_kde)

            emp_p, prob_kde, prob_norm = compute_probs_for_arr(arr_y, kde, target_y)
            summary_rows.append({
                'origin': o,
                'target': t,
                'shot_type': shot_name,
                'distance': from_yards(target_y),
                'empirical_p': emp_p,
                'kde_p': prob_kde,
                'norm_p': prob_norm,
            })

    summary_df = pd.DataFrame(summary_rows)
    # show table with NaNs filled as N/A
    st.dataframe(summary_df.fillna('N/A'))

    # Add expanders to show detailed plots for pairs with data
    for idx, row in summary_df.iterrows():
        if row['empirical_p'] is None:
            continue
        key = f"{row['origin']}->{row['target']} ({row['shot_type']})"
        with st.expander(key):
            arr_y = None
            kde = None
            if row['origin'] in shot_map:
                _, arr_y, kde = shot_map[row['origin']]
            else:
                arr_y, kde = app_arr_y, app_kde
            # plot sample histogram + kde + normal
            arr_disp = from_yards(arr_y)
            tgt_disp = row['distance']
            fig, ax = plt.subplots()
            ax.hist(arr_disp, bins=20, density=True, alpha=0.4)
            try:
                xs = np.linspace(max(0, arr_disp.min() - arr_disp.std()), arr_disp.max() + arr_disp.std(), 300)
                if kde is not None:
                    ys = kde(xs if units == 'yards' else xs / 3.0)
                    ax.plot(xs, ys, label='KDE')
                mu = float(arr_disp.mean())
                sd = float(arr_disp.std(ddof=0) if arr_disp.size > 1 else max(1.0, 0.1 * mu))
                ax.plot(xs, normal_pdf(xs, mu, sd), linestyle='--', label='Normal')
            except Exception:
                pass
            ax.axvline(tgt_disp, color='red', linestyle=':', label=f'Target {tgt_disp:.1f} {units}')
            ax.set_title(f"Samples for {row['origin']} -> {row['target']} ({row['shot_type']})")
            ax.legend()
            st.pyplot(fig)

    # --- Visualize one hole by simulating sequences ---
    st.markdown("---")
    st.header("Simulate & visualize one hole")
    sim_cols = st.columns(3)
    with sim_cols[0]:
        sim_n = st.number_input("Simulations", min_value=100, max_value=20000, value=2000, step=100)
    with sim_cols[1]:
        sample_method = st.selectbox("Sampling method", options=["empirical", "kde", "normal"], index=0)
    with sim_cols[2]:
        max_strokes = st.number_input("Max strokes per hole", min_value=3, max_value=12, value=8)

    def build_kde_safe(arr):
        try:
            from scipy.stats import gaussian_kde
            if arr.size > 1:
                return gaussian_kde(arr)
        except Exception:
            return None
        return None

    tee_kde = build_kde_safe(tee_arr_y)
    app_kde = build_kde_safe(app_arr_y)
    bunk_kde = build_kde_safe(bunk_arr_y)
    putt_kde = build_kde_safe(putt_arr_y)

    def sample(arr_y, kde, method):
        if arr_y.size == 0:
            return 0.0
        if method == 'empirical':
            return float(np.random.choice(arr_y))
        elif method == 'kde' and kde is not None:
            return float(kde.resample(1).flatten()[0])
        else:
            # normal
            mu = float(arr_y.mean())
            sd = float(arr_y.std(ddof=0) if arr_y.size > 1 else max(1.0, 0.1 * mu))
            return float(np.random.normal(mu, sd))

    # hole length in yards
    hole_length = to_yards(d_tee_fairway + d_fairway_green + d_green_hole)

    # run simulations
    landing_positions_per_shot = []  # list of lists
    strokes_result = []
    for _ in range(int(sim_n)):
        remaining = hole_length
        shot_idx = 0
        positions = []
        while remaining > 0 and shot_idx < max_strokes:
            shot_idx += 1
            # choose distribution: first shot -> tee, second+ -> approach unless within putt threshold
            if remaining <= t_putt:
                # use putt distribution
                s = sample(putt_arr_y, putt_kde, sample_method)
            else:
                if shot_idx == 1:
                    s = sample(tee_arr_y, tee_kde, sample_method)
                else:
                    # approach or bunker/chip: use approach distribution
                    s = sample(app_arr_y, app_kde, sample_method)
            # ensure non-negative carry
            s = max(0.0, s)
            remaining = max(0.0, remaining - s)
            positions.append(remaining)
        strokes_result.append(shot_idx)
        # append landing positions for each shot index
        for i, pos in enumerate(positions):
            if len(landing_positions_per_shot) <= i:
                landing_positions_per_shot.append([])
            landing_positions_per_shot[i].append(pos)

    # plot strokes histogram
    fig, ax = plt.subplots()
    ax.hist(strokes_result, bins=range(1, max_strokes + 2), align='left', color='#2ca02c', rwidth=0.8)
    ax.set_xlabel('Strokes to finish hole')
    ax.set_ylabel('Frequency')
    ax.set_title('Simulated strokes distribution (one hole)')
    st.pyplot(fig)

    # plot landing distributions after each shot (converted back to units)
    with st.expander('Landing distributions after each shot'):
        for i, arr in enumerate(landing_positions_per_shot):
            if len(arr) == 0:
                continue
            arr = np.array(arr)
            # convert to display units
            arr_disp = from_yards(arr)
            fig, ax = plt.subplots()
            sns.kdeplot(arr_disp, fill=True, ax=ax)
            ax.set_xlabel(f'Remaining distance after shot {i+1} ({units})')
            ax.set_title(f'Remaining distance after shot {i+1} (n={len(arr)})')
            ax.axvline(0, color='red', linestyle=':')
            st.pyplot(fig)
            

if __name__ == "__main__":
    main()