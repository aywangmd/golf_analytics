import pandas as pd
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from auth import get_user_shots
from typing import Optional


@st.cache_data
def load_data(path: str = 'data/PGA_Data.csv') -> pd.DataFrame:
    import os

    candidates = [path, os.path.join('StreamlitApp', 'pages', 'PGA_Data.csv'), os.path.join('..', 'PGA_Data.csv'), os.path.join('..', '..', 'PGA_Data.csv'), os.path.join('..', 'golfanalytics', 'PGA_Data.csv'), os.path.join('..', 'golf_analytics', 'PGA_Data.csv'), os.path.join('..', 'StreamlitApp', 'pages', 'PGA_Data.csv')]
    found = None
    for c in candidates:
        if c and os.path.exists(c):
            found = c
            break
    if not found:
        raise FileNotFoundError(f'PGA_Data.csv not found. Checked: {candidates}')

    df = pd.read_csv(found)
    cols = [c for c in ['player', 'season', 'sg_ott', 'sg_app', 'sg_arg', 'sg_putt', 'sg_total'] if c in df.columns]
    df = df[cols]
    return df

def normalize(series: pd.Series) -> pd.Series:
    if series.max() == series.min():
        return pd.Series(0.5, index=series.index)
    return (series - series.min()) / (series.max() - series.min())

def build_transition_matrix_from_stats(p_tee_fairway, p_fairway_green, p_rough_green, p_bunker_green, p_green_hole):
    states = ['Tee', 'Fairway', 'Rough', 'Bunker', 'Green', 'Hole']
    T = pd.DataFrame(0.0, index=states, columns=states)

    T.loc['Tee', 'Fairway'] = p_tee_fairway
    T.loc['Tee', 'Rough'] = (1 - p_tee_fairway) * 0.7
    T.loc['Tee', 'Bunker'] = (1 - p_tee_fairway) * 0.3

    T.loc['Fairway', 'Green'] = p_fairway_green
    T.loc['Fairway', 'Rough'] = (1 - p_fairway_green) * 0.5
    T.loc['Fairway', 'Bunker'] = (1 - p_fairway_green) * 0.3
    T.loc['Fairway', 'Fairway'] = (1 - p_fairway_green) * 0.2

    T.loc['Rough', 'Green'] = p_rough_green
    T.loc['Rough', 'Rough'] = (1 - p_rough_green) * 0.5
    T.loc['Rough', 'Bunker'] = (1 - p_rough_green) * 0.5

    T.loc['Bunker', 'Green'] = p_bunker_green
    T.loc['Bunker', 'Rough'] = 1 - p_bunker_green

    T.loc['Green', 'Hole'] = p_green_hole
    T.loc['Green', 'Green'] = 1 - p_green_hole

    T.loc['Hole', 'Hole'] = 1.0

    return T

def compute_player_probs(df: pd.DataFrame, player: Optional[str] = None):
    if player and player != 'All players':
        dff = df[df['player'] == player]
    else:
        dff = df.copy()

    if dff.empty:
        dff = df.copy()

    p_tee_fairway = normalize(dff['sg_ott']).mean() if 'sg_ott' in dff else 0.5
    p_fairway_green = normalize(dff['sg_app']).mean() if 'sg_app' in dff else 0.5
    p_rough_green = normalize(dff['sg_app'] * 0.6).mean() if 'sg_app' in dff else 0.3
    p_bunker_green = normalize(dff['sg_arg']).mean() if 'sg_arg' in dff else 0.4
    p_green_hole = normalize(dff['sg_putt']).mean() if 'sg_putt' in dff else 0.5

    return dict(
        p_tee_fairway=p_tee_fairway,
        p_fairway_green=p_fairway_green,
        p_rough_green=p_rough_green,
        p_bunker_green=p_bunker_green,
        p_green_hole=p_green_hole,
    )

def play_one_hole(T: pd.DataFrame, start_state='Tee', hole_out='Hole') -> int:
    state = start_state
    strokes = 0
    states = list(T.columns)
    while state != hole_out:
        probs = T.loc[state].values
        next_state = np.random.choice(states, p=probs)
        state = next_state
        strokes += 1
    return strokes

def play_round(T: pd.DataFrame, n_holes=18) -> int:
    return sum(play_one_hole(T) for _ in range(n_holes))

def monte_carlo_rounds(T: pd.DataFrame, n_rounds=2000, n_holes=18):
    results = [play_round(T, n_holes) for _ in range(n_rounds)]
    return np.array(results)

def expected_strokes_absorbing(T: pd.DataFrame, start_state='Tee', hole_out='Hole') -> float:
    transient_states = [s for s in T.index if s != hole_out]
    Q = T.loc[transient_states, transient_states].values
    I = np.eye(Q.shape[0])
    N = np.linalg.inv(I - Q)
    ones = np.ones((Q.shape[0], 1))
    t = N.dot(ones)
    expected_steps = dict(zip(transient_states, t.flatten()))
    return expected_steps[start_state]

def plot_transition_heatmap(T: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(T, annot=True, fmt='.2f', cmap='Blues', ax=ax)
    ax.set_title('Transition Matrix')
    return fig

def plot_network(T: pd.DataFrame):
    G = nx.DiGraph()
    for i in T.index:
        for j in T.columns:
            p = float(T.loc[i, j])
            if p > 0:
                G.add_edge(i, j, weight=p)

    pos = nx.spring_layout(G, seed=42)
    fig, ax = plt.subplots(figsize=(7, 6))
    weights = [d['weight'] * 5 for (_, _, d) in G.edges(data=True)]
    nx.draw_networkx_nodes(G, pos, node_color='#1f77b4', node_size=1400, ax=ax)
    nx.draw_networkx_edges(G, pos, arrowstyle='->', arrowsize=12, width=weights, ax=ax)
    nx.draw_networkx_labels(G, pos, font_color='white', ax=ax)
    edge_labels = {(u, v): f'{d['weight']:.2f}' for u, v, d in G.edges(data=True)}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='black', ax=ax)
    ax.set_title('Markov Chain - transitions')
    ax.axis('off')
    return fig

def main():
    st.title('Game Prediction - PGA Players')
    df = load_data()

    players = ['All players'] + sorted(df['player'].unique().tolist()) if 'player' in df.columns else ['All players']
    player = st.sidebar.selectbox('Choose player', players)

    st.sidebar.markdown('---')
    n_rounds = st.sidebar.slider('Monte Carlo rounds', 100, 20000, 2000, step=100)
    n_holes = st.sidebar.slider('Holes per round', 1, 36, 18)
    show_data = st.sidebar.checkbox('Show raw data', value=False)

    st.subheader(f'Player: {player}')
    if show_data:
        st.dataframe(df[df['player'] == player] if player != 'All players' else df)

    probs = compute_player_probs(df, player)
    T = build_transition_matrix_from_stats(**probs)

    st.markdown('### Transition Matrix')
    st.dataframe(T.style.format('{:.2f}'))

    col1, col2 = st.columns(2)
    with col1:
        st.pyplot(plot_transition_heatmap(T))
    with col2:
        st.pyplot(plot_network(T))

    # analytical approach outcome
    exp_one_hole = expected_strokes_absorbing(T)
    st.write(f'Expected strokes for one hole: {exp_one_hole:.2f}')
    st.write(f'Expected strokes for {n_holes} holes: {exp_one_hole * n_holes:.2f}')

    # monte carlo
    if st.button('Run Monte Carlo'):
        with st.spinner('Running simulations...'):
            results = monte_carlo_rounds(T, n_rounds=n_rounds, n_holes=n_holes)
            mean = results.mean()
            std = results.std()
            st.write(f'Monte Carlo expected strokes after {n_holes} holes: {mean:.2f} ± {std:.2f}')
            fig, ax = plt.subplots()
            ax.hist(results, bins=30, color='#4c72b0', alpha=0.8)
            ax.set_xlabel('Strokes')
            ax.set_ylabel('Frequency')
            ax.set_title('Histogram of simulated round totals')
            st.pyplot(fig)

if __name__ == "__main__":
    main()
