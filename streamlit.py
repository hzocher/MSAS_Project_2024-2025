import streamlit as st
import pandas as pd
import nfl_data_py as nfl
import numpy as np
import matplotlib.pyplot as plt

st.title("4th Down Decision Model")

# Load or fetch data once and cache it
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("pbp2.csv", index_col=0, na_values=["", "NA"], low_memory=False)
    except FileNotFoundError:
        df = nfl.import_pbp_data(
            years=list(range(1999, 2020)) + list(range(2021, 2025)),
            downcast=True,
            cache=False
        )
    df = df[(df.down == 4) & df.epa.notna()]
    df = df[(df['epa'] >= -5) & (df['epa'] <= 5)]
    df = df[df['yardline_100'] <= 75]
    return df

# Load data once at the start
df = load_data()

# Plotting functions
def plot_conversion_vs_punt(pass_run_df, punt_df, ydstogo, yardline):
    pass_run_df = pass_run_df.copy()
    punt_df = punt_df.copy()
    pass_run_df['yard_bin'] = (pass_run_df['yardline_100'] // 5) * 5
    punt_df['yard_bin'] = (punt_df['yardline_100'] // 5) * 5

    pos = pass_run_df[pass_run_df.epa > 0].groupby('yard_bin')['epa'].mean()
    neg = pass_run_df[pass_run_df.epa < 0].groupby('yard_bin')['epa'].mean()
    punt_epa = punt_df.groupby('yard_bin')['epa'].mean()

    converted = pass_run_df['fourth_down_converted'].sum()
    failed = pass_run_df['fourth_down_failed'].sum()
    rate = converted / (converted + failed) if (converted + failed) > 0 else 0

    epa_diff = pos * rate + neg * (1 - rate)
    result = epa_diff - punt_epa

    fig, ax = plt.subplots()
    ax.plot(result.index, result, marker='o', linestyle='-', label='EPA vs Punt', color='gold')
    ax.axhline(y=0, color='black', linestyle='--')
    ax.axvline(x=yardline, color='red', linestyle=':', label='Your Yardline')
    ax.set_title(f'4th and {ydstogo} EPA vs Punt')
    ax.set_xlabel('Yards to End Zone')
    ax.set_ylabel('EPA Difference')
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.set_ylim(-2, 2)
    ax.set_xlim(30, 80)
    ax.legend()
    st.pyplot(fig)

def plot_conversion_vs_fg(pass_run_df, fg_df, full_fg, ydstogo, yardline):
    pass_run_df = pass_run_df.copy()
    fg_df = fg_df.copy()
    full_fg = full_fg.copy()
    
    max_kick = fg_df['yardline_100'].max()
    pass_run_df = pass_run_df[pass_run_df['yardline_100'] <= max_kick]

    pass_run_df['yard_bin'] = (pass_run_df['yardline_100'] // 5) * 5
    fg_df['yard_bin'] = (fg_df['yardline_100'] // 5) * 5
    full_fg['yard_bin'] = (full_fg['yardline_100'] // 5) * 5

    conv = pass_run_df['fourth_down_converted'].sum()
    fail = pass_run_df['fourth_down_failed'].sum()
    rate = conv / (conv + fail) if (conv + fail) > 0 else 0

    pr_pos = pass_run_df[pass_run_df.epa > 0].groupby('yard_bin')['epa'].mean()
    pr_neg = pass_run_df[pass_run_df.epa < 0].groupby('yard_bin')['epa'].mean()
    pr_epa = pr_pos * rate + pr_neg * (1 - rate)

    fg_pos = fg_df[fg_df.epa > 0].groupby('yard_bin')['epa'].mean()
    fg_neg = fg_df[fg_df.epa < 0].groupby('yard_bin')['epa'].mean()
    fg_rates = (full_fg[full_fg.field_goal_result == 'made'].groupby('yard_bin').size()
                / full_fg.groupby('yard_bin').size()).fillna(0)
    fg_epa = fg_pos * fg_rates + fg_neg * (1 - fg_rates)

    result = pr_epa - fg_epa

    fig, ax = plt.subplots()
    ax.plot(result.index, result, marker='o', linestyle='-', color='blue', label='EPA vs Field Goal')
    ax.axhline(y=0, color='grey', linestyle='--')
    ax.axvline(x=yardline, color='red', linestyle=':', label='Your Yardline')
    ax.set_title(f'4th and {ydstogo} EPA vs Field Goal')
    ax.set_xlabel('Yards to End Zone')
    ax.set_ylabel('EPA Difference')
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_xlim(0, 50)
    ax.legend()
    st.pyplot(fig)

def get_decision(df, ydstogo, yardline):
    df = df[df.ydstogo == ydstogo].copy()
    df['yard_bin'] = (df['yardline_100'] // 5) * 5

    pass_run_df = df[df['play_type'].isin(['pass', 'run'])]
    punt_df = df[df['play_type'] == 'punt']
    fg_df = df[df['play_type'] == 'field_goal']

    # Handle binning
    user_bin = (yardline // 5) * 5

    # Conversion data
    conv = pass_run_df['fourth_down_converted'].sum()
    fail = pass_run_df['fourth_down_failed'].sum()
    if conv + fail == 0:
        return "Insufficient data for this situation.", pass_run_df, punt_df, fg_df
    conv_rate = conv / (conv + fail)

    pr_pos = pass_run_df[(pass_run_df.epa > 0) & (pass_run_df.yard_bin == user_bin)].epa.mean()
    pr_neg = pass_run_df[(pass_run_df.epa < 0) & (pass_run_df.yard_bin == user_bin)].epa.mean()
    pr_epa = conv_rate * pr_pos + (1 - conv_rate) * pr_neg

    punt_epa = punt_df[punt_df.yard_bin == user_bin].epa.mean()
    fg_epa = fg_df[fg_df.yard_bin == user_bin].epa.mean()

    options = {
        'Go for it': pr_epa,
        'Punt': punt_epa,
        'Field Goal': fg_epa
    }

    # Pick the best option
    best = max(options.items(), key=lambda x: x[1] if pd.notna(x[1]) else -np.inf)
    recommendation = f"Recommended decision: **{best[0]}** (EPA: {best[1]:.2f})"

    return recommendation, pass_run_df, punt_df, fg_df

# User input
ydstogo = st.number_input("Enter yards to go (1-10):", min_value=1, max_value=10, value=1, step=1)
yardline = st.number_input("Enter yards from end zone (1-75):", min_value=1, max_value=75, value=50, step=1)

# Make decision
if st.button("Get Recommendation"):
    full_fg = df[df.play_type == 'field_goal']
    decision, pass_run_df, punt_df, fg_df = get_decision(df, ydstogo, yardline)
    st.markdown(decision)

    plot_conversion_vs_punt(pass_run_df, punt_df, ydstogo, yardline)
    plot_conversion_vs_fg(pass_run_df, fg_df, full_fg, ydstogo, yardline)