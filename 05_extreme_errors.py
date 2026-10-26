import pandas as pd
import matplotlib.pyplot as plt

# ==================================================
# 1. LOAD AND PREPARE DATA
# ==================================================

df = pd.read_csv("data/demand_forecast_clean.csv")

df["Datetime"] = pd.to_datetime(df["Datetime"])
df["Date"] = pd.to_datetime(df["Date"])

df = df.sort_values(
    ["Date", "Settlement_Period"]
).reset_index(drop=True)

df["Forecast_Error"] = (
    df["Demand_Forecast"] - df["Demand_Outturn"]
)

df["Absolute_Error"] = df["Forecast_Error"].abs()

df["APE"] = (
    df["Absolute_Error"] / df["Demand_Outturn"]
) * 100

df["Demand_GW"] = df["Demand_Outturn"] / 1000

# Calculate demand changes within each date.
# This avoids comparing the end of one day with the start
# of the next day.

df["Demand_Change_GW"] = (
    df.groupby("Date")["Demand_GW"].diff()
)

df["Absolute_Demand_Change_GW"] = (
    df["Demand_Change_GW"].abs()
)

# Position of each observation within its day.
df["Row_In_Day"] = df.groupby("Date").cumcount()


# ==================================================
# 2. DEFINE EXTREME ERRORS
# ==================================================

threshold_95 = df["Absolute_Error"].quantile(0.95)
threshold_99 = df["Absolute_Error"].quantile(0.99)

df["Is_Extreme"] = (
    df["Absolute_Error"] >= threshold_95
)

print("\n" + "=" * 55)
print("FILE 5: TEMPORAL PATTERNS AND ERROR RELATIONSHIPS")
print("=" * 55)

print(f"\n95th percentile error: {threshold_95:.0f} MW")
print(f"99th percentile error: {threshold_99:.0f} MW")

print(
    f"Observations in the largest 5% of errors: "
    f"{df['Is_Extreme'].sum():,}"
)


# ==================================================
# QUESTION 1:
# ARE RAPID DEMAND CHANGES ASSOCIATED WITH LARGER ERRORS?
# ==================================================

# Remove the first observation of each day because it has
# no preceding half-hour observation within that date.

relationship_df = df.dropna(
    subset=["Absolute_Demand_Change_GW"]
).copy()

# Divide observations into five groups by the size of
# the preceding half-hour demand change.

relationship_df["Change_Group"] = pd.qcut(
    relationship_df["Absolute_Demand_Change_GW"],
    q=5,
    labels=[
        "Smallest 20%",
        "20–40%",
        "40–60%",
        "60–80%",
        "Largest 20%"
    ]
)

change_summary = (
    relationship_df
    .groupby("Change_Group", observed=False)
    .agg(
        Observations=("Absolute_Error", "size"),
        Median_Demand_Change_GW=(
            "Absolute_Demand_Change_GW", "median"
        ),
        Median_Error_MW=("Absolute_Error", "median"),
        Mean_Error_MW=("Absolute_Error", "mean"),
        Mean_APE=("APE", "mean")
    )
    .round(2)
)

print("\n--- DEMAND CHANGE AND FORECAST ERROR ---")
print(change_summary.to_string())


# ==================================================
# FIGURE 1: DOES FASTER DEMAND CHANGE MEAN LARGER ERRORS?
# ==================================================

plot_data = change_summary.copy()

# Use the median demand change to give each group
# a meaningful, numeric label.
plot_data["Bucket_Label"] = (
    plot_data["Median_Demand_Change_GW"]
    .map(lambda x: f"{x:.2f} GW")
)

# Order groups from the smallest to largest demand change.
plot_data = plot_data.sort_values(
    "Median_Demand_Change_GW"
)

plt.figure(figsize=(10, 5.5))

plt.plot(
    plot_data["Bucket_Label"],
    plot_data["Median_Error_MW"],
    marker="o",
    markersize=8,
    linewidth=2
)

# Overall median error as a reference point.
overall_median_error = relationship_df["Absolute_Error"].median()

plt.axhline(
    overall_median_error,
    linestyle="--",
    linewidth=1.5,
    label=f"Overall median error: {overall_median_error:.0f} MW"
)

plt.xlabel(
    "Median absolute demand change from previous half-hour (GW)"
)

plt.ylabel("Median absolute forecast error (MW)")

plt.title(
    "Do Larger Demand Changes Coincide With Larger Forecast Errors?"
)

plt.legend()
plt.grid(axis="y", alpha=0.25)
plt.tight_layout()
plt.show()


# ==================================================
# QUESTION 2:
# WHAT HAPPENS AROUND AN EXTREME-ERROR EPISODE?
# ==================================================

# An episode is a run of consecutive half-hour observations
# where absolute forecast error is at or above the 95th
# percentile threshold.
#
# Episodes cannot continue across different dates.

previous_extreme = (
    df.groupby("Date")["Is_Extreme"]
    .shift(fill_value=False)
)

previous_period = (
    df.groupby("Date")["Settlement_Period"]
    .shift()
)

new_episode = (
    df["Is_Extreme"]
    & (
        ~previous_extreme
        | (
            df["Settlement_Period"]
            != previous_period + 1
        )
    )
)

df["Episode_ID"] = new_episode.cumsum()

df.loc[~df["Is_Extreme"], "Episode_ID"] = pd.NA


# Summarise each episode.

episode_records = []

extreme_rows = df[df["Is_Extreme"]]

for episode_id, episode in extreme_rows.groupby("Episode_ID"):

    episode = episode.sort_values("Settlement_Period")

    peak_idx = episode["Absolute_Error"].idxmax()
    peak = df.loc[peak_idx]

    episode_records.append({
        "Episode_ID": int(episode_id),
        "Date": episode["Date"].iloc[0].date(),
        "Start_Period": episode["Settlement_Period"].iloc[0],
        "End_Period": episode["Settlement_Period"].iloc[-1],
        "Duration_Minutes": len(episode) * 30,
        "Extreme_Observations": len(episode),
        "Peak_Error_MW": peak["Absolute_Error"],
        "Demand_at_Peak_GW": peak["Demand_GW"],
        "Peak_Direction": (
            "Forecast too high"
            if peak["Forecast_Error"] > 0
            else "Forecast too low"
            if peak["Forecast_Error"] < 0
            else "Exact"
        ),
        "Start_Row": episode["Row_In_Day"].iloc[0]
    })

episodes = pd.DataFrame(episode_records)

episodes = episodes.sort_values(
    "Peak_Error_MW",
    ascending=False
).reset_index(drop=True)

print("\n--- EXTREME-ERROR EPISODES ---")

print(f"Number of episodes: {len(episodes):,}")

print(
    f"Single-period episodes: "
    f"{(episodes['Extreme_Observations'] == 1).sum():,}"
)

print(
    f"Multi-period episodes: "
    f"{(episodes['Extreme_Observations'] > 1).sum():,}"
)

print(
    f"Longest episode: "
    f"{episodes['Duration_Minutes'].max():.0f} minutes"
)

print("\n--- TEN EPISODES WITH THE LARGEST PEAK ERRORS ---")

print(
    episodes[
        [
            "Date",
            "Start_Period",
            "End_Period",
            "Duration_Minutes",
            "Peak_Error_MW",
            "Demand_at_Peak_GW",
            "Peak_Direction"
        ]
    ]
    .head(10)
    .round(2)
    .to_string(index=False)
)


# ==================================================
# ALIGN OBSERVATIONS AROUND EPISODE STARTS
# ==================================================

# Compare the four half-hours before an episode begins
# with the periods during and after its start.
#
# Relative period 0 = first extreme-error observation.
# Negative values = before the episode.
# Positive values = after the episode begins.

event_window = range(-4, 9)

event_records = []

for _, episode in episodes.iterrows():

    day = df[df["Date"].dt.date == episode["Date"]]

    start_row = int(episode["Start_Row"])

    for relative_period in event_window:

        target_row = start_row + relative_period

        if target_row < 0 or target_row >= len(day):
            continue

        observation = day.iloc[target_row]

        event_records.append({
            "Episode_ID": episode["Episode_ID"],
            "Relative_Half_Hours": relative_period,
            "Absolute_Error": observation["Absolute_Error"]
        })

event_df = pd.DataFrame(event_records)

event_profile = (
    event_df
    .groupby("Relative_Half_Hours")["Absolute_Error"]
    .agg(["mean", "median", "count"])
    .reindex(event_window)
)

print("\n--- ERROR PROFILE AROUND EPISODE STARTS ---")

print(
    event_profile
    .rename(columns={
        "mean": "Mean_Error_MW",
        "median": "Median_Error_MW",
        "count": "Episodes_Represented"
    })
    .round(2)
    .to_string()
)


# Figure 2: Average error before, during and after
# the start of an extreme-error episode.

plt.figure(figsize=(10, 6))

plt.plot(
    event_profile.index,
    event_profile["mean"],
    marker="o",
    linewidth=2,
    label="Mean absolute error"
)

plt.plot(
    event_profile.index,
    event_profile["median"],
    marker="o",
    linewidth=2,
    label="Median absolute error"
)

plt.axhline(
    threshold_95,
    linestyle="--",
    linewidth=1.5,
    label="Extreme-error threshold"
)

plt.axvline(
    0,
    linestyle=":",
    linewidth=1.5,
    color="black",
    label="Episode begins"
)

plt.xlabel("Half-hour periods relative to episode start")
plt.ylabel("Absolute forecast error (MW)")
plt.title("Forecast Errors Before, During and After Extreme Episodes")

plt.xticks(
    list(event_window),
    [
        "Start" if i == 0
        else f"{abs(i)} before" if i < 0
        else f"{i} after"
        for i in event_window
    ],
    rotation=45
)

plt.legend()
plt.grid(axis="y", alpha=0.25)
plt.tight_layout()
plt.show()
