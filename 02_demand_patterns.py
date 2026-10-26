import pandas as pd
import matplotlib.pyplot as plt


# --------------------------------------------------
# 1. Load cleaned data
# --------------------------------------------------

df = pd.read_csv("data/demand_forecast_clean.csv")

df["Datetime"] = pd.to_datetime(df["Datetime"])
df["Date"] = pd.to_datetime(df["Date"])


# --------------------------------------------------
# 2. Convert demand from MW to GW
# --------------------------------------------------

df["Demand_GW"] = df["Demand_Outturn"] / 1000


# ==================================================
# 3. DEMAND THROUGHOUT THE DAY
# ==================================================

demand_by_time = (
    df.groupby(["Hour", "Minute"])["Demand_GW"]
    .mean()
    .reset_index()
)

demand_by_time["Time_of_Day"] = (
    demand_by_time["Hour"].astype(str).str.zfill(2)
    + ":"
    + demand_by_time["Minute"].astype(str).str.zfill(2)
)

# --------------------------------------------------
# Identify major peaks and troughs
# --------------------------------------------------

demand = demand_by_time["Demand_GW"]

# Compare each point with nearby points.
# This helps identify meaningful turning points
# rather than every small fluctuation.

window = 3

rolling_max = demand.rolling(
    window=window * 2 + 1,
    center=True
).max()

rolling_min = demand.rolling(
    window=window * 2 + 1,
    center=True
).min()

major_peaks = demand == rolling_max
major_troughs = demand == rolling_min

# --------------------------------------------------
# Plots
# --------------------------------------------------

plt.figure(figsize=(11, 5.5))

plt.plot(
    demand_by_time["Time_of_Day"],
    demand,
    linewidth=2
)

# Major peaks
plt.plot(
    demand_by_time.loc[major_peaks, "Time_of_Day"],
    demand[major_peaks],
    marker="o",
    linestyle="none"
)

# Major troughs
plt.plot(
    demand_by_time.loc[major_troughs, "Time_of_Day"],
    demand[major_troughs],
    marker="o",
    linestyle="none"
)

plt.xlabel("Time of day")
plt.ylabel("Average electricity demand (GW)")
plt.title("Average GB Electricity Demand Throughout the Day")

plt.xticks(
    range(0, 48, 4),
    demand_by_time["Time_of_Day"].iloc[::4]
)

plt.grid(axis="y", alpha=0.25)
plt.tight_layout()
plt.show()


# ==================================================
# 4. DEMAND BY DAY OF WEEK
# ==================================================

weekday_labels = [
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday"
]

demand_by_weekday = (
    df.groupby("Weekday_Num")["Demand_GW"]
    .mean()
    .reindex(range(7))
)


print("\n--- AVERAGE DEMAND BY DAY OF WEEK ---")
print(
    pd.DataFrame({
        "Day": weekday_labels,
        "Average_Demand_GW": demand_by_weekday.values
    })
)


plt.figure(figsize=(10, 5.5))

plt.plot(
    weekday_labels,
    demand_by_weekday.values,
    marker="o",
    linewidth=2
)

plt.xlabel("Day of week")
plt.ylabel("Average electricity demand (GW)")
plt.title("Average GB Electricity Demand by Day of Week")

plt.grid(axis="y", alpha=0.25)

plt.tight_layout()
plt.show()


# ==================================================
# 5. DEMAND BY MONTH
# ==================================================

month_labels = [
    "Jan", "Feb", "Mar", "Apr", "May", "Jun",
    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"
]

demand_by_month = (
    df.groupby("Month_Num")["Demand_GW"]
    .mean()
    .reindex(range(1, 13))
)


print("\n--- AVERAGE DEMAND BY MONTH ---")
print(
    pd.DataFrame({
        "Month": month_labels,
        "Average_Demand_GW": demand_by_month.values
    })
)


plt.figure(figsize=(10, 5.5))

plt.plot(
    month_labels,
    demand_by_month.values,
    marker="o",
    linewidth=2
)

plt.xlabel("Month")
plt.ylabel("Average electricity demand (GW)")
plt.title("Average GB Electricity Demand by Month")

plt.grid(axis="y", alpha=0.25)

plt.tight_layout()
plt.show()


# ==================================================
# 6. HOW OFTEN DIFFERENT DEMAND LEVELS OCCUR
# ==================================================

plt.figure(figsize=(10, 5.5))
plt.hist(
    df["Demand_GW"],
    bins=50
)

plt.xlabel("Actual electricity demand (GW)")
plt.ylabel("Number of half-hour observations")
plt.title("How Often Different Levels of Electricity Demand Occur")

plt.grid(axis="y", alpha=0.25)
plt.tight_layout()
plt.show()


# ==================================================
# 7. BASELINE DEMAND STATISTICS
# ==================================================

print("\n" + "=" * 50)
print("BASELINE DEMAND STATISTICS")
print("=" * 50)

# --------------------------------------------------
# Overall demand
# --------------------------------------------------

print("\n--- OVERALL DEMAND ---")

print(f"Mean demand:   {df['Demand_GW'].mean():.2f} GW")
print(f"Median demand: {df['Demand_GW'].median():.2f} GW")
print(f"Minimum demand:{df['Demand_GW'].min():.2f} GW")
print(f"Maximum demand:{df['Demand_GW'].max():.2f} GW")
print(f"Std deviation: {df['Demand_GW'].std():.2f} GW")

# --------------------------------------------------
# Intraday peaks and troughs
# --------------------------------------------------

print("\n--- INTRADAY PEAKS AND TROUGHS ---")

highest_time = demand_by_time.loc[
    demand_by_time["Demand_GW"].idxmax()
]

lowest_time = demand_by_time.loc[
    demand_by_time["Demand_GW"].idxmin()
]

print(
    f"Highest average demand: "
    f"{highest_time['Demand_GW']:.2f} GW at "
    f"{highest_time['Time_of_Day']}"
)

print(
    f"Lowest average demand: "
    f"{lowest_time['Demand_GW']:.2f} GW at "
    f"{lowest_time['Time_of_Day']}"
)

# --------------------------------------------------
# Day of week
# --------------------------------------------------

print("\n--- DAY OF WEEK ---")

weekday_stats = (
    df.groupby("Weekday_Num")["Demand_GW"]
    .agg(["mean", "median", "min", "max"])
    .reindex(range(7))
)

weekday_stats.index = weekday_labels

weekday_stats.columns = [
    "Mean_GW",
    "Median_GW",
    "Minimum_GW",
    "Maximum_GW"
]

print(weekday_stats.round(2))

print(
    f"\nHighest-demand day: "
    f"{weekday_stats['Mean_GW'].idxmax()} "
    f"({weekday_stats['Mean_GW'].max():.2f} GW)"
)

print(
    f"Lowest-demand day: "
    f"{weekday_stats['Mean_GW'].idxmin()} "
    f"({weekday_stats['Mean_GW'].min():.2f} GW)"
)

# --------------------------------------------------
# Month
# --------------------------------------------------

print("\n--- MONTH ---")

month_stats = (
    df.groupby("Month_Num")["Demand_GW"]
    .agg(["mean", "median", "min", "max"])
    .reindex(range(1, 13))
)

month_stats.index = month_labels

month_stats.columns = [
    "Mean_GW",
    "Median_GW",
    "Minimum_GW",
    "Maximum_GW"
]

print(month_stats.round(2))

print(
    f"\nHighest-demand month: "
    f"{month_stats['Mean_GW'].idxmax()} "
    f"({month_stats['Mean_GW'].max():.2f} GW)"
)

print(
    f"Lowest-demand month: "
    f"{month_stats['Mean_GW'].idxmin()} "
    f"({month_stats['Mean_GW'].min():.2f} GW)"
)

# --------------------------------------------------
# Weekday vs weekend
# --------------------------------------------------

print("\n--- WEEKDAY VS WEEKEND ---")

weekday_weekend = (
    df.groupby("Is_Weekend")["Demand_GW"]
    .agg(["mean", "median", "min", "max"])
)

weekday_weekend.index = [
    "Weekday",
    "Weekend"
]

weekday_weekend.columns = [
    "Mean_GW",
    "Median_GW",
    "Minimum_GW",
    "Maximum_GW"
]

print(weekday_weekend.round(2))

# --------------------------------------------------
# Demand range
# --------------------------------------------------

print("\n--- DEMAND RANGE ---")

demand_range = (
    df["Demand_GW"].max()
    - df["Demand_GW"].min()
)

print(f"Observed demand range: {demand_range:.2f} GW")


# ==================================================
# DEMAND DYNAMICS
# ==================================================

# Change in demand from the previous settlement period
# Calculated within each date to avoid comparing the end of one day
# with the beginning of the next.
df["Demand_Change_GW"] = (
    df.groupby("Date")["Demand_GW"].diff()
)

df["Absolute_Demand_Change_GW"] = (
    df["Demand_Change_GW"].abs()
)

print("\n--- DEMAND CHANGE BETWEEN HALF-HOUR PERIODS ---")

demand_change_stats = df["Demand_Change_GW"].describe(
    percentiles=[0.50, 0.75, 0.90, 0.95, 0.99]
)

print(demand_change_stats.round(2))

print("\n--- ABSOLUTE DEMAND CHANGE ---")

absolute_change_stats = (
    df["Absolute_Demand_Change_GW"]
    .describe(percentiles=[0.50, 0.75, 0.90, 0.95, 0.99])
)

print(absolute_change_stats.round(2))


# Average absolute demand change by time of day
demand_change_by_time = (
    df.groupby(["Hour", "Minute"])["Absolute_Demand_Change_GW"]
    .mean()
    .reset_index()
)

demand_change_by_time["Time_of_Day"] = (
    demand_change_by_time["Hour"].astype(str).str.zfill(2)
    + ":"
    + demand_change_by_time["Minute"].astype(str).str.zfill(2)
)

print("\n--- AVERAGE DEMAND CHANGE BY TIME OF DAY ---")

print(
    demand_change_by_time[
        ["Time_of_Day", "Absolute_Demand_Change_GW"]
    ].round(2)
)


# Plot: how dynamic demand is throughout the day
plt.figure(figsize=(11, 5.5))

plt.plot(
    demand_change_by_time["Time_of_Day"],
    demand_change_by_time["Absolute_Demand_Change_GW"],
    linewidth=2
)

plt.xlabel("Time of day")
plt.ylabel("Average absolute demand change (GW)")
plt.title("How Quickly Electricity Demand Changes Throughout the Day")

plt.xticks(
    range(0, 48, 4),
    demand_change_by_time["Time_of_Day"].iloc[::4]
)

plt.grid(axis="y", alpha=0.25)
plt.tight_layout()
plt.show()


### Demand baseline: Across the study period, average half-hourly GB electricity demand was 26.24 GW, with a median of 25.39 GW and a standard deviation of 5.97 GW. 
# Observed demand ranged from 12.62 GW to 47.32 GW, a total range of 34.70 GW. Demand followed a pronounced daily cycle, reaching its lowest average level at 
# 05:00 (19.98 GW) and its highest at 19:00 (32.09 GW). Demand was also systematically higher on weekdays than weekends, averaging 27.12 GW versus 24.03 GW. 
# Among individual weekdays, Wednesday had the highest average demand (27.41 GW) and Sunday the lowest (23.82 GW), with Tuesday to Thursday generally higher than 
# Monday and Friday. Seasonal variation was substantial: average demand was highest in January (32.20 GW) and lowest in August (22.66 GW). Demand was particularly 
# high from November through February, while May through August had the lowest average levels. Overall, the data show strong intraday, weekly and seasonal structure, 
# alongside substantial variation in the level of demand.
