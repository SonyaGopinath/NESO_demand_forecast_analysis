import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("data/demand_forecast_clean.csv")

df["Datetime"] = pd.to_datetime(df["Datetime"])
df["Date"] = pd.to_datetime(df["Date"])

# --------------------------------------------------
# FORECAST ERROR MEASURES
# --------------------------------------------------

df["Forecast_Error"] = (
    df["Demand_Forecast"] - df["Demand_Outturn"]
)

df["Absolute_Error"] = df["Forecast_Error"].abs()

df["APE"] = (
    df["Absolute_Error"] / df["Demand_Outturn"]
) * 100

df["Demand_GW"] = df["Demand_Outturn"] / 1000


# --------------------------------------------------
# 1. FORECAST INACCURACY BY TIME OF DAY
# --------------------------------------------------

error_by_time = (
    df.groupby(["Hour", "Minute"])["APE"]
    .mean()
    .reset_index()
)

error_by_time["Time_of_Day"] = (
    error_by_time["Hour"].astype(str).str.zfill(2)
    + ":"
    + error_by_time["Minute"].astype(str).str.zfill(2)
)

print("\n--- FORECAST INACCURACY BY TIME OF DAY ---")

print(
    error_by_time[
        ["Time_of_Day", "APE"]
    ].round(2)
)

highest_time = error_by_time.loc[
    error_by_time["APE"].idxmax()
]

lowest_time = error_by_time.loc[
    error_by_time["APE"].idxmin()
]

print(
    f"\nHighest average error: "
    f"{highest_time['APE']:.2f}% at "
    f"{highest_time['Time_of_Day']}"
)

print(
    f"Lowest average error: "
    f"{lowest_time['APE']:.2f}% at "
    f"{lowest_time['Time_of_Day']}"
)


plt.figure(figsize=(11, 5.5))

plt.plot(
    error_by_time["Time_of_Day"],
    error_by_time["APE"],
    linewidth=2,
    marker="o",
    markersize=3
)

plt.xlabel("Time of day")
plt.ylabel("Mean absolute forecast error (%)")
plt.title("Forecast Inaccuracy Throughout the Day")

plt.xticks(
    range(0, 48, 4),
    error_by_time["Time_of_Day"].iloc[::4]
)

plt.grid(axis="y", alpha=0.25)
plt.tight_layout()
plt.show()


# --------------------------------------------------
# 2. FORECAST INACCURACY BY DEMAND LEVEL
# --------------------------------------------------

df["Demand_Quintile"], demand_bins = pd.qcut(
    df["Demand_GW"],
    q=5,
    retbins=True,
    labels=[
        "Lowest 20%",
        "20–40%",
        "40–60%",
        "60–80%",
        "Highest 20%"
    ]
)

error_by_demand = (
    df.groupby(
        "Demand_Quintile",
        observed=False
    )["APE"]
    .mean()
)

print("\n--- FORECAST INACCURACY BY DEMAND LEVEL ---")

print(
    error_by_demand.round(2)
)


plt.figure(figsize=(10, 5.5))

plt.bar(
    error_by_demand.index,
    error_by_demand.values
)

plt.xlabel("Actual demand level")
plt.ylabel("Mean absolute forecast error (%)")
plt.title("Forecast Inaccuracy Across Demand Levels")

plt.grid(axis="y", alpha=0.25)
plt.tight_layout()
plt.show()


# --------------------------------------------------
# 3. COMPACT SUMMARY TABLE
# --------------------------------------------------

weekday_labels = [
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday"
]

month_labels = [
    "Jan", "Feb", "Mar", "Apr", "May", "Jun",
    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"
]


error_by_weekday = (
    df.groupby("Weekday_Num")["APE"]
    .mean()
    .reindex(range(7))
)

error_by_month = (
    df.groupby("Month_Num")["APE"]
    .mean()
    .reindex(range(1, 13))
)


summary_table = pd.DataFrame({
    "Dimension": [
        "Day of week",
        "Day of week",
        "Month",
        "Month",
        "Demand level",
        "Demand level"
    ],
    "Category": [
        weekday_labels[error_by_weekday.values.argmin()],
        weekday_labels[error_by_weekday.values.argmax()],
        month_labels[error_by_month.values.argmin()],
        month_labels[error_by_month.values.argmax()],
        error_by_demand.index[error_by_demand.values.argmin()],
        error_by_demand.index[error_by_demand.values.argmax()]
    ],
    "Mean_APE": [
        error_by_weekday.min(),
        error_by_weekday.max(),
        error_by_month.min(),
        error_by_month.max(),
        error_by_demand.min(),
        error_by_demand.max()
    ],
    "Pattern": [
        "Lowest",
        "Highest",
        "Lowest",
        "Highest",
        "Lowest",
        "Highest"
    ]
})

print("\n--- SUMMARY OF OTHER FORECAST INACCURACY PATTERNS ---")

print(
    summary_table.round({"Mean_APE": 2}).to_string(index=False)
)

# This analysis examined when GB electricity demand forecasts were less accurate by comparing absolute percentage error across time of day, 
# day of week, month, and demand level. Forecast inaccuracy varied substantially throughout the day, increasing from around 1.7–2.0% during 
# the early morning to a peak of 3.79% at 13:30 before declining through the afternoon and evening. Inaccuracy also varied across the week, with 
# the highest average error on Sunday (2.81%) and the lowest on Wednesday (2.36%). Across months, average error was highest in April (2.96%) and 
# March (2.94%), and lowest in November (1.99%). Demand level showed a different pattern: the lowest-demand 20% of observations had a mean error 
# of 2.74%, compared with 2.09% for the highest-demand 20%. This indicates that larger demand levels were associated with lower forecast error 
# in this dataset. Overall, forecast difficulty varied systematically with the timing and level of demand, with the strongest variation occurring 
# across the day.

### JUSTIFICATION FOR NEXT ANALYSIS
# These patterns show that forecast accuracy varies across time and demand conditions, but averages alone do not reveal what happens during the 
# largest errors. The next analysis therefore focuses on extreme forecast errors, comparing the largest 5% and 1% of errors to understand their 
# size, direction and timing. It will also examine whether these errors occur as isolated observations or cluster into episodes, and whether 
# those episodes coincide with particular demand levels or rapid changes in demand. This will help distinguish occasional large mismatches from 
# periods when forecasting difficulties persist across consecutive half-hour intervals.
