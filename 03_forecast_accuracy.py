import pandas as pd
import matplotlib.pyplot as plt

# ==================================================
# 1. LOAD CLEANED DATA
# ==================================================

df = pd.read_csv("data/demand_forecast_clean.csv")

df["Datetime"] = pd.to_datetime(df["Datetime"])
df["Date"] = pd.to_datetime(df["Date"])


# ==================================================
# 2. CALCULATE FORECAST ERROR
# ==================================================

# Positive error = forecast was higher than actual demand
# Negative error = forecast was lower than actual demand

df["Forecast_Error"] = (
    df["Demand_Forecast"] - df["Demand_Outturn"]
)

# Absolute error in MW
df["Absolute_Error_Calc"] = (
    df["Forecast_Error"].abs()
)

# Absolute percentage error
df["APE_Calc"] = (
    df["Absolute_Error_Calc"]
    / df["Demand_Outturn"]
) * 100


# ==================================================
# 3. OVERALL FORECAST ACCURACY
# ==================================================

print("\n" + "=" * 50)
print("FORECAST ACCURACY")
print("=" * 50)

print("\n--- FORECAST ERROR ---")

print(
    f"Mean forecast error: "
    f"{df['Forecast_Error'].mean():.2f} MW"
)

print(
    f"Median forecast error: "
    f"{df['Forecast_Error'].median():.2f} MW"
)

print(
    f"Mean absolute error: "
    f"{df['Absolute_Error_Calc'].mean():.2f} MW"
)

print(
    f"Median absolute error: "
    f"{df['Absolute_Error_Calc'].median():.2f} MW"
)

print(
    f"Standard deviation: "
    f"{df['Forecast_Error'].std():.2f} MW"
)


# ==================================================
# 4. ERROR PERCENTILES
# ==================================================

print("\n--- ABSOLUTE ERROR PERCENTILES ---")

error_percentiles = (
    df["Absolute_Error_Calc"]
    .quantile([0.50, 0.75, 0.90, 0.95, 0.99])
)

for percentile, value in error_percentiles.items():
    print(
        f"{percentile * 100:.0f}th percentile: "
        f"{value:.0f} MW"
    )


print("\n--- ABSOLUTE PERCENTAGE ERROR PERCENTILES ---")

ape_percentiles = (
    df["APE_Calc"]
    .quantile([0.50, 0.75, 0.90, 0.95, 0.99])
)

for percentile, value in ape_percentiles.items():
    print(
        f"{percentile * 100:.0f}th percentile: "
        f"{value:.2f}%"
    )


# ==================================================
# 5. FORECAST BIAS
# ==================================================

print("\n--- FORECAST BIAS ---")

positive_errors = (
    df["Forecast_Error"] > 0
).sum()

negative_errors = (
    df["Forecast_Error"] < 0
).sum()

zero_errors = (
    df["Forecast_Error"] == 0
).sum()

print(
    f"Forecast higher than actual: "
    f"{positive_errors:,} observations "
    f"({positive_errors / len(df) * 100:.2f}%)"
)

print(
    f"Forecast lower than actual: "
    f"{negative_errors:,} observations "
    f"({negative_errors / len(df) * 100:.2f}%)"
)

print(
    f"Forecast exactly equal to actual: "
    f"{zero_errors:,} observations "
    f"({zero_errors / len(df) * 100:.2f}%)"
)



# ==================================================
# 6. DISTRIBUTION OF FORECAST ERROR
# ==================================================

plt.figure(figsize=(10, 5.5))

plt.hist(
    df["Forecast_Error"],
    bins=60
)

plt.axvline(
    0,
    linestyle="--",
    linewidth=1.5
)

plt.xlabel("Forecast error (MW)")
plt.ylabel("Number of half-hour observations")
plt.title("Distribution of Signed Forecast Errors")

plt.grid(axis="y", alpha=0.25)
plt.tight_layout()
plt.show()


# ==================================================
# 7. DISTRIBUTION OF ABSOLUTE PERCENTAGE ERROR
# ==================================================

plt.figure(figsize=(10, 5.5))

plt.hist(
    df["APE_Calc"],
    bins=50
)

plt.xlabel("Absolute forecast error (%)")
plt.ylabel("Number of half-hour observations")
plt.title("Distribution of Absolute Percentage Forecast Errors")

plt.grid(axis="y", alpha=0.25)
plt.tight_layout()
plt.show()    

## This analysis examined forecast error (forecast demand minus actual demand), absolute error (the size of the error regardless of direction), 
# absolute percentage error (error size relative to actual demand), and forecast direction (whether demand was over- or under-forecast). 
# This analysis examined how closely day-ahead electricity demand forecasts matched actual demand, using signed forecast error, absolute error, 
# absolute percentage error, and forecast direction. Across the study period, forecasts were generally close to actual demand. The median absolute 
# percentage error was 1.89%, meaning the typical forecast differed from actual demand by around 1.9%, regardless of whether demand was over- 
# or under-forecast. The error was below 3.43% for 75% of observations and below 5.39% for 90%, showing that most forecasts were relatively accurate. 
# The signed error was centred close to zero, with a mean of +14 MW. Forecasts were higher than actual demand in 50.86% of observations and lower 
# in 49.08%, showing little overall tendency to consistently over- or under-forecast. However, the distribution of absolute percentage errors 
# had a clear upper tail. The 99th percentile was 11.20%, meaning that only 1% of forecasts had a larger error, but these observations were 
# substantially less accurate than the typical forecast. This provides the basis for examining extreme forecast errors separately in the next analysis.
