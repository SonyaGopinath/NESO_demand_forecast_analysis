import pandas as pd

# Loading raw data
df = pd.read_csv("data/demand_forecast_performance.csv")


# --------------------------------------------------
# 1. Converting the date/time columns
# --------------------------------------------------

df["Date"] = pd.to_datetime(df["Date"])
df["Datetime"] = pd.to_datetime(df["Datetime"])
df["Publish_Datetime"] = pd.to_datetime(df["Publish_Datetime"])


# --------------------------------------------------
# 2. Adding time features
# --------------------------------------------------

df["Year"] = df["Datetime"].dt.year
df["Month_Num"] = df["Datetime"].dt.month
df["Day"] = df["Datetime"].dt.day
df["Hour"] = df["Datetime"].dt.hour
df["Minute"] = df["Datetime"].dt.minute
df["Weekday"] = df["Datetime"].dt.day_name()
df["Weekday_Num"] = df["Datetime"].dt.dayofweek
df["Is_Weekend"] = df["Weekday_Num"] >= 5


# --------------------------------------------------
# 3. Calculating forecast error
# --------------------------------------------------

df["Calculated_Absolute_Error"] = (
    df["Demand_Forecast"] - df["Demand_Outturn"]
).abs()

df["Error_Difference"] = (
    df["Calculated_Absolute_Error"] - df["Absolute_Error"]
).abs()


# --------------------------------------------------
# 4. Checking unusual dates
# --------------------------------------------------

rows_per_date = df.groupby("Date").size()

print("\n--- ROWS PER DATE ---")
print(rows_per_date.value_counts().sort_index())

print("\n--- DATES WITH UNUSUAL ROW COUNTS ---")
print(rows_per_date[rows_per_date != 48])


# --------------------------------------------------
# 5. Checking error discrepancies
# --------------------------------------------------

print("\n--- ERROR DISCREPANCIES ---")
print("Rows with different error:", (df["Error_Difference"] > 0).sum())
print("Largest difference:", df["Error_Difference"].max())


# --------------------------------------------------
# 6. Removing irrelevant columns
# --------------------------------------------------

df = df.drop(columns=[
    "_id",
    "Month",
    "Day",
    "TRIAD_Avoidance_Estimate",
    "TRIAD_Avoidance_Corrected_Demand_Outturn",
    "Calculated_Absolute_Error",
    "Error_Difference"
])


# --------------------------------------------------
# 7. Removing incomplete final day
# --------------------------------------------------

df = df[df["Date"] < df["Date"].max()]


# --------------------------------------------------
# 8. Saving cleaned dataset
# --------------------------------------------------

df.to_csv("data/demand_forecast_clean.csv", index=False)


# --------------------------------------------------
# 9. Final checks
# --------------------------------------------------

print("\n--- CLEAN DATA ---")
print(df.shape)
print(df.dtypes)
print(df.head())
