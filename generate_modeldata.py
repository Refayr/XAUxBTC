import numpy as np
import pandas as pd
import yfinance as yf
import os
from functools import partial
from pandas.tseries.holiday import USFederalHolidayCalendar
from utilsforecast.feature_engineering import pipeline, time_features
import time


# Create a datasets folder
os.makedirs('model_datasets', exist_ok=True)

print("\n" + "=" * 70)
print("Generate Real Dataset")
print("=" * 70)


ticker_list = [
    "GC=F",
    "BTC-USD",
    "TRX-USD",
    "BNB-USD",
    "LEO-USD",
    "GT-USD",
    "SUN-USD",
    "OKB-USD",
    "XRP-USD",
]

frames = []
for i in ticker_list:
 
    df = yf.download(i, start="2022-05-16", end="2024-12-09")
    df = df[["Close"]]
    base = i.split("-")[0]
    df = df.rename(columns={"Close": base})
    frames.append(df)

    print(i)  # To tell us the process is in which ticker now

result = pd.concat(frames, axis=1)

df = result

if df.columns.nlevels > 1:
    df.columns = df.columns.droplevel(0)


df = df.reset_index().melt(id_vars="Date", var_name="Ticker", value_name="Price")

# Sort by date and ticker
df = df.sort_values(["Date", "Ticker"]).reset_index(drop=True)

# Remove na
# df = df.dropna()

# Replace GC=F with xAu
df["Ticker"] = df["Ticker"].replace("GC=F", "xAU")

# Remove -USD from all tickers
df["Ticker"] = df["Ticker"].str.replace("-USD", "", regex=False)
print(df.head())

# Ensure date is datetime
df["Date"] = pd.to_datetime(df["Date"])

df = df.rename(columns={"Date": "ds", "Ticker": "unique_id", "Price": "y"})


df_long = df.reset_index()

print("Original long format:")
print(df_long.head())

print("Columns: ", df_long.columns.tolist())


# Step 2: Pivot to wide format (each ticker becomes a column)
df_wide = df_long.pivot(index="ds", columns="unique_id", values="y")

print("\nAfter pivoting to wide format:")
print(df_wide.head())

print('Columns:' ,df_wide.columns.tolist())


# Step 3: Reset index to make Date a regular column again
df_wide = df_wide.reset_index()


print("\nAfter cleaning column names:")
print(df_wide.head())
print('Columns:' ,df_wide.columns.tolist())


# Step 6: Get list of crypto columns (all except ds and xAU)
crypto_cols = [col for col in df_wide.columns if col not in ["ds", "xAU"]]

print(f"\nCrypto columns to add: {crypto_cols}")

df_au = df_wide[["ds", "xAU"]]
df_au["unique_id"] = "XAU"
df_au = df_au.rename(columns={"xAU": "y"})

print(df_au.head())
df_au["ds"] = pd.to_datetime(df_au["ds"])

df_au["y"] = df_au["y"].interpolate(method="linear")
print(df_au.isnull().sum())

print("Real gold dataset generated -- df_au")


try:
    df_au.to_csv('model_datasets/gold_raw.csv', index=False)
    print("✓ Saved gold_raw.csv")
except: print("⚠ Could not save gold_raw.csv")

##### End of generate real dataset




# Load data
print("\n" + "=" * 70)
print("Generate Normalized Dataset")
print("=" * 70)
df_norm = pd.read_csv("20251201_timeseries.csv")




df_au_norm = df_norm[["ds"]]
df_au_norm["unique_id"] = "XAU"
df_au_norm["y"] = df_norm["XAU"]
df_au_norm = df_au_norm.drop_duplicates(subset=["ds"])
df_au_norm = df_au_norm.reset_index(drop=True)


print(df_au_norm.head())
df_au_norm["ds"] = pd.to_datetime(df_au_norm["ds"])
print("Normalized gold daataset generated -- df_au_norm")



# Define time features: day of week, week of year, month of year
time_feature_funcs = [partial(time_features, features=["day", "week", "month"])]

# Build feature matrices
# This creates:
# - exog_df: features for historical data (training)
# - futr_exog_df: features for future horizon (forecasting)
horizon = 30
exog_df, futr_exog_df = pipeline(
    df=df_au, features=time_feature_funcs, freq="D", h=horizon
)
exog_df["months_since_start"] = (
    exog_df["ds"].dt.year - exog_df["ds"].min().year
) * 12 + (exog_df["ds"].dt.month - exog_df["ds"].min().month)



# US holidays (affect markets)
cal = USFederalHolidayCalendar()
holidays = cal.holidays(start=df["ds"].min(), end=df["ds"].max())

exog_df["is_holiday"] = exog_df["ds"].isin(holidays).astype(int)
exog_df["days_to_holiday"] = exog_df["ds"].apply(
    lambda x: (holidays[holidays > x].min() - x).days
    if len(holidays[holidays > x]) > 0
    else 365
)


print("\nTime features created:")
print(exog_df.head())
print(f"\nExogenous features shape: {exog_df.shape}")
print(f"Future exogenous features shape: {futr_exog_df.shape}")

try:
    exog_df.to_csv('model_datasets/time_features.csv', index=False)
    print("✓ Saved time_features.csv")
except: print("⚠ Could not save time_features.csv")



df_wide = df_norm.pivot(index="ds", columns="unique_id", values="y")
print(df_wide.head())

df_wide = df_wide.reset_index()

exog_df["ds"] = pd.to_datetime(exog_df["ds"])
df_wide["ds"] = pd.to_datetime(df_wide["ds"])

df_exogf = pd.merge(df_wide, exog_df, on="ds", how="right")

for col in df_exogf.columns:
    if df_exogf[col].isnull().any():
        df_exogf[col] = df_exogf[col].interpolate(method="linear")
        print(f"✓ Filled NaN in {col}")

print(df_exogf.isnull().sum())
# df_exogf = df_exogf.replace([np.inf, -np.inf], np.nan).dropna()

print(df_exogf.head())

try:
    df_exogf.to_csv('model_datasets/features_complete.csv', index=False)
    print("✓ Saved features_complete.csv (MAIN DATASET)")
except: print("⚠ Could not save features_complete.csv")




