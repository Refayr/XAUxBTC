import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from utilsforecast.evaluation import evaluate
from statsforecast import StatsForecast
from statsforecast.models import (
    Naive,
    SeasonalNaive,
    HistoricAverage,
    WindowAverage,
    AutoARIMA,
)

from utilsforecast.losses import mae, mse, rmse, mape
from sklearn.metrics import mean_squared_error

from utilsforecast.feature_engineering import pipeline, time_features
from functools import partial

from pandas.tseries.holiday import USFederalHolidayCalendar

from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error

# Backup dataset
print("\n" + "=" * 70)
print("Generate Real Dataset")
print("=" * 70)
import yfinance as yf

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

##### End of dataset backup


# Load data
print("\n" + "=" * 70)
print("Generate Normalized Dataset")
print("=" * 70)
df_norm = pd.read_csv("20251201_timeseries.csv")

# Keep tickers only have correlation > 80%
# tickers=['XAU', 'TRX', 'BTC', 'BNB', 'LEO', 'GT', 'SUN', 'OKB', 'XRP']
# df=df[df['ticker'].isin(tickers)]

# Define the time we need
# df = df[(df['date'] >= '2023-01-01') & (df['date'] <= '2025-06-01')]




df_au_norm = df_norm[["ds"]]
df_au_norm["unique_id"] = "XAU"
df_au_norm["y"] = df_norm["XAU"]
df_au_norm = df_au_norm.drop_duplicates(subset=["ds"])
df_au_norm = df_au_norm.reset_index(drop=True)


print(df_au_norm.head())
df_au_norm["ds"] = pd.to_datetime(df_au_norm["ds"])
print("Normalized gold daataset generated -- df_au_norm")


# Set forecast horizon
horizon = 30

models = [
    Naive(),  # Always predicts the last observed value
    HistoricAverage(),  # Predicts the average of all past values
    WindowAverage(window_size=7),  # Predicts the average of the last 7 days
    SeasonalNaive(
        season_length=7
    ),  # Predicts the value from the same weekday last week
]

# 1) Build TEST set: the last 30 days
test = df_au.groupby("unique_id").tail(horizon)

# 2) Build TRAIN set: everything before the last 30 days
train = df_au.drop(test.index).reset_index(drop=True)

print(f"Train set size: {len(train)} days")
print(f"Test set size: {len(test)} days")
print(f"Train date range: {train['ds'].min()} to {train['ds'].max()}")
print(f"Test date range: {test['ds'].min()} to {test['ds'].max()}")

# 3) Initialize and fit the Naive model on TRAIN only
sf = StatsForecast(
    models=models,
    freq="D",  # Daily frequency
    n_jobs=-1,
)

sf.fit(df=train)

# 4) Forecast the next 30 days
preds = sf.predict(h=horizon)

# 5) Merge true test values with predictions
eval_df = pd.merge(test, preds, how="left", on=["ds", "unique_id"])


# Define ARIMA models
arima_models = [
    AutoARIMA(seasonal=False, alias="ARIMA"),
    AutoARIMA(season_length=7, alias="SARIMA"),
]

# Create forecasting engine for ARIMA
sf_arima = StatsForecast(models=arima_models, freq="D")
sf_arima.fit(df=train)

print("ARIMA models fitted!")

# Forecast with ARIMA models
arima_preds = sf_arima.predict(h=horizon)

print("ARIMA predictions completed!")
print("ARIMA columns:", arima_preds.columns.tolist())

# Merge ARIMA predictions with evaluation dataframe
arima_eval_df = pd.merge(arima_preds, eval_df, how="inner", on=["ds", "unique_id"])

print("\nFinal evaluation dataframe columns:")
print(arima_eval_df.columns.tolist())


# 6) Evaluate metrics
metrics = [mae, mse, rmse, mape]

evaluation = evaluate(
    eval_df,
    metrics=metrics,
)

evaluation_summary = (
    evaluation.drop(["unique_id"], axis=1).groupby("metric").mean().reset_index()
)

print("\n=== Evaluation Metrics ===")
print(evaluation_summary)

# Define all model names
all_model_names = [
    "Naive",
    "HistoricAverage",
    "WindowAverage",
    "SeasonalNaive",
    "ARIMA",
    "SARIMA",
]
colors = ["red", "orange", "purple", "brown", "darkblue", "darkgreen"]
linestyles = ["--", "-.", ":", "--", "-", "-"]
markers = ["x", "s", "^", "d", "o", "v"]

# Plot 1: Full Timeline (Train + Test)
plt.figure(figsize=(16, 8))

# Plot training data
plt.plot(train["ds"], train["y"], label="Train", color="blue", linewidth=2, alpha=0.6)

# Plot actual test data
plt.plot(
    test["ds"],
    test["y"],
    label="Actual Test",
    color="black",
    linewidth=3,
    marker="o",
    markersize=6,
    zorder=10,
)

# Plot all model predictions
for i, model_name in enumerate(all_model_names):
    if model_name in arima_eval_df.columns:
        plt.plot(
            arima_eval_df["ds"],
            arima_eval_df[model_name],
            label=model_name,
            color=colors[i],
            linewidth=2,
            linestyle=linestyles[i],
            marker=markers[i],
            markersize=5,
            alpha=0.8,
        )

plt.xlabel("Date", fontsize=13)
plt.ylabel("Gold Price (xAu)", fontsize=13)
plt.title(
    "Gold Price Prediction - All Models (Baseline + ARIMA)",
    fontsize=15,
    fontweight="bold",
)
plt.legend(fontsize=10, loc="best", ncol=2)
plt.grid(True, alpha=0.3)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


#######################################################
################## Cross Validation #################################
# Define all models
models = [
    Naive(),
    HistoricAverage(),
    WindowAverage(window_size=7),
    SeasonalNaive(season_length=7),
    AutoARIMA(seasonal=False, alias="ARIMA"),
    AutoARIMA(season_length=7, alias="SARIMA"),
]

# Fit models on train set
sf_split = StatsForecast(models=models, freq="D", n_jobs=-1)
sf_split.fit(df=train)

# Predict
split_preds = sf_split.predict(h=horizon)

# Evaluate
split_eval_df = pd.merge(test, split_preds, how="left", on=["ds", "unique_id"])

metrics = [mae, mse, rmse, mape]

split_evaluation = evaluate(
    split_eval_df,
    metrics=metrics,
)

split_summary = (
    split_evaluation.drop(["unique_id"], axis=1).groupby("metric").mean().reset_index()
)

print("\n=== Simple Split - Evaluation Summary ===")
print(split_summary)

# Reshape for model comparison
split_model_pivot = split_evaluation.melt(
    id_vars=["unique_id", "metric"], var_name="model", value_name="value"
).pivot_table(index="model", columns="metric", values="value")

print("\n=== Simple Split - Model Comparison ===")
print(split_model_pivot)

# ========== PART 2: TIME SERIES CROSS-VALIDATION ==========
print("\n" + "=" * 70)
print("PART 2: TIME SERIES CROSS-VALIDATION APPROACH")
print("=" * 70)

# Create StatsForecast instance for CV
sf_cv = StatsForecast(models=models, freq="D", n_jobs=-1)

# Apply time-series cross-validation
#     - h = 30 → each test window covers 30 days (forecast horizon)
#     - n_windows = 5 → number of rolling evaluation windows
#     - step_size = 30 → after each CV step, move forward 30 days
#     - refit = True → retrain models at each step (more realistic evaluation)

n_windows = 5
step_size = horizon

print("\nCross-Validation Configuration:")
print(f"  - Horizon (h): {horizon} days")
print(f"  - Number of windows: {n_windows}")
print(f"  - Step size: {step_size} days")
print("  - Refit models: True")
print(f"  - Total evaluations: {horizon * n_windows} days")

print("\nRunning cross-validation... (this may take a few minutes)")

cv_df = sf_cv.cross_validation(
    h=horizon, df=df_au, n_windows=n_windows, step_size=step_size, refit=True
)

print(f"Cross-validation completed! Generated {len(cv_df)} predictions.")
print(f"CV dataframe columns: {cv_df.columns.tolist()}")

# Evaluate CV results
cv_eval = evaluate(
    cv_df.drop(columns=["cutoff"]),  # cutoff not needed for metrics
    metrics=metrics,
)

cv_summary = (
    cv_eval.drop(columns=["unique_id"]).groupby("metric", as_index=False).mean()
)

print("\n=== Cross-Validation - Evaluation Summary ===")
print(cv_summary)

# Reshape CV results for model comparison
cv_model_pivot = cv_eval.melt(
    id_vars=["unique_id", "metric"], var_name="model", value_name="value"
).pivot_table(index="model", columns="metric", values="value")

print("\n=== Cross-Validation - Model Comparison ===")
print(cv_model_pivot)

# ========== PART 3: COMPARE SIMPLE SPLIT vs CV ==========
print("\n" + "=" * 70)
print("PART 3: COMPARISON - SIMPLE SPLIT vs CROSS-VALIDATION")
print("=" * 70)

# Combine results for comparison
comparison_data = []

for model in split_model_pivot.index:
    for metric in ["mae", "rmse"]:
        split_val = (
            split_model_pivot.loc[model, metric]
            if model in split_model_pivot.index
            else np.nan
        )
        cv_val = (
            cv_model_pivot.loc[model, metric]
            if model in cv_model_pivot.index
            else np.nan
        )

        comparison_data.append(
            {
                "Model": model,
                "Metric": metric.upper(),
                "Simple Split": split_val,
                "Cross-Validation": cv_val,
                "Difference": cv_val - split_val,
            }
        )

comparison_df = pd.DataFrame(comparison_data)

print("\n=== MAE and RMSE Comparison ===")
for metric in ["MAE", "RMSE"]:
    print(f"\n{metric}:")
    metric_comp = comparison_df[comparison_df["Metric"] == metric]
    print(metric_comp.to_string(index=False))

# ========== PART 4: VISUALIZATIONS ==========
print("\n" + "=" * 70)
print("PART 4: CREATING VISUALIZATIONS")
print("=" * 70)

model_names = [
    "Naive",
    "HistoricAverage",
    "WindowAverage",
    "SeasonalNaive",
    "ARIMA",
    "SARIMA",
]
colors = ["red", "orange", "purple", "brown", "darkblue", "darkgreen"]

# Plot 1: Simple Split - Test Period
fig, ax = plt.subplots(figsize=(15, 7))

ax.plot(
    test["ds"],
    test["y"],
    label="Actual",
    color="black",
    linewidth=3,
    marker="o",
    markersize=7,
    zorder=10,
)

for i, model in enumerate(model_names):
    if model in split_eval_df.columns:
        ax.plot(
            split_eval_df["ds"],
            split_eval_df[model],
            label=model,
            color=colors[i],
            linewidth=2,
            linestyle="--",
            marker="x",
            markersize=5,
            alpha=0.8,
        )

ax.set_xlabel("Date", fontsize=12)
ax.set_ylabel("Gold Price (xAu)", fontsize=12)
ax.set_title(
    "Simple Train/Test Split - Last 30 Days Prediction", fontsize=14, fontweight="bold"
)
ax.legend(fontsize=10, loc="best", ncol=2)
ax.grid(True, alpha=0.3)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Plot 2: Cross-Validation - All Windows
fig, ax = plt.subplots(figsize=(16, 8))

# Plot actual values
ax.plot(df_au["ds"], df_au["y"], label="Actual", color="black", linewidth=2, alpha=0.7)

# Plot CV predictions with different colors per window
unique_cutoffs = cv_df["cutoff"].unique()
print(f"\nCross-validation cutoff dates: {unique_cutoffs}")

# Plot one model's CV predictions to show the rolling windows
model_to_show = "ARIMA"  # Choose one model to visualize CV windows
if model_to_show in cv_df.columns:
    for i, cutoff in enumerate(unique_cutoffs):
        window_data = cv_df[cv_df["cutoff"] == cutoff]
        ax.plot(
            window_data["ds"],
            window_data[model_to_show],
            color="red",
            linewidth=1.5,
            linestyle="--",
            alpha=0.6,
            marker="o",
            markersize=3,
        )

        # Mark the cutoff point
        ax.axvline(x=cutoff, color="gray", linestyle=":", alpha=0.5, linewidth=1)

ax.set_xlabel("Date", fontsize=12)
ax.set_ylabel("Gold Price (xAu)", fontsize=12)
ax.set_title(
    f"Time Series Cross-Validation - Rolling Windows ({model_to_show} predictions)",
    fontsize=14,
    fontweight="bold",
)
ax.legend(fontsize=10, loc="best")
ax.grid(True, alpha=0.3)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Plot 3: Model Performance Comparison - Split vs CV
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# MAE Comparison
ax1 = axes[0]
mae_split = split_model_pivot["mae"].sort_index()
mae_cv = cv_model_pivot["mae"].sort_index()

x = np.arange(len(mae_split))
width = 0.35

bars1 = ax1.bar(
    x - width / 2, mae_split.values, width, label="Simple Split", color="skyblue"
)
bars2 = ax1.bar(
    x + width / 2, mae_cv.values, width, label="Cross-Validation", color="lightcoral"
)

ax1.set_xlabel("Models", fontsize=11)
ax1.set_ylabel("Mean Absolute Error (MAE)", fontsize=11)
ax1.set_title("MAE: Simple Split vs Cross-Validation", fontsize=13, fontweight="bold")
ax1.set_xticks(x)
ax1.set_xticklabels(mae_split.index, rotation=45, ha="right")
ax1.legend()
ax1.grid(True, alpha=0.3, axis="y")

# Add value labels
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax1.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{height:.2f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

# RMSE Comparison
ax2 = axes[1]
rmse_split = split_model_pivot["rmse"].sort_index()
rmse_cv = cv_model_pivot["rmse"].sort_index()

bars3 = ax2.bar(
    x - width / 2, rmse_split.values, width, label="Simple Split", color="skyblue"
)
bars4 = ax2.bar(
    x + width / 2, rmse_cv.values, width, label="Cross-Validation", color="lightcoral"
)

ax2.set_xlabel("Models", fontsize=11)
ax2.set_ylabel("Root Mean Squared Error (RMSE)", fontsize=11)
ax2.set_title("RMSE: Simple Split vs Cross-Validation", fontsize=13, fontweight="bold")
ax2.set_xticks(x)
ax2.set_xticklabels(rmse_split.index, rotation=45, ha="right")
ax2.legend()
ax2.grid(True, alpha=0.3, axis="y")

# Add value labels
for bars in [bars3, bars4]:
    for bar in bars:
        height = bar.get_height()
        ax2.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{height:.2f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

plt.tight_layout()
plt.show()

# Plot 4: Best Models Comparison
fig, ax = plt.subplots(figsize=(10, 6))

best_models_split = split_model_pivot["mae"].sort_values().head(3)
best_models_cv = cv_model_pivot["mae"].sort_values().head(3)

print("\n=== Top 3 Models by MAE ===")
print("\nSimple Split:")
for model, mae_val in best_models_split.items():
    print(f"  {model}: {mae_val:.4f}")

print("\nCross-Validation:")
for model, mae_val in best_models_cv.items():
    print(f"  {model}: {mae_val:.4f}")

# Create ranking plot
models_to_compare = list(set(best_models_split.index) | set(best_models_cv.index))
x_pos = np.arange(len(models_to_compare))

split_vals = [
    split_model_pivot.loc[m, "mae"] if m in split_model_pivot.index else 0
    for m in models_to_compare
]
cv_vals = [
    cv_model_pivot.loc[m, "mae"] if m in cv_model_pivot.index else 0
    for m in models_to_compare
]

bars1 = ax.bar(x_pos - 0.2, split_vals, 0.4, label="Simple Split", color="#2E86AB")
bars2 = ax.bar(x_pos + 0.2, cv_vals, 0.4, label="Cross-Validation", color="#A23B72")

ax.set_xlabel("Models", fontsize=12)
ax.set_ylabel("Mean Absolute Error (MAE)", fontsize=12)
ax.set_title("Top Performing Models - MAE Comparison", fontsize=14, fontweight="bold")
ax.set_xticks(x_pos)
ax.set_xticklabels(models_to_compare, rotation=45, ha="right")
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3, axis="y")

plt.tight_layout()
plt.show()

# ========== PART 5: FINAL SUMMARY ==========
print("\n" + "=" * 70)
print("PART 5: FINAL SUMMARY")
print("=" * 70)

print("\n=== Best Model (by MAE) ===")
best_split = split_model_pivot["mae"].idxmin()
best_cv = cv_model_pivot["mae"].idxmin()

print(
    f"\nSimple Split: {best_split} (MAE: {split_model_pivot.loc[best_split, 'mae']:.4f})"
)
print(f"Cross-Validation: {best_cv} (MAE: {cv_model_pivot.loc[best_cv, 'mae']:.4f})")

print("\n=== Key Insights ===")
print("1. Cross-validation provides more robust performance estimates")
print("2. CV evaluates models across multiple time periods, not just one")
print("3. Models may perform differently in CV vs simple split")
print("4. CV better reflects real-world forecasting performance")

print("\n" + "=" * 70)
print("ANALYSIS COMPLETE!")
print("=" * 70)



# Define time features: day of week, week of year, month of year
time_feature_funcs = [partial(time_features, features=["day", "week", "month"])]

# Build feature matrices
# This creates:
# - exog_df: features for historical data (training)
# - futr_exog_df: features for future horizon (forecasting)
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

# 1) Build TEST set: the last 30 days
test_exogf = df_exogf.groupby("unique_id").tail(horizon)

# 2) Build TRAIN set: everything before the last 30 days
train_exogf = df_exogf.drop(test.index).reset_index(drop=True)


# Define all models
models_exogf = [
    AutoARIMA(seasonal=False, alias="ARIMA"),
    AutoARIMA(season_length=7, alias="SARIMA"),
]

# Fit models on train set
sf_split = StatsForecast(models=models_exogf, freq="D", n_jobs=-1)
sf_split.fit(df=train_exogf)

# Predict
test_exogf_X = test_exogf.drop(columns=["y"])
split_preds = sf_split.predict(h=horizon, X_df=test_exogf_X)

# Evaluate
split_eval_df = pd.merge(test_exogf, split_preds, how="left", on=["ds", "unique_id"])
eval_columns = ["ds", "unique_id", "y", "ARIMA", "SARIMA"]
split_eval_df_clean = split_eval_df[eval_columns]

metrics = [mae, mse, rmse, mape]

split_evaluation = evaluate(
    split_eval_df_clean,  # corrected from split_eval_df
    metrics=metrics,
)

split_summary = (
    split_evaluation.drop(["unique_id"], axis=1).groupby("metric").mean().reset_index()
)

print("\n=== Simple Split - Evaluation Summary ===")
print(split_summary)

# Reshape for model comparison
split_model_pivot = split_evaluation.melt(
    id_vars=["unique_id", "metric"], var_name="model", value_name="value"
).pivot_table(index="model", columns="metric", values="value")

print("\n=== Simple Split - Model Comparison ===")
print(split_model_pivot)

# ========== PART 2: TIME SERIES CROSS-VALIDATION ==========
print("\n" + "=" * 70)
print("PART 2: TIME SERIES CROSS-VALIDATION APPROACH")
print("=" * 70)

# Create StatsForecast instance for CV
sf_cv = StatsForecast(models=models_exogf, freq="D", n_jobs=-1)

# Apply time-series cross-validation

n_windows = 5
step_size = horizon

print("\nCross-Validation Configuration:")
print(f"  - Horizon (h): {horizon} days")
print(f"  - Number of windows: {n_windows}")
print(f"  - Step size: {step_size} days")
print("  - Refit models: True")
print(f"  - Total evaluations: {horizon * n_windows} days")

print("\nRunning cross-validation... (this may take a few minutes)")

cv_df = sf_cv.cross_validation(
    h=horizon, df=df_exogf, n_windows=n_windows, step_size=step_size, refit=True
)

print(f"Cross-validation completed! Generated {len(cv_df)} predictions.")
print(f"CV dataframe columns: {cv_df.columns.tolist()}")

# Evaluate CV results
cv_eval = evaluate(
    cv_df.drop(columns=["cutoff"]),  # cutoff not needed for metrics
    metrics=metrics,
)

cv_summary = (
    cv_eval.drop(columns=["unique_id"]).groupby("metric", as_index=False).mean()
)

print("\n=== Cross-Validation - Evaluation Summary ===")
print(cv_summary)

# Reshape CV results for model comparison
cv_model_pivot = cv_eval.melt(
    id_vars=["unique_id", "metric"], var_name="model", value_name="value"
).pivot_table(index="model", columns="metric", values="value")

print("\n=== Cross-Validation - Model Comparison ===")
print(cv_model_pivot)


# ========== PART 3: VISUALIZATIONS ==========
print("\n" + "=" * 70)
print("PART 4: CREATING VISUALIZATIONS")
print("=" * 70)

model_names = ["ARIMA", "SARIMA"]
colors = ["red", "orange"]


# Plot 2: Simple Split - Test Period
fig, ax = plt.subplots(figsize=(15, 7))

ax.plot(
    test["ds"],
    test["y"],
    label="Actual",
    color="black",
    linewidth=3,
    marker="o",
    markersize=7,
    zorder=10,
)

for i, model in enumerate(model_names):
    if model in split_eval_df.columns:
        ax.plot(
            split_eval_df["ds"],
            split_eval_df[model],
            label=model,
            color=colors[i],
            linewidth=2,
            linestyle="--",
            marker="x",
            markersize=5,
            alpha=0.8,
        )

ax.set_xlabel("Date", fontsize=12)
ax.set_ylabel("Gold Price (xAu)", fontsize=12)
ax.set_title(
    "Simple Train/Test Split - Last 30 Days Prediction", fontsize=14, fontweight="bold"
)
ax.legend(fontsize=10, loc="best", ncol=2)
ax.grid(True, alpha=0.3)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

"""
SIMPLE LINEAR REGRESSION 
"""


# ============================================================================
# STEP 1: CREATE FEATURES FOR LINEAR REGRESSION
# ============================================================================

print("\n" + "=" * 70)
print("LINEAR REGRESSION FORECASTING")
print("=" * 70)

# Create time features
print("\nCreating time features...")
df_lr = df_exogf.copy()
df_lr["ds"] = pd.to_datetime(df_lr["ds"])


# Cyclical features (better for periodic patterns)
df_lr["day_sin"] = np.sin(2 * np.pi * df_lr["day"] / 30)
df_lr["day_cos"] = np.cos(2 * np.pi * df_lr["day"] / 30)
df_lr["month_sin"] = np.sin(2 * np.pi * df_lr["month"] / 12)
df_lr["month_cos"] = np.cos(2 * np.pi * df_lr["month"] / 12)

print("✓ Time features created")

# Create lag features (if you want to use past values)
print("Creating lag features...")
df_lr = df_lr.sort_values(["unique_id", "ds"])

for lag in [1, 7, 14]:
    df_lr[f"y_lag_{lag}"] = df_lr.groupby("unique_id")["y"].shift(lag)

print("✓ Lag features created")

# Drop rows with NaN from lagging
df_lr = df_lr.dropna()
print(f"✓ Data shape after feature engineering: {df_lr.shape}")

print(df_lr.head())

# ============================================================================
# STEP 2: PREPARE TRAIN/TEST SPLITS
# ============================================================================

print("\nPreparing train/test splits...")

# Get the same train/test indices as before
train_lr = df_lr.groupby("unique_id").tail(horizon)
test_lr = df_lr.drop(test.index).reset_index(drop=True)

print(f"Train shape: {train_lr.shape}")
print(f"Test shape: {test_lr.shape}")

# Define feature columns (exclude target and identifiers)
exclude_cols = ["ds", "unique_id", "y"]
feature_cols = [col for col in train_lr.columns if col not in exclude_cols]

print(f"Number of features: {len(feature_cols)}")


# Prepare X and y
X_train = train_lr[feature_cols]
y_train = train_lr["y"]
X_test = test_lr[feature_cols]
y_test = test_lr["y"]


# ============================================================================
# STEP 3: TRAIN LINEAR REGRESSION MODELS
# ============================================================================

# Scale features (important for regression)
print("\nScaling features...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train Linear Regression
print("\nTraining Linear Regression...")
lr_model = LinearRegression()
lr_model.fit(X_train_scaled, y_train)
lr_preds = lr_model.predict(X_test_scaled)

# Train Ridge Regression (with regularization)
print("Training Ridge Regression...")
ridge_model = Ridge(alpha=1.0)
ridge_model.fit(X_train_scaled, y_train)
ridge_preds = ridge_model.predict(X_test_scaled)

print("✓ Models trained successfully")

# ============================================================================
# STEP 4: CALCULATE METRICS
# ============================================================================

print("\n" + "=" * 70)
print("LINEAR REGRESSION - METRICS")
print("=" * 70)

# Linear Regression metrics
lr_mae = mean_absolute_error(y_test, lr_preds)
lr_rmse = np.sqrt(mean_squared_error(y_test, lr_preds))
lr_mape = mean_absolute_percentage_error(y_test, lr_preds) * 100

print("\nLinear Regression:")
print(f"  MAE:  {lr_mae:.4f}")
print(f"  RMSE: {lr_rmse:.4f}")
print(f"  MAPE: {lr_mape:.2f}%")

# Ridge Regression metrics
ridge_mae = mean_absolute_error(y_test, ridge_preds)
ridge_rmse = np.sqrt(mean_squared_error(y_test, ridge_preds))
ridge_mape = mean_absolute_percentage_error(y_test, ridge_preds) * 100

print("\nRidge Regression:")
print(f"  MAE:  {ridge_mae:.4f}")
print(f"  RMSE: {ridge_rmse:.4f}")
print(f"  MAPE: {ridge_mape:.2f}%")

print("=" * 70)

# ============================================================================
# STEP 5: ADD PREDICTIONS TO EVALUATION DATAFRAME
# ============================================================================

# Add predictions to test dataframe
test_lr_eval = test_lr.copy()
test_lr_eval["LinearRegression"] = lr_preds
test_lr_eval["Ridge"] = ridge_preds

# ============================================================================
# STEP 6: PLOT RESULTS
# ============================================================================

print("\nGenerating plots...")

# Create figure with subplots
fig, axes = plt.subplots(3, 1, figsize=(18, 14))

# ===== Plot 1: Full Timeline =====
ax1 = axes[0]

# Plot training data
ax1.plot(
    train_lr["ds"], train_lr["y"], color="blue", linewidth=2, alpha=0.6, label="Train"
)

# Plot test actual
ax1.plot(
    test_lr["ds"],
    test_lr["y"],
    color="black",
    linewidth=3,
    marker="o",
    markersize=7,
    label="Actual (Test)",
    zorder=5,
)

# Plot Linear Regression predictions
ax1.plot(
    test_lr_eval["ds"],
    test_lr_eval["LinearRegression"],
    color="red",
    linewidth=2.5,
    linestyle="--",
    marker="x",
    markersize=6,
    label="Linear Regression",
    alpha=0.8,
    zorder=4,
)

# Plot Ridge predictions
ax1.plot(
    test_lr_eval["ds"],
    test_lr_eval["Ridge"],
    color="orange",
    linewidth=2.5,
    linestyle="--",
    marker="^",
    markersize=5,
    label="Ridge Regression",
    alpha=0.8,
    zorder=4,
)

# Add vertical line at train/test split
if len(train_lr) > 0:
    ax1.axvline(
        x=train_lr["ds"].max(),
        color="red",
        linestyle=":",
        linewidth=2,
        alpha=0.6,
        label="Train/Test Split",
    )

ax1.set_xlabel("Date", fontsize=12, fontweight="bold")
ax1.set_ylabel("Gold Price", fontsize=12, fontweight="bold")
ax1.set_title(
    "Linear Regression Models - Full Timeline", fontsize=14, fontweight="bold"
)
ax1.legend(loc="best", fontsize=11, ncol=2)
ax1.grid(True, alpha=0.3)
plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha="right")

# ===== Plot 2: Test Period Zoomed =====
ax2 = axes[1]

ax2.plot(
    test_lr_eval["ds"],
    test_lr_eval["y"],
    color="black",
    linewidth=3.5,
    marker="o",
    markersize=9,
    label="Actual",
    zorder=5,
)

ax2.plot(
    test_lr_eval["ds"],
    test_lr_eval["LinearRegression"],
    color="red",
    linewidth=3,
    linestyle="--",
    marker="x",
    markersize=8,
    label="Linear Regression",
    alpha=0.8,
    zorder=4,
)

ax2.plot(
    test_lr_eval["ds"],
    test_lr_eval["Ridge"],
    color="orange",
    linewidth=3,
    linestyle="--",
    marker="^",
    markersize=7,
    label="Ridge",
    alpha=0.8,
    zorder=4,
)

ax2.set_xlabel("Date", fontsize=12, fontweight="bold")
ax2.set_ylabel("Gold Price", fontsize=12, fontweight="bold")
ax2.set_title("Test Period (Zoomed)", fontsize=14, fontweight="bold")
ax2.legend(loc="best", fontsize=11)
ax2.grid(True, alpha=0.3)
plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha="right")

# ===== Plot 3: Prediction Errors =====
ax3 = axes[2]

lr_errors = test_lr_eval["y"] - test_lr_eval["LinearRegression"]
ridge_errors = test_lr_eval["y"] - test_lr_eval["Ridge"]

ax3.plot(
    test_lr_eval["ds"],
    lr_errors,
    color="red",
    linewidth=2.5,
    marker="o",
    markersize=5,
    label="Linear Reg Error",
    alpha=0.7,
)

ax3.plot(
    test_lr_eval["ds"],
    ridge_errors,
    color="orange",
    linewidth=2.5,
    marker="^",
    markersize=5,
    label="Ridge Error",
    alpha=0.7,
)

ax3.axhline(y=0, color="black", linestyle="--", linewidth=1, alpha=0.5)

ax3.set_xlabel("Date", fontsize=12, fontweight="bold")
ax3.set_ylabel("Prediction Error (Actual - Predicted)", fontsize=12, fontweight="bold")
ax3.set_title("Prediction Errors Over Time", fontsize=14, fontweight="bold")
ax3.legend(loc="best", fontsize=11)
ax3.grid(True, alpha=0.3)
plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45, ha="right")

# Overall title
fig.suptitle(
    "Linear Regression Forecasting Results", fontsize=16, fontweight="bold", y=0.995
)

plt.tight_layout()
plt.savefig("linear_regression_predictions.png", dpi=300, bbox_inches="tight")
print("✓ Plot saved as 'linear_regression_predictions.png'")
plt.show()

# ============================================================================
# STEP 7: FEATURE IMPORTANCE
# ============================================================================

print("\n" + "=" * 70)
print("TOP 10 MOST IMPORTANT FEATURES")
print("=" * 70)

# Get feature importance (coefficients)
feature_importance = pd.DataFrame(
    {"Feature": feature_cols, "Coefficient": lr_model.coef_}
)
feature_importance["Abs_Coefficient"] = feature_importance["Coefficient"].abs()
feature_importance = feature_importance.sort_values("Abs_Coefficient", ascending=False)

print(feature_importance.head(10).to_string(index=False))
print("=" * 70)

# Plot feature importance
plt.figure(figsize=(12, 8))
top_features = feature_importance.head(15)

plt.barh(
    range(len(top_features)),
    top_features["Abs_Coefficient"],
    color=["red" if c < 0 else "green" for c in top_features["Coefficient"]],
    alpha=0.7,
    edgecolor="black",
)
plt.yticks(range(len(top_features)), top_features["Feature"])
plt.xlabel("Absolute Coefficient", fontsize=12, fontweight="bold")
plt.title(
    "Top 15 Feature Importance (Linear Regression)", fontsize=14, fontweight="bold"
)
plt.grid(True, alpha=0.3, axis="x")
plt.tight_layout()
plt.savefig("feature_importance.png", dpi=300, bbox_inches="tight")
print("\n✓ Feature importance plot saved as 'feature_importance.png'")
plt.show()

# ============================================================================
# STEP 8: METRICS COMPARISON TABLE
# ============================================================================

print("\n" + "=" * 70)
print("METRICS COMPARISON: ALL MODELS")
print("=" * 70)

# Create comparison dataframe
comparison_data = {
    "Model": ["Linear Regression", "Ridge Regression"],
    "MAE": [lr_mae, ridge_mae],
    "RMSE": [lr_rmse, ridge_rmse],
    "MAPE (%)": [lr_mape, ridge_mape],
}

# Add ARIMA/SARIMA
if "split_evaluation" in locals() or "split_evaluation" in globals():
    try:
        arima_metrics = split_evaluation[split_evaluation["metric"] == "mae"][
            "ARIMA"
        ].values[0]
        sarima_metrics = split_evaluation[split_evaluation["metric"] == "mae"][
            "SARIMA"
        ].values[0]

        comparison_data["Model"].extend(["ARIMA", "SARIMA"])
        comparison_data["MAE"].extend([arima_metrics, sarima_metrics])

        # Get RMSE for ARIMA/SARIMA
        arima_rmse = split_evaluation[split_evaluation["metric"] == "rmse"][
            "ARIMA"
        ].values[0]
        sarima_rmse = split_evaluation[split_evaluation["metric"] == "rmse"][
            "SARIMA"
        ].values[0]
        comparison_data["RMSE"].extend([arima_rmse, sarima_rmse])

        # Get MAPE
        arima_mape = split_evaluation[split_evaluation["metric"] == "mape"][
            "ARIMA"
        ].values[0]
        sarima_mape = split_evaluation[split_evaluation["metric"] == "mape"][
            "SARIMA"
        ].values[0]
        comparison_data["MAPE (%)"].extend([arima_mape, sarima_mape])
    except:
        pass

comparison_df = pd.DataFrame(comparison_data)
print(comparison_df.to_string(index=False))

# Find best model
best_model_idx = comparison_df["RMSE"].idxmin()
best_model = comparison_df.loc[best_model_idx, "Model"]
best_rmse = comparison_df.loc[best_model_idx, "RMSE"]

print("\n" + "=" * 70)
print(f"✓ BEST MODEL: {best_model}")
print(f"  RMSE: {best_rmse:.4f}")
print("=" * 70)

# ============================================================================
# DONE!
# ============================================================================

print("\n✓ Linear Regression analysis complete!")
print("\nGenerated files:")
print("  - linear_regression_predictions.png")
print("  - feature_importance.png")

