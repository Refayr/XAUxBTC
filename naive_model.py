import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from utilsforecast.plotting import plot_series
from utilsforecast.evaluation import evaluate
from statsforecast import StatsForecast

from utilsforecast.losses import mae, mse, rmse, mape
from sklearn.metrics import mean_squared_error

#Backup dataset
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

ticker_list=['GC=F','BTC-USD','TRX-USD','BNB-USD','LEO-USD','GT-USD','SUN-USD','OKB-USD','XRP-USD']

frames = []
for i in ticker_list:
    df=yf.download(i,start='2022-01-04',end='2024-12-04')
    df=df[['Close']]
    base = i.split('-')[0]  
    df=df.rename(columns={'Close':base})
    frames.append(df)

    print(i)#To tell us the process is in which ticker now

result = pd.concat(frames, axis=1)

df=result

if df.columns.nlevels > 1:
    df.columns = df.columns.droplevel(0)


df = df.reset_index().melt(id_vars='Date', var_name='Ticker', value_name='Price')

# Sort by date and ticker
df = df.sort_values(['Date', 'Ticker']).reset_index(drop=True)

#Remove na
df = df.dropna()

# Replace GC=F with xAu
df['Ticker'] = df['Ticker'].replace('GC=F', 'xAU')

# Remove -USD from all tickers
df['Ticker'] = df['Ticker'].str.replace('-USD', '', regex=False)
print(df.head())


# Ensure date is datetime
df['Date'] = pd.to_datetime(df['Date'])

df = df.rename(columns={
    'Date': 'ds',           
    'Ticker': 'unique_id',  
    'Price': 'y'            
})



df_long = df.reset_index()

print("Original long format:")
print(df_long.head())
print(f"Columns: {df_long.columns.tolist()}")

# Step 2: Pivot to wide format (each ticker becomes a column)
df_wide = df_long.pivot(index='ds', columns='unique_id', values='y')

print("\nAfter pivoting to wide format:")
print(df_wide.head())
print(f"Columns: {df_wide.columns.tolist()}")

# Step 3: Reset index to make Date a regular column again
df_wide = df_wide.reset_index()


print("\nAfter cleaning column names:")
print(df_wide.head())
print(f"Columns: {df_wide.columns.tolist()}")

# Step 6: Get list of crypto columns (all except ds and xAU)
crypto_cols = [col for col in df_wide.columns if col not in ['ds', 'xAU']]

print(f"\nCrypto columns to add: {crypto_cols}")


'''
# Load data
df = pd.read_csv("dataset.csv")

# Keep tickers only have correlation > 80%
tickers=['XAU', 'TRX', 'BTC', 'BNB', 'LEO', 'GT', 'SUN', 'OKB', 'XRP']
df=df[df['ticker'].isin(tickers)]

#Define the time we need
df = df[(df['date'] >= '2023-01-01') & (df['date'] <= '2025-06-01')]

'''


from statsforecast import StatsForecast

from statsforecast.models import Naive, SeasonalNaive, HistoricAverage, WindowAverage, AutoARIMA
import matplotlib.pyplot as plt

df_au = df[df['unique_id'] == 'xAU'].copy()

# Set forecast horizon
horizon = 30

models = [
    Naive(),                    # Always predicts the last observed value
    HistoricAverage(),          # Predicts the average of all past values
    WindowAverage(window_size=7),  # Predicts the average of the last 7 days
    SeasonalNaive(season_length=7)  # Predicts the value from the same weekday last week

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
    freq='D',  # Daily frequency
    n_jobs=-1
)

sf.fit(df=train)

# 4) Forecast the next 30 days
preds = sf.predict(h=horizon)

# 5) Merge true test values with predictions
eval_df = pd.merge(test, preds, how='left', on=['ds', 'unique_id'])


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
arima_eval_df = pd.merge(arima_preds, eval_df, how='inner', on=['ds', 'unique_id'])

print("\nFinal evaluation dataframe columns:")
print(arima_eval_df.columns.tolist())


# 6) Evaluate metrics
metrics = [mae, mse, rmse, mape]

evaluation = evaluate(
    eval_df,
    metrics=metrics,
)

evaluation_summary = (
    evaluation
    .drop(['unique_id'], axis=1)
    .groupby('metric')
    .mean()
    .reset_index()
)

print("\n=== Evaluation Metrics ===")
print(evaluation_summary)

# Define all model names
all_model_names = ['Naive', 'HistoricAverage', 'WindowAverage', 'SeasonalNaive', 'ARIMA', 'SARIMA']
colors = ['red', 'orange', 'purple', 'brown', 'darkblue', 'darkgreen']
linestyles = ['--', '-.', ':', '--', '-', '-']
markers = ['x', 's', '^', 'd', 'o', 'v']

# Plot 1: Full Timeline (Train + Test)
plt.figure(figsize=(16, 8))

# Plot training data
plt.plot(train['ds'], train['y'], label='Train', color='blue', linewidth=2, alpha=0.6)

# Plot actual test data
plt.plot(test['ds'], test['y'], label='Actual Test', color='black', linewidth=3, marker='o', markersize=6, zorder=10)

# Plot all model predictions
for i, model_name in enumerate(all_model_names):
    if model_name in arima_eval_df.columns:
        plt.plot(arima_eval_df['ds'], arima_eval_df[model_name], 
                label=model_name, 
                color=colors[i], 
                linewidth=2, 
                linestyle=linestyles[i], 
                marker=markers[i], 
                markersize=5,
                alpha=0.8)

plt.xlabel('Date', fontsize=13)
plt.ylabel('Gold Price (xAu)', fontsize=13)
plt.title('Gold Price Prediction - All Models (Baseline + ARIMA)', fontsize=15, fontweight='bold')
plt.legend(fontsize=10, loc='best', ncol=2)
plt.grid(True, alpha=0.3)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Plot 2: Test Period Only (Zoomed)
plt.figure(figsize=(15, 8))

plt.plot(test['ds'], test['y'], label='Actual', color='black', linewidth=3.5, marker='o', markersize=8, zorder=10)

for i, model_name in enumerate(all_model_names):
    if model_name in arima_eval_df.columns:
        plt.plot(arima_eval_df['ds'], arima_eval_df[model_name], 
                label=model_name, 
                color=colors[i], 
                linewidth=2.5, 
                linestyle=linestyles[i], 
                marker=markers[i], 
                markersize=6,
                alpha=0.85)

plt.xlabel('Date', fontsize=13)
plt.ylabel('Gold Price (xAu)', fontsize=13)
plt.title('Gold Price - Test Period Comparison (30 Days) - All Models', fontsize=15, fontweight='bold')
plt.legend(fontsize=11, loc='best', ncol=2)
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
    AutoARIMA(season_length=7, alias="SARIMA")
]

# Fit models on train set
sf_split = StatsForecast(models=models, freq="D", n_jobs=-1)
sf_split.fit(df=train)

# Predict
split_preds = sf_split.predict(h=horizon)

# Evaluate
split_eval_df = pd.merge(test, split_preds, how='left', on=['ds', 'unique_id'])

metrics = [mae, mse, rmse, mape]

split_evaluation = evaluate(
    split_eval_df,
    metrics=metrics,
)

split_summary = (
    split_evaluation
    .drop(['unique_id'], axis=1)
    .groupby('metric')
    .mean()
    .reset_index()
)

print("\n=== Simple Split - Evaluation Summary ===")
print(split_summary)

# Reshape for model comparison
split_model_pivot = split_evaluation.melt(
    id_vars=['unique_id', 'metric'],
    var_name='model',
    value_name='value'
).pivot_table(
    index='model',
    columns='metric',
    values='value'
)

print("\n=== Simple Split - Model Comparison ===")
print(split_model_pivot)

# ========== PART 2: TIME SERIES CROSS-VALIDATION ==========
print("\n" + "="*70)
print("PART 2: TIME SERIES CROSS-VALIDATION APPROACH")
print("="*70)

# Create StatsForecast instance for CV
sf_cv = StatsForecast(models=models, freq="D", n_jobs=-1)

# Apply time-series cross-validation
#     - h = 30 → each test window covers 30 days (forecast horizon)
#     - n_windows = 5 → number of rolling evaluation windows
#     - step_size = 30 → after each CV step, move forward 30 days
#     - refit = True → retrain models at each step (more realistic evaluation)

n_windows = 5
step_size = horizon

print(f"\nCross-Validation Configuration:")
print(f"  - Horizon (h): {horizon} days")
print(f"  - Number of windows: {n_windows}")
print(f"  - Step size: {step_size} days")
print(f"  - Refit models: True")
print(f"  - Total evaluations: {horizon * n_windows} days")

print("\nRunning cross-validation... (this may take a few minutes)")

cv_df = sf_cv.cross_validation(
    h=horizon,
    df=df_au,
    n_windows=n_windows,
    step_size=step_size,
    refit=True
)

print(f"Cross-validation completed! Generated {len(cv_df)} predictions.")
print(f"CV dataframe columns: {cv_df.columns.tolist()}")

# Evaluate CV results
cv_eval = evaluate(
    cv_df.drop(columns=["cutoff"]),  # cutoff not needed for metrics
    metrics=metrics,
)

cv_summary = (
    cv_eval
    .drop(columns=["unique_id"])
    .groupby("metric", as_index=False)
    .mean()
)

print("\n=== Cross-Validation - Evaluation Summary ===")
print(cv_summary)

# Reshape CV results for model comparison
cv_model_pivot = cv_eval.melt(
    id_vars=['unique_id', 'metric'],
    var_name='model',
    value_name='value'
).pivot_table(
    index='model',
    columns='metric',
    values='value'
)

print("\n=== Cross-Validation - Model Comparison ===")
print(cv_model_pivot)

# ========== PART 3: COMPARE SIMPLE SPLIT vs CV ==========
print("\n" + "="*70)
print("PART 3: COMPARISON - SIMPLE SPLIT vs CROSS-VALIDATION")
print("="*70)

# Combine results for comparison
comparison_data = []

for model in split_model_pivot.index:
    for metric in ['mae', 'rmse']:
        split_val = split_model_pivot.loc[model, metric] if model in split_model_pivot.index else np.nan
        cv_val = cv_model_pivot.loc[model, metric] if model in cv_model_pivot.index else np.nan
        
        comparison_data.append({
            'Model': model,
            'Metric': metric.upper(),
            'Simple Split': split_val,
            'Cross-Validation': cv_val,
            'Difference': cv_val - split_val
        })

comparison_df = pd.DataFrame(comparison_data)

print("\n=== MAE and RMSE Comparison ===")
for metric in ['MAE', 'RMSE']:
    print(f"\n{metric}:")
    metric_comp = comparison_df[comparison_df['Metric'] == metric]
    print(metric_comp.to_string(index=False))

# ========== PART 4: VISUALIZATIONS ==========
print("\n" + "="*70)
print("PART 4: CREATING VISUALIZATIONS")
print("="*70)

model_names = ['Naive', 'HistoricAverage', 'WindowAverage', 'SeasonalNaive', 'ARIMA', 'SARIMA']
colors = ['red', 'orange', 'purple', 'brown', 'darkblue', 'darkgreen']

# Plot 1: Simple Split - Test Period
fig, ax = plt.subplots(figsize=(15, 7))

ax.plot(test['ds'], test['y'], label='Actual', color='black', linewidth=3, marker='o', markersize=7, zorder=10)

for i, model in enumerate(model_names):
    if model in split_eval_df.columns:
        ax.plot(split_eval_df['ds'], split_eval_df[model], 
                label=model, color=colors[i], linewidth=2, 
                linestyle='--', marker='x', markersize=5, alpha=0.8)

ax.set_xlabel('Date', fontsize=12)
ax.set_ylabel('Gold Price (xAu)', fontsize=12)
ax.set_title('Simple Train/Test Split - Last 30 Days Prediction', fontsize=14, fontweight='bold')
ax.legend(fontsize=10, loc='best', ncol=2)
ax.grid(True, alpha=0.3)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Plot 2: Cross-Validation - All Windows
fig, ax = plt.subplots(figsize=(16, 8))

# Plot actual values
ax.plot(df_au['ds'], df_au['y'], label='Actual', color='black', linewidth=2, alpha=0.7)

# Plot CV predictions with different colors per window
unique_cutoffs = cv_df['cutoff'].unique()
print(f"\nCross-validation cutoff dates: {unique_cutoffs}")

# Plot one model's CV predictions to show the rolling windows
model_to_show = 'ARIMA'  # Choose one model to visualize CV windows
if model_to_show in cv_df.columns:
    for i, cutoff in enumerate(unique_cutoffs):
        window_data = cv_df[cv_df['cutoff'] == cutoff]
        ax.plot(window_data['ds'], window_data[model_to_show], 
                color='red', linewidth=1.5, linestyle='--', 
                alpha=0.6, marker='o', markersize=3)
        
        # Mark the cutoff point
        ax.axvline(x=cutoff, color='gray', linestyle=':', alpha=0.5, linewidth=1)

ax.set_xlabel('Date', fontsize=12)
ax.set_ylabel('Gold Price (xAu)', fontsize=12)
ax.set_title(f'Time Series Cross-Validation - Rolling Windows ({model_to_show} predictions)', 
             fontsize=14, fontweight='bold')
ax.legend(fontsize=10, loc='best')
ax.grid(True, alpha=0.3)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Plot 3: Model Performance Comparison - Split vs CV
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# MAE Comparison
ax1 = axes[0]
mae_split = split_model_pivot['mae'].sort_index()
mae_cv = cv_model_pivot['mae'].sort_index()

x = np.arange(len(mae_split))
width = 0.35

bars1 = ax1.bar(x - width/2, mae_split.values, width, label='Simple Split', color='skyblue')
bars2 = ax1.bar(x + width/2, mae_cv.values, width, label='Cross-Validation', color='lightcoral')

ax1.set_xlabel('Models', fontsize=11)
ax1.set_ylabel('Mean Absolute Error (MAE)', fontsize=11)
ax1.set_title('MAE: Simple Split vs Cross-Validation', fontsize=13, fontweight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels(mae_split.index, rotation=45, ha='right')
ax1.legend()
ax1.grid(True, alpha=0.3, axis='y')

# Add value labels
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}', ha='center', va='bottom', fontsize=9)

# RMSE Comparison
ax2 = axes[1]
rmse_split = split_model_pivot['rmse'].sort_index()
rmse_cv = cv_model_pivot['rmse'].sort_index()

bars3 = ax2.bar(x - width/2, rmse_split.values, width, label='Simple Split', color='skyblue')
bars4 = ax2.bar(x + width/2, rmse_cv.values, width, label='Cross-Validation', color='lightcoral')

ax2.set_xlabel('Models', fontsize=11)
ax2.set_ylabel('Root Mean Squared Error (RMSE)', fontsize=11)
ax2.set_title('RMSE: Simple Split vs Cross-Validation', fontsize=13, fontweight='bold')
ax2.set_xticks(x)
ax2.set_xticklabels(rmse_split.index, rotation=45, ha='right')
ax2.legend()
ax2.grid(True, alpha=0.3, axis='y')

# Add value labels
for bars in [bars3, bars4]:
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}', ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.show()

# Plot 4: Best Models Comparison
fig, ax = plt.subplots(figsize=(10, 6))

best_models_split = split_model_pivot['mae'].sort_values().head(3)
best_models_cv = cv_model_pivot['mae'].sort_values().head(3)

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

split_vals = [split_model_pivot.loc[m, 'mae'] if m in split_model_pivot.index else 0 
              for m in models_to_compare]
cv_vals = [cv_model_pivot.loc[m, 'mae'] if m in cv_model_pivot.index else 0 
           for m in models_to_compare]

bars1 = ax.bar(x_pos - 0.2, split_vals, 0.4, label='Simple Split', color='#2E86AB')
bars2 = ax.bar(x_pos + 0.2, cv_vals, 0.4, label='Cross-Validation', color='#A23B72')

ax.set_xlabel('Models', fontsize=12)
ax.set_ylabel('Mean Absolute Error (MAE)', fontsize=12)
ax.set_title('Top Performing Models - MAE Comparison', fontsize=14, fontweight='bold')
ax.set_xticks(x_pos)
ax.set_xticklabels(models_to_compare, rotation=45, ha='right')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.show()

# ========== PART 5: FINAL SUMMARY ==========
print("\n" + "="*70)
print("PART 5: FINAL SUMMARY")
print("="*70)

print("\n=== Best Model (by MAE) ===")
best_split = split_model_pivot['mae'].idxmin()
best_cv = cv_model_pivot['mae'].idxmin()

print(f"\nSimple Split: {best_split} (MAE: {split_model_pivot.loc[best_split, 'mae']:.4f})")
print(f"Cross-Validation: {best_cv} (MAE: {cv_model_pivot.loc[best_cv, 'mae']:.4f})")

print("\n=== Key Insights ===")
print("1. Cross-validation provides more robust performance estimates")
print("2. CV evaluates models across multiple time periods, not just one")
print("3. Models may perform differently in CV vs simple split")
print("4. CV better reflects real-world forecasting performance")

print("\n" + "="*70)
print("ANALYSIS COMPLETE!")
print("="*70)





from utilsforecast.feature_engineering import pipeline, time_features
from functools import partial

# Define time features: day of week, week of year, month of year
time_feature_funcs = [
    partial(time_features, features=["day", "week", "month"])
]

# Build feature matrices
# This creates:
# - exog_df: features for historical data (training)
# - futr_exog_df: features for future horizon (forecasting)
exog_df, futr_exog_df = pipeline(
    df=df_au,
    features=time_feature_funcs,
    freq="D",
    h=horizon
)

print("\nTime features created:")
print(exog_df.head())
print(f"\nExogenous features shape: {exog_df.shape}")
print(f"Future exogenous features shape: {futr_exog_df.shape}")



exog_enhanced = pd.merge(
    exog_df.reset_index(),  # Your dataset with time features
    df_wide[['ds'] + crypto_cols],  # Crypto prices in wide format
    on='ds',
    how='left'
)


from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

train_lr = exog_enhanced.iloc[:-horizon]
test_lr  = exog_enhanced.iloc[-horizon:]

X_cols = ["day", "week", "month", 'BNB', 'BTC', 'GT', 'LEO', 'OKB', 'SUN', 'TRX', 'XRP']
X_train, y_train = train_lr[X_cols], train_lr["y"]
X_test,  y_test  = test_lr[X_cols],  test_lr["y"]


lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)

mae_lr = mean_absolute_error(y_test, y_pred_lr)
rmse_lr = np.sqrt(mean_squared_error(y_test, y_pred_lr))
r2_lr = r2_score(y_test, y_pred_lr)
mape_lr = np.mean(np.abs((y_test - y_pred_lr) / y_test)) * 100

print("="*70)
print("LINEAR REGRESSION MODEL PERFORMANCE")
print("="*70)
print(f"MAE:  {mae_lr:.4f}")
print(f"RMSE: {rmse_lr:.4f}")
print(f"R²:   {r2_lr:.4f}")
print(f"MAPE: {mape_lr:.2f}%")

# Prepare data for plotting
# Get dates from the original dataframe
train_lr_with_dates = train_lr.reset_index()
test_lr_with_dates = test_lr.reset_index()

# Plot 1: Full Timeline (Train + Test with Predictions)
fig = plt.figure(figsize=(18, 8))

# Plot training data
plt.plot(train_lr_with_dates['ds'], train_lr_with_dates['y'], 
         label='Training Data', color='blue', linewidth=1.5, alpha=0.6)

# Plot actual test data
plt.plot(test_lr_with_dates['ds'], y_test, 
         label='Actual (Test)', color='black', linewidth=3, marker='o', markersize=7, zorder=10)

# Plot predictions
plt







