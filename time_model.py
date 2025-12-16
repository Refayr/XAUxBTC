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

from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr



print("\n" + "=" * 70)
print("Starting load dataset for forecasting")
print("=" * 70)

time_df = pd.read_csv("model_datasets/time_features.csv")
df_exogf = pd.read_csv("model_datasets/features_complete.csv")
df_exogf['ds'] = pd.to_datetime(df_exogf['ds'])
time_df['ds'] = pd.to_datetime(time_df['ds'])

print("\n" + "=" * 70)
print("End of dataset for forecasting")
print("Starting build train/test splits")
print("=" * 70)

# ========================================
# PART 1: Train/Test Split for BOTH Datasets
# ========================================

horizon = 30

# Split 1: Time features only
test_time = time_df.groupby("unique_id").tail(horizon)
train_time = time_df.drop(test_time.index).reset_index(drop=True)

# Split 2: Time + Crypto features
test_exogf = df_exogf.groupby("unique_id").tail(horizon)
train_exogf = df_exogf.drop(test_exogf.index).reset_index(drop=True)


print("\n" + "=" * 70)
print("End of build train/test splits")
print("Starting build and evaluate models")
print("=" * 70)

# ========================================
# PART 2: Train Models on BOTH Datasets
# ========================================

print("\n" + "=" * 70)
print("Training Models")
print("=" * 70)

# Models for time features only (baseline)
models_time = [
    AutoARIMA(seasonal=False, alias="ARIMA_TimeOnly"),
    AutoARIMA(season_length=7, alias="SARIMA_TimeOnly"),
]

# Models for time + crypto features
models_crypto = [
    AutoARIMA(seasonal=False, alias="ARIMA_WithCrypto"),
    AutoARIMA(season_length=7, alias="SARIMA_WithCrypto"),
]

print("\n1) Training models on TIME FEATURES ONLY...")
sf_time = StatsForecast(models=models_time, freq="D", n_jobs=-1)
sf_time.fit(df=train_time)
print("   ✓ Time-only models trained")

print("\n2) Training models on TIME + CRYPTO FEATURES...")
sf_crypto = StatsForecast(models=models_crypto, freq="D", n_jobs=-1)
sf_crypto.fit(df=train_exogf)
print("   ✓ Time+Crypto models trained")

# ========================================
# PART 3: Make Predictions
# ========================================
print("\n" + "=" * 70)
print("Making Predictions")
print("=" * 70)

# Predict with time-only models
test_time_X = test_time.drop(columns=["y"])
preds_time = sf_time.predict(h=horizon, X_df=test_time_X)

# Predict with time+crypto models
test_exogf_X = test_exogf.drop(columns=["y"])
preds_crypto = sf_crypto.predict(h=horizon, X_df=test_exogf_X)

print(f"Time-only predictions: {preds_time.shape}")
print(f"Time+Crypto predictions: {preds_crypto.shape}")

# ========================================
# PART 4: Merge and Evaluate
# ========================================
print("\n" + "=" * 70)
print("Evaluating Models")
print("=" * 70)

# Merge actual values with predictions
eval_time = pd.merge(test_time, preds_time, how="left", on=["ds", "unique_id"])
eval_crypto = pd.merge(test_exogf, preds_crypto, how="left", on=["ds", "unique_id"])

metrics = [mae, mse, rmse, mape]

# Evaluate time-only models
print("\n1) Evaluating TIME-ONLY models...")
time_evaluation = evaluate(
    eval_time[["ds", "unique_id", "y", "ARIMA_TimeOnly", "SARIMA_TimeOnly"]],
    metrics=metrics,
)
time_summary = time_evaluation.drop(["unique_id"], axis=1).groupby("metric").mean().reset_index()

# Evaluate time+crypto models
print("2) Evaluating TIME+CRYPTO models...")
crypto_evaluation = evaluate(
    eval_crypto[["ds", "unique_id", "y", "ARIMA_WithCrypto", "SARIMA_WithCrypto"]],
    metrics=metrics,
)
crypto_summary = crypto_evaluation.drop(["unique_id"], axis=1).groupby("metric").mean().reset_index()

# ========================================
# PART 5: Results Comparison
# ========================================
print("\n" + "=" * 70)
print("RESULTS: TIME FEATURES ONLY (Baseline)")
print("=" * 70)
print(time_summary)

print("\n" + "=" * 70)
print("RESULTS: TIME + CRYPTO FEATURES")
print("=" * 70)
print(crypto_summary)

# Calculate improvement
print("\n" + "=" * 70)
print("IMPROVEMENT FROM ADDING CRYPTO FEATURES")
print("=" * 70)

# Compare best models from each approach
time_mae = time_evaluation[time_evaluation['metric'] == 'mae'][['ARIMA_TimeOnly', 'SARIMA_TimeOnly']].min().min()
crypto_mae = crypto_evaluation[crypto_evaluation['metric'] == 'mae'][['ARIMA_WithCrypto', 'SARIMA_WithCrypto']].min().min()

improvement = ((time_mae - crypto_mae) / time_mae) * 100

print(f"Best Time-Only MAE: ${time_mae:.2f}")
print(f"Best Time+Crypto MAE: ${crypto_mae:.2f}")
print(f"Improvement: {improvement:.2f}%")

if improvement > 0:
    print(f"\n✅ YES! Crypto features improve prediction by {improvement:.1f}%")
else:
    print(f"\n❌ NO. Crypto features do not improve prediction (worse by {abs(improvement):.1f}%)")

# ========================================
# PART 6: Visualization - All Models on One Graph
# ========================================
print("\n" + "=" * 70)
print("Creating Comparison Visualization")
print("=" * 70)

# Merge all predictions into one dataframe
combined_df = test_exogf[["ds", "unique_id", "y"]].copy()
combined_df = combined_df.merge(preds_time[["ds", "unique_id", "ARIMA_TimeOnly", "SARIMA_TimeOnly"]], 
                                 on=["ds", "unique_id"], how="left")
combined_df = combined_df.merge(preds_crypto[["ds", "unique_id", "ARIMA_WithCrypto", "SARIMA_WithCrypto"]], 
                                on=["ds", "unique_id"], how="left")

# Plot
fig, ax = plt.subplots(figsize=(16, 8))

# Actual values
ax.plot(combined_df["ds"], combined_df["y"], 
        label="Actual", color="black", linewidth=3, 
        marker="o", markersize=7, zorder=10)

# Time-only models (dashed lines)
ax.plot(combined_df["ds"], combined_df["ARIMA_TimeOnly"], 
        label="ARIMA (Time Only)", color="blue", linewidth=2, 
        linestyle="--", marker="s", markersize=5, alpha=0.7)

ax.plot(combined_df["ds"], combined_df["SARIMA_TimeOnly"], 
        label="SARIMA (Time Only)", color="cyan", linewidth=2, 
        linestyle="--", marker="^", markersize=5, alpha=0.7)

# Time+Crypto models (solid lines)
ax.plot(combined_df["ds"], combined_df["ARIMA_WithCrypto"], 
        label="ARIMA (Time + Crypto)", color="red", linewidth=2, 
        linestyle="-", marker="x", markersize=6, alpha=0.8)

ax.plot(combined_df["ds"], combined_df["SARIMA_WithCrypto"], 
        label="SARIMA (Time + Crypto)", color="orange", linewidth=2, 
        linestyle="-", marker="+", markersize=7, alpha=0.8)

ax.set_xlabel("Date", fontsize=13, fontweight="bold")
ax.set_ylabel("Gold Price (xAU)", fontsize=13, fontweight="bold")
ax.set_title(f"Model Comparison: Do Crypto Features Help Predict Gold?\n(Improvement: {improvement:.1f}%)", 
             fontsize=15, fontweight="bold")
ax.legend(fontsize=11, loc="best", ncol=2, framealpha=0.9)
ax.grid(True, alpha=0.3, linestyle=":")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('gold_prediction_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

print("✓ Plot saved as 'gold_prediction_comparison.png'")

# ========================================
# PART 7: Statistical Summary Table
# ========================================
print("\n" + "=" * 70)
print("DETAILED MODEL COMPARISON TABLE")
print("=" * 70)

# Create comparison table
comparison_table = pd.DataFrame({
    'Model': ['ARIMA_TimeOnly', 'SARIMA_TimeOnly', 'ARIMA_WithCrypto', 'SARIMA_WithCrypto'],
    'Features': ['Time Only', 'Time Only', 'Time + Crypto', 'Time + Crypto'],
})

# Add metrics
for metric_name in ['mae', 'rmse', 'mape']:
    metric_col = metric_name.upper()
    
    # Get values for time-only models
    time_vals = time_evaluation[time_evaluation['metric'] == metric_name][
        ['ARIMA_TimeOnly', 'SARIMA_TimeOnly']].mean().tolist()
    # Get values for crypto models
    crypto_vals = crypto_evaluation[crypto_evaluation['metric'] == metric_name][
        ['ARIMA_WithCrypto', 'SARIMA_WithCrypto']].mean().tolist()
    
    comparison_table[metric_col] = time_vals + crypto_vals

print(comparison_table.to_string(index=False))

    


# ========================================
# PART 8: Better Evaluation Metrics
# ========================================
print("\n" + "=" * 70)
print("ADVANCED METRICS: Capturing Movement vs Just Minimizing Error")
print("=" * 70)

def calculate_advanced_metrics(actual, predicted, model_name):
    """Calculate metrics that capture movement patterns"""
    
    # 1. Correlation (measures if model follows actual movements)
    corr, p_value = pearsonr(actual, predicted)
    
    # 2. R² Score (variance explained)
    from sklearn.metrics import r2_score
    r2 = r2_score(actual, predicted)
    
    # 3. Directional Accuracy (does it predict up/down correctly?)
    actual_direction = np.diff(actual) > 0
    pred_direction = np.diff(predicted) > 0
    directional_accuracy = np.mean(actual_direction == pred_direction) * 100
    
    # 4. Variance captured (is model just predicting mean?)
    pred_variance = np.var(predicted)
    actual_variance = np.var(actual)
    variance_ratio = pred_variance / actual_variance
    
    # 5. Mean Absolute Percentage Error
    mape = np.mean(np.abs((actual - predicted) / actual)) * 100
    
    # 6. Tracking Error (standard deviation of differences)
    tracking_error = np.std(actual - predicted)
    
    return {
        'Model': model_name,
        'Correlation': corr,
        'R²': r2,
        'Directional_Accuracy_%': directional_accuracy,
        'Variance_Ratio': variance_ratio,  # Should be close to 1.0
        'MAPE_%': mape,
        'Tracking_Error': tracking_error,
        'Is_Flat_Line': 'YES' if variance_ratio < 0.1 else 'NO'
    }

# Calculate for all models
actual = combined_df['y'].values
results = []

results.append(calculate_advanced_metrics(
    actual, combined_df['ARIMA_TimeOnly'].values, 'ARIMA_TimeOnly'
))
results.append(calculate_advanced_metrics(
    actual, combined_df['SARIMA_TimeOnly'].values, 'SARIMA_TimeOnly'
))
results.append(calculate_advanced_metrics(
    actual, combined_df['ARIMA_WithCrypto'].values, 'ARIMA_WithCrypto'
))
results.append(calculate_advanced_metrics(
    actual, combined_df['SARIMA_WithCrypto'].values, 'SARIMA_WithCrypto'
))

advanced_metrics_df = pd.DataFrame(results)

print("\n" + "=" * 70)
print("ADVANCED METRICS COMPARISON")
print("=" * 70)
print(advanced_metrics_df.to_string(index=False))

# Interpretation
print("\n" + "=" * 70)
print("INTERPRETATION")
print("=" * 70)
print("\n📊 What these metrics mean:")
print("  • Correlation: How well predictions follow actual movements (higher is better)")
print("  • R²: Variance explained by model (closer to 1.0 is better)")
print("  • Directional Accuracy: % of correct up/down predictions (>50% is better than random)")
print("  • Variance Ratio: Prediction variance / Actual variance")
print("    - Close to 1.0 = Capturing movements")
print("    - Close to 0.0 = Flat line (just predicting mean)")
print("  • Is_Flat_Line: YES if variance ratio < 0.1 (model predicts near-constant)")

# Determine winner
time_only_r2 = advanced_metrics_df[advanced_metrics_df['Model'].str.contains('TimeOnly')]['R²'].max()
crypto_r2 = advanced_metrics_df[advanced_metrics_df['Model'].str.contains('WithCrypto')]['R²'].max()

time_only_corr = advanced_metrics_df[advanced_metrics_df['Model'].str.contains('TimeOnly')]['Correlation'].max()
crypto_corr = advanced_metrics_df[advanced_metrics_df['Model'].str.contains('WithCrypto')]['Correlation'].max()

time_only_var = advanced_metrics_df[advanced_metrics_df['Model'].str.contains('TimeOnly')]['Variance_Ratio'].max()
crypto_var = advanced_metrics_df[advanced_metrics_df['Model'].str.contains('WithCrypto')]['Variance_Ratio'].max()

print("\n" + "=" * 70)
print("🎯 FINAL VERDICT: DO CRYPTO FEATURES HELP?")
print("=" * 70)

if crypto_r2 > time_only_r2 and crypto_corr > time_only_corr:
    print(f"\n✅ YES! Crypto features significantly improve prediction!")
    print(f"\n   Time-Only:")
    print(f"     - R² = {time_only_r2:.4f} (explains {time_only_r2*100:.1f}% of variance)")
    print(f"     - Correlation = {time_only_corr:.4f}")
    print(f"     - Variance Ratio = {time_only_var:.4f} {'⚠️ FLAT LINE!' if time_only_var < 0.1 else ''}")
    print(f"\n   Time + Crypto:")
    print(f"     - R² = {crypto_r2:.4f} (explains {crypto_r2*100:.1f}% of variance)")
    print(f"     - Correlation = {crypto_corr:.4f}")
    print(f"     - Variance Ratio = {crypto_var:.4f}")
    print(f"\n   Improvement: {((crypto_r2 - time_only_r2) / abs(time_only_r2 + 0.0001)) * 100:.1f}% better R²")
else:
    print("❌ Crypto features do not consistently improve predictions")

# Check for flat line models
flat_models = advanced_metrics_df[advanced_metrics_df['Is_Flat_Line'] == 'YES']['Model'].tolist()
if flat_models:
    print(f"\n⚠️  WARNING: These models are basically flat lines (not capturing movements):")
    for model in flat_models:
        print(f"   - {model}")
    print("\n💡 This explains why they have low MAE but look like straight lines!")
    print("   They're just predicting near the average, not tracking actual movements.")

