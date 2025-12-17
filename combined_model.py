import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from statsforecast import StatsForecast
from statsforecast.models import AutoARIMA
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")

print("\n" + "=" * 70)
print("ARIMA + LSTM ENSEMBLE MODEL COMPARISON")
print("Combining statistical and deep learning approaches")
print("=" * 70)

# ========================================
# PART 1: Load Data
# ========================================
print("\nLoading datasets...")

time_df = pd.read_csv("model_datasets/time_features.csv")
df_exogf = pd.read_csv("model_datasets/features_complete.csv")
time_df["ds"] = pd.to_datetime(time_df["ds"])
df_exogf["ds"] = pd.to_datetime(df_exogf["ds"])

print(f"Time features dataset: {time_df.shape}")
print(f"Time + Crypto dataset: {df_exogf.shape}")

# ========================================
# PART 2: LSTM Model Definition
# ========================================


class LSTMPredictor(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.2):
        super(LSTMPredictor, self).__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
        )

        self.fc1 = nn.Linear(hidden_size, 32)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(32, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_output = lstm_out[:, -1, :]
        out = self.relu(self.fc1(last_output))
        out = self.dropout(out)
        out = self.fc2(out)
        return out


# ========================================
# PART 3: Prepare Data for Both Models
# ========================================


def prepare_data_for_ensemble(df, test_size=60, seq_length=20):
    """Prepare data for both ARIMA and LSTM"""

    # For ARIMA: needs 'unique_id' column
    df_arima = df.copy()
    if "unique_id" not in df_arima.columns:
        df_arima["unique_id"] = "XAU"

    # Split for ARIMA
    train_arima = df_arima.iloc[:-test_size].copy()
    test_arima = df_arima.iloc[-test_size:].copy()

    # For LSTM: prepare sequences
    feature_cols = [col for col in df.columns if col not in ["ds", "unique_id", "y"]]

    train_size = len(df) - test_size
    train_features = df[feature_cols].values[:train_size]
    train_target = df["y"].values[:train_size].reshape(-1, 1)
    test_features = df[feature_cols].values[train_size:]
    test_target = df["y"].values[train_size:].reshape(-1, 1)

    all_features = df[feature_cols].values
    all_target = df["y"].values.reshape(-1, 1)

    # Scale
    feature_scaler = MinMaxScaler()
    target_scaler = MinMaxScaler()

    feature_scaler.fit(all_features)
    target_scaler.fit(all_target)

    train_features_scaled = feature_scaler.fit_transform(train_features)
    train_target_scaled = target_scaler.fit_transform(train_target)
    test_features_scaled = feature_scaler.transform(test_features)
    test_target_scaled = target_scaler.transform(test_target)

    # Create sequences
    def create_sequences(features, target, seq_length):
        X, y = [], []
        for i in range(len(features) - seq_length):
            X.append(features[i : i + seq_length])
            y.append(target[i + seq_length])
        return np.array(X), np.array(y)

    X_train, y_train = create_sequences(
        train_features_scaled, train_target_scaled, seq_length
    )
    X_test, y_test = create_sequences(
        test_features_scaled, test_target_scaled, seq_length
    )

    X_train = torch.FloatTensor(X_train)
    y_train = torch.FloatTensor(y_train)
    X_test = torch.FloatTensor(X_test)
    y_test = torch.FloatTensor(y_test)

    return {
        "arima_train": train_arima,
        "arima_test": test_arima,
        "lstm_X_train": X_train,
        "lstm_y_train": y_train,
        "lstm_X_test": X_test,
        "lstm_y_test": y_test,
        "feature_scaler": feature_scaler,
        "target_scaler": target_scaler,
        "feature_cols": feature_cols,
        "test_dates": df["ds"].iloc[-len(y_test) :].values,
    }


print("\n" + "=" * 70)
print("Preparing Data for Ensemble Models")
print("=" * 70)

# Prepare both datasets
print("\n1) Time-Only Dataset:")
data_time = prepare_data_for_ensemble(time_df, test_size=60, seq_length=20)
print(f"   ARIMA train: {data_time['arima_train'].shape}")
print(f"   LSTM sequences: {data_time['lstm_X_train'].shape}")

print("\n2) Time+Crypto Dataset:")
data_crypto = prepare_data_for_ensemble(df_exogf, test_size=60, seq_length=20)
print(f"   ARIMA train: {data_crypto['arima_train'].shape}")
print(f"   LSTM sequences: {data_crypto['lstm_X_train'].shape}")

# ========================================
# PART 4: Train ARIMA Models
# ========================================
print("\n" + "=" * 70)
print("Training ARIMA Models")
print("=" * 70)


def train_arima(train_data, test_data, X_test=None):
    """Train ARIMA with optional exogenous variables"""
    horizon = len(test_data)

    model = AutoARIMA(seasonal=False)
    sf = StatsForecast(models=[model], freq="D", n_jobs=-1)

    sf.fit(df=train_data)

    # Prepare test data
    if X_test is not None:
        test_X = test_data.drop(columns=["y"])
        predictions = sf.predict(h=horizon, X_df=test_X)
    else:
        predictions = sf.predict(h=horizon)

    return predictions["AutoARIMA"].values


print("\n1) Training ARIMA on Time-Only features...")
arima_pred_time = train_arima(
    data_time["arima_train"], data_time["arima_test"], X_test=data_time["arima_test"]
)
print(f"   ✓ ARIMA predictions: {arima_pred_time.shape}")

print("\n2) Training ARIMA on Time+Crypto features...")
arima_pred_crypto = train_arima(
    data_crypto["arima_train"],
    data_crypto["arima_test"],
    X_test=data_crypto["arima_test"],
)
print(f"   ✓ ARIMA predictions: {arima_pred_crypto.shape}")

# ========================================
# PART 5: Train LSTM Models
# ========================================
print("\n" + "=" * 70)
print("Training LSTM Models")
print("=" * 70)


def train_lstm(X_train, y_train, X_test, y_test, input_size, epochs=100):
    """Train LSTM model"""

    model = LSTMPredictor(input_size=input_size, hidden_size=64, num_layers=2)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    best_loss = float("inf")
    patience = 15
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        # Validation
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_test)
            val_loss = criterion(val_outputs, y_test).item()

        # Early stopping
        if val_loss < best_loss:
            best_loss = val_loss
            patience_counter = 0
            best_state = model.state_dict().copy()
        else:
            patience_counter += 1

        if (epoch + 1) % 20 == 0:
            print(f"   Epoch [{epoch + 1}/{epochs}], Val Loss: {val_loss:.6f}")

        if patience_counter >= patience:
            print(f"   Early stopping at epoch {epoch + 1}")
            break

    model.load_state_dict(best_state)
    return model


print("\n1) Training LSTM on Time-Only features...")
lstm_model_time = train_lstm(
    data_time["lstm_X_train"],
    data_time["lstm_y_train"],
    data_time["lstm_X_test"],
    data_time["lstm_y_test"],
    input_size=data_time["lstm_X_train"].shape[2],
)

# Get LSTM predictions
lstm_model_time.eval()
with torch.no_grad():
    lstm_pred_time_scaled = lstm_model_time(data_time["lstm_X_test"])
    try:
        lstm_pred_time_scaled = lstm_pred_time_scaled.detach().numpy()
    except:
        lstm_pred_time_scaled = np.array(lstm_pred_time_scaled.tolist())

    lstm_pred_time = (
        data_time["target_scaler"].inverse_transform(lstm_pred_time_scaled).flatten()
    )

print(f"   ✓ LSTM predictions: {lstm_pred_time.shape}")

print("\n2) Training LSTM on Time+Crypto features...")
lstm_model_crypto = train_lstm(
    data_crypto["lstm_X_train"],
    data_crypto["lstm_y_train"],
    data_crypto["lstm_X_test"],
    data_crypto["lstm_y_test"],
    input_size=data_crypto["lstm_X_train"].shape[2],
)

# Get LSTM predictions
lstm_model_crypto.eval()
with torch.no_grad():
    lstm_pred_crypto_scaled = lstm_model_crypto(data_crypto["lstm_X_test"])
    try:
        lstm_pred_crypto_scaled = lstm_pred_crypto_scaled.detach().numpy()
    except:
        lstm_pred_crypto_scaled = np.array(lstm_pred_crypto_scaled.tolist())

    lstm_pred_crypto = (
        data_crypto["target_scaler"]
        .inverse_transform(lstm_pred_crypto_scaled)
        .flatten()
    )

print(f"   ✓ LSTM predictions: {lstm_pred_crypto.shape}")

# ========================================
# PART 6: Align Predictions (CRITICAL)
# ========================================
print("\n" + "=" * 70)
print("Aligning Predictions")
print("=" * 70)

# ARIMA predicts all test points, LSTM needs sequences
# Match lengths by taking last N predictions from ARIMA
lstm_len_time = len(lstm_pred_time)
lstm_len_crypto = len(lstm_pred_crypto)

arima_pred_time_aligned = arima_pred_time[-lstm_len_time:]
arima_pred_crypto_aligned = arima_pred_crypto[-lstm_len_crypto:]

# Get actual values (aligned)
actual_time = data_time["arima_test"]["y"].values[-lstm_len_time:]
actual_crypto = data_crypto["arima_test"]["y"].values[-lstm_len_crypto:]

# Get aligned dates
dates_time = data_time["test_dates"]
dates_crypto = data_crypto["test_dates"]

print("\nTime-Only Dataset:")
print(f"  ARIMA: {len(arima_pred_time_aligned)}")
print(f"  LSTM: {len(lstm_pred_time)}")
print(f"  Actual: {len(actual_time)}")

print("\nTime+Crypto Dataset:")
print(f"  ARIMA: {len(arima_pred_crypto_aligned)}")
print(f"  LSTM: {len(lstm_pred_crypto)}")
print(f"  Actual: {len(actual_crypto)}")

# ========================================
# PART 7: Create Ensemble Predictions
# ========================================
print("\n" + "=" * 70)
print("Creating Ensemble Predictions")
print("=" * 70)

# Simple Average
ensemble_avg_time = (arima_pred_time_aligned + lstm_pred_time) / 2
ensemble_avg_crypto = (arima_pred_crypto_aligned + lstm_pred_crypto) / 2

print("✓ Simple average ensemble created")


# Weighted Average (based on individual performance)
# Calculate weights from validation performance
def calculate_weights(arima_preds, lstm_preds, actual):
    """Calculate optimal weights based on MAE"""
    mae_arima = mean_absolute_error(actual, arima_preds)
    mae_lstm = mean_absolute_error(actual, lstm_preds)

    # Inverse MAE as weights (lower error = higher weight)
    weight_arima = (1 / mae_arima) / ((1 / mae_arima) + (1 / mae_lstm))
    weight_lstm = (1 / mae_lstm) / ((1 / mae_arima) + (1 / mae_lstm))

    return weight_arima, weight_lstm


w_arima_time, w_lstm_time = calculate_weights(
    arima_pred_time_aligned, lstm_pred_time, actual_time
)
w_arima_crypto, w_lstm_crypto = calculate_weights(
    arima_pred_crypto_aligned, lstm_pred_crypto, actual_crypto
)

ensemble_weighted_time = (
    w_arima_time * arima_pred_time_aligned + w_lstm_time * lstm_pred_time
)
ensemble_weighted_crypto = (
    w_arima_crypto * arima_pred_crypto_aligned + w_lstm_crypto * lstm_pred_crypto
)

print("\n✓ Weighted ensemble created")
print(f"  Time-Only weights: ARIMA={w_arima_time:.3f}, LSTM={w_lstm_time:.3f}")
print(f"  Time+Crypto weights: ARIMA={w_arima_crypto:.3f}, LSTM={w_lstm_crypto:.3f}")

# ========================================
# PART 8: Calculate Metrics for All Models
# ========================================
print("\n" + "=" * 70)
print("Evaluating All Models")
print("=" * 70)


def calculate_metrics(actual, predicted, model_name):
    """Calculate comprehensive metrics"""
    mae = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    r2 = r2_score(actual, predicted)
    corr, _ = pearsonr(actual, predicted)

    # Directional accuracy
    actual_diff = np.diff(actual)
    pred_diff = np.diff(predicted)
    dir_acc = np.mean((actual_diff > 0) == (pred_diff > 0)) * 100

    return {
        "Model": model_name,
        "MAE": mae,
        "RMSE": rmse,
        "R²": r2,
        "Correlation": corr,
        "Dir_Accuracy_%": dir_acc,
    }


# Time-Only Dataset
results_time = []
results_time.append(
    calculate_metrics(actual_time, arima_pred_time_aligned, "ARIMA_TimeOnly")
)
results_time.append(calculate_metrics(actual_time, lstm_pred_time, "LSTM_TimeOnly"))
results_time.append(
    calculate_metrics(actual_time, ensemble_avg_time, "Ensemble_Avg_TimeOnly")
)
results_time.append(
    calculate_metrics(actual_time, ensemble_weighted_time, "Ensemble_Weighted_TimeOnly")
)

# Time+Crypto Dataset
results_crypto = []
results_crypto.append(
    calculate_metrics(actual_crypto, arima_pred_crypto_aligned, "ARIMA_Time+Crypto")
)
results_crypto.append(
    calculate_metrics(actual_crypto, lstm_pred_crypto, "LSTM_Time+Crypto")
)
results_crypto.append(
    calculate_metrics(actual_crypto, ensemble_avg_crypto, "Ensemble_Avg_Time+Crypto")
)
results_crypto.append(
    calculate_metrics(
        actual_crypto, ensemble_weighted_crypto, "Ensemble_Weighted_Time+Crypto"
    )
)

results_time_df = pd.DataFrame(results_time)
results_crypto_df = pd.DataFrame(results_crypto)

print("\n📊 TIME-ONLY DATASET RESULTS:")
print("=" * 70)
print(results_time_df.to_string(index=False))

print("\n📊 TIME+CRYPTO DATASET RESULTS:")
print("=" * 70)
print(results_crypto_df.to_string(index=False))

# ========================================
# PART 9: Find Best Models
# ========================================
print("\n" + "=" * 70)
print("🏆 BEST MODELS")
print("=" * 70)

best_time = results_time_df.loc[results_time_df["R²"].idxmax()]
best_crypto = results_crypto_df.loc[results_crypto_df["R²"].idxmax()]

print(f"\nBest for Time-Only: {best_time['Model']}")
print(f"  R²: {best_time['R²']:.4f}")
print(f"  MAE: ${best_time['MAE']:.2f}")

print(f"\nBest for Time+Crypto: {best_crypto['Model']}")
print(f"  R²: {best_crypto['R²']:.4f}")
print(f"  MAE: ${best_crypto['MAE']:.2f}")

# Compare best models
if best_crypto["R²"] > best_time["R²"]:
    improvement = ((best_crypto["R²"] - best_time["R²"]) / abs(best_time["R²"])) * 100
    print(f"\n✅ Crypto features improve best model by {improvement:.1f}%!")
else:
    print("\n❌ Crypto features do not improve the best model")

# ========================================
# PART 10: Visualization
# ========================================
print("\n" + "=" * 70)
print("Creating Visualizations")
print("=" * 70)

# Plot 1: Time-Only Dataset - All Models
fig, ax = plt.subplots(figsize=(16, 8))

ax.plot(
    dates_time,
    actual_time,
    label="Actual",
    color="black",
    linewidth=3,
    marker="o",
    markersize=6,
    zorder=10,
)
ax.plot(
    dates_time,
    arima_pred_time_aligned,
    label="ARIMA",
    color="blue",
    linewidth=2,
    linestyle="--",
    alpha=0.7,
)
ax.plot(
    dates_time,
    lstm_pred_time,
    label="LSTM",
    color="green",
    linewidth=2,
    linestyle="--",
    alpha=0.7,
)
ax.plot(
    dates_time,
    ensemble_avg_time,
    label="Ensemble (Avg)",
    color="red",
    linewidth=2,
    linestyle="-",
    marker="x",
    markersize=5,
    alpha=0.8,
)
ax.plot(
    dates_time,
    ensemble_weighted_time,
    label="Ensemble (Weighted)",
    color="purple",
    linewidth=2,
    linestyle="-",
    marker="+",
    markersize=6,
    alpha=0.8,
)

ax.set_xlabel("Date", fontsize=13, fontweight="bold")
ax.set_ylabel("Gold Price (xAU)", fontsize=13, fontweight="bold")
ax.set_title(
    f"Time-Only Models: ARIMA + LSTM Ensemble\nBest: {best_time['Model']} (R²={best_time['R²']:.3f})",
    fontsize=15,
    fontweight="bold",
)
ax.legend(fontsize=10, loc="best", framealpha=0.9)
ax.grid(True, alpha=0.3, linestyle=":")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("results/ensemble_time_only.png", dpi=300, bbox_inches="tight")
plt.show()

# Plot 2: Time+Crypto Dataset - All Models
fig, ax = plt.subplots(figsize=(16, 8))

ax.plot(
    dates_crypto,
    actual_crypto,
    label="Actual",
    color="black",
    linewidth=3,
    marker="o",
    markersize=6,
    zorder=10,
)
ax.plot(
    dates_crypto,
    arima_pred_crypto_aligned,
    label="ARIMA",
    color="blue",
    linewidth=2,
    linestyle="--",
    alpha=0.7,
)
ax.plot(
    dates_crypto,
    lstm_pred_crypto,
    label="LSTM",
    color="green",
    linewidth=2,
    linestyle="--",
    alpha=0.7,
)
ax.plot(
    dates_crypto,
    ensemble_avg_crypto,
    label="Ensemble (Avg)",
    color="red",
    linewidth=2,
    linestyle="-",
    marker="x",
    markersize=5,
    alpha=0.8,
)
ax.plot(
    dates_crypto,
    ensemble_weighted_crypto,
    label="Ensemble (Weighted)",
    color="purple",
    linewidth=2,
    linestyle="-",
    marker="+",
    markersize=6,
    alpha=0.8,
)

ax.set_xlabel("Date", fontsize=13, fontweight="bold")
ax.set_ylabel("Gold Price (xAU)", fontsize=13, fontweight="bold")
ax.set_title(
    f"Time+Crypto Models: ARIMA + LSTM Ensemble\nBest: {best_crypto['Model']} (R²={best_crypto['R²']:.3f})",
    fontsize=15,
    fontweight="bold",
)
ax.legend(fontsize=10, loc="best", framealpha=0.9)
ax.grid(True, alpha=0.3, linestyle=":")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("results/ensemble_time_crypto.png", dpi=300, bbox_inches="tight")
plt.show()

# Plot 3: Comparison Bar Chart
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# R² Comparison
models = ["ARIMA", "LSTM", "Ensemble\n(Avg)", "Ensemble\n(Weighted)"]
r2_time = [results_time_df.iloc[i]["R²"] for i in range(4)]
r2_crypto = [results_crypto_df.iloc[i]["R²"] for i in range(4)]

x = np.arange(len(models))
width = 0.35

axes[0].bar(
    x - width / 2, r2_time, width, label="Time-Only", color="steelblue", alpha=0.8
)
axes[0].bar(
    x + width / 2, r2_crypto, width, label="Time+Crypto", color="coral", alpha=0.8
)
axes[0].set_xlabel("Model", fontsize=12, fontweight="bold")
axes[0].set_ylabel("R² Score", fontsize=12, fontweight="bold")
axes[0].set_title("R² Comparison (Higher is Better)", fontsize=13, fontweight="bold")
axes[0].set_xticks(x)
axes[0].set_xticklabels(models)
axes[0].legend()
axes[0].grid(True, alpha=0.3, axis="y")

# MAE Comparison
mae_time = [results_time_df.iloc[i]["MAE"] for i in range(4)]
mae_crypto = [results_crypto_df.iloc[i]["MAE"] for i in range(4)]

axes[1].bar(
    x - width / 2, mae_time, width, label="Time-Only", color="steelblue", alpha=0.8
)
axes[1].bar(
    x + width / 2, mae_crypto, width, label="Time+Crypto", color="coral", alpha=0.8
)
axes[1].set_xlabel("Model", fontsize=12, fontweight="bold")
axes[1].set_ylabel("MAE ($)", fontsize=12, fontweight="bold")
axes[1].set_title("MAE Comparison (Lower is Better)", fontsize=13, fontweight="bold")
axes[1].set_xticks(x)
axes[1].set_xticklabels(models)
axes[1].legend()
axes[1].grid(True, alpha=0.3, axis="y")

plt.tight_layout()
plt.savefig("results/ensemble_comparison_bars.png", dpi=300, bbox_inches="tight")
plt.show()

# ========================================
# PART 11: Save Results
# ========================================
print("\n" + "=" * 70)
print("Saving Results")
print("=" * 70)

# Save predictions
predictions_time_df = pd.DataFrame(
    {
        "Date": dates_time,
        "Actual": actual_time,
        "ARIMA": arima_pred_time_aligned,
        "LSTM": lstm_pred_time,
        "Ensemble_Avg": ensemble_avg_time,
        "Ensemble_Weighted": ensemble_weighted_time,
    }
)
predictions_time_df.to_csv("results/ensemble_predictions_time.csv", index=False)

predictions_crypto_df = pd.DataFrame(
    {
        "Date": dates_crypto,
        "Actual": actual_crypto,
        "ARIMA": arima_pred_crypto_aligned,
        "LSTM": lstm_pred_crypto,
        "Ensemble_Avg": ensemble_avg_crypto,
        "Ensemble_Weighted": ensemble_weighted_crypto,
    }
)
predictions_crypto_df.to_csv("results/ensemble_predictions_crypto.csv", index=False)

# Save metrics
results_time_df.to_csv("results/ensemble_metrics_time.csv", index=False)
results_crypto_df.to_csv("results/ensemble_metrics_crypto.csv", index=False)

print("✓ Files saved:")
print("  - ensemble_predictions_time.csv")
print("  - ensemble_predictions_crypto.csv")
print("  - ensemble_metrics_time.csv")
print("  - ensemble_metrics_crypto.csv")
print("  - ensemble_time_only.png")
print("  - ensemble_time_crypto.png")
print("  - ensemble_comparison_bars.png")

print("\n" + "=" * 70)
print("✅ ENSEMBLE ANALYSIS COMPLETE!")
print("=" * 70)

# Final Summary
print("\n" + "=" * 70)
print("📊 FINAL SUMMARY")
print("=" * 70)

print("\n🏆 Best Model Overall:")
all_results = pd.concat([results_time_df, results_crypto_df], ignore_index=True)
best_idx = all_results["R²"].idxmax()

# FIX: Access values explicitly
best_model_name = all_results.loc[best_idx, "Model"]
best_r2 = all_results.loc[best_idx, "R²"]
best_mae = all_results.loc[best_idx, "MAE"]
best_corr = all_results.loc[best_idx, "Correlation"]
best_dir = all_results.loc[best_idx, "Dir_Accuracy_%"]

print(f"  Model: {best_model_name}")
print(f"  R²: {best_r2:.4f}")
print(f"  MAE: ${best_mae:.2f}")
print(f"  Correlation: {best_corr:.4f}")
print(f"  Directional Accuracy: {best_dir:.1f}%")


# Compare best models safely
best_time_idx = results_time_df["R²"].idxmax()
best_crypto_idx = results_crypto_df["R²"].idxmax()

best_time_r2 = results_time_df.loc[best_time_idx, "R²"]
best_crypto_r2 = results_crypto_df.loc[best_crypto_idx, "R²"]
best_time_name = results_time_df.loc[best_time_idx, "Model"]
best_crypto_name = results_crypto_df.loc[best_crypto_idx, "Model"]

print("\n📊 Dataset Comparison:")
print(f"  Best Time-Only: {best_time_name} (R² = {best_time_r2:.4f})")
print(f"  Best Time+Crypto: {best_crypto_name} (R² = {best_crypto_r2:.4f})")

if best_crypto_r2 > best_time_r2:
    improvement = ((best_crypto_r2 - best_time_r2) / abs(best_time_r2)) * 100
    print(f"\n  ✅ Crypto features improve predictions by {improvement:.1f}%")
else:
    print("\n  ❌ Time features alone are sufficient")

# Show all ensemble results
print("\n📋 All Ensemble Results Summary:")
print("=" * 70)

# Format nicely
for dataset, df in [("Time-Only", results_time_df), ("Time+Crypto", results_crypto_df)]:
    print(f"\n{dataset}:")
    for idx, row in df.iterrows():
        print(f"  {row['Model']:<30} R²={row['R²']:.4f}  MAE=${row['MAE']:.2f}")


print("\n" + "=" * 70)
print("✅ ANALYSIS COMPLETE!")
print("=" * 70)
