import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

print("\n" + "=" * 70)
print("DEEP LEARNING MODEL COMPARISON")
print("Dataset 1: Time Features Only")
print("Dataset 2: Time + Crypto Features")
print("=" * 70)

# ========================================
# PART 1: Load and Prepare Data
# ========================================
print("\nLoading datasets...")

time_df = pd.read_csv("model_datasets/time_features.csv")
df_exogf = pd.read_csv("model_datasets/features_complete.csv")
time_df['ds'] = pd.to_datetime(time_df['ds'])
df_exogf['ds'] = pd.to_datetime(df_exogf['ds'])

# Remove unique_id column and keep only features
time_df = time_df.sort_values('ds').reset_index(drop=True)
df_exogf = df_exogf.sort_values('ds').reset_index(drop=True)

print(f"Time features dataset: {time_df.shape}")
print(f"Time + Crypto dataset: {df_exogf.shape}")

# ========================================
# PART 2: Data Preparation Function
# ========================================

def prepare_dl_data(df, target_col='y', test_ratio=0.2, seq_length=20):
    """
    Prepare data for deep learning
    Returns: X_train, y_train, X_test, y_test, feature_scaler, target_scaler
    """
    # Remove non-feature columns
    feature_cols = [col for col in df.columns if col not in ['ds', 'unique_id', target_col]]
    
    # Extract features and target
    features = df[feature_cols].values
    target = df[target_col].values.reshape(-1, 1)

    total_samples = len(df)
    print(f"  Total samples: {total_samples}")
    print(f"  Features: {len(feature_cols)}")
    
    # Auto-adjust seq_length if data is too small
    if seq_length > total_samples * 0.1:
        seq_length = max(10, int(total_samples * 0.1))
        print(f"  ⚠ Adjusted seq_length to {seq_length} (10% of data)")

    # Calculate test size
    test_size = max(seq_length + 10, int(total_samples * test_ratio))
    train_size = total_samples - test_size
    
    print(f"  Train size: {train_size}, Test size: {test_size}")
    print(f"  Sequence length: {seq_length}")
    
    # Validate sizes
    if train_size < seq_length + 10:
        raise ValueError(f"Not enough training data! Total samples: {total_samples}, need at least {seq_length + test_size + 10}")
    
    train_features = features[:train_size]
    train_target = target[:train_size]
    test_features = features[train_size:]
    test_target = target[train_size:]
    
    # Scale features and target separately
    feature_scaler = MinMaxScaler()
    target_scaler = MinMaxScaler()
    
    train_features_scaled = feature_scaler.fit_transform(train_features)
    train_target_scaled = target_scaler.fit_transform(train_target)
    test_features_scaled = feature_scaler.transform(test_features)
    test_target_scaled = target_scaler.transform(test_target)
    
    # Create sequences
    def create_sequences(features, target, seq_length):
        X, y = [], []
        for i in range(len(features) - seq_length):
            X.append(features[i:i+seq_length])
            y.append(target[i+seq_length])
        return np.array(X), np.array(y)
    
    X_train, y_train = create_sequences(train_features_scaled, train_target_scaled, seq_length)
    X_test, y_test = create_sequences(test_features_scaled, test_target_scaled, seq_length)
    
    # Convert to PyTorch tensors
    X_train = torch.FloatTensor(X_train)
    y_train = torch.FloatTensor(y_train)
    X_test = torch.FloatTensor(X_test)
    y_test = torch.FloatTensor(y_test)
    
    print(f"  X_train: {X_train.shape} (samples, seq_length, features)")
    print(f"  X_test: {X_test.shape}")
    
    return X_train, y_train, X_test, y_test, feature_scaler, target_scaler, feature_cols

# Prepare both datasets
print("\n" + "=" * 70)
print("Preparing Time-Only Dataset")
print("=" * 70)
X_train_time, y_train_time, X_test_time, y_test_time, \
    scaler_feat_time, scaler_tgt_time, features_time = prepare_dl_data(time_df)

print("\n" + "=" * 70)
print("Preparing Time+Crypto Dataset")
print("=" * 70)
X_train_crypto, y_train_crypto, X_test_crypto, y_test_crypto, \
    scaler_feat_crypto, scaler_tgt_crypto, features_crypto = prepare_dl_data(df_exogf)

# ========================================
# PART 3: Define LSTM Model
# ========================================

class LSTMPredictor(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.2):
        super(LSTMPredictor, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        self.fc1 = nn.Linear(hidden_size, 32)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, 1)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_output = lstm_out[:, -1, :]
        out = self.relu(self.fc1(last_output))
        out = self.dropout(out)
        out = self.relu(self.fc2(out))
        out = self.fc3(out)
        return out

# ========================================
# PART 4: Training Function
# ========================================

def train_model(model, X_train, y_train, X_test, y_test, 
                epochs=100, batch_size=32, lr=0.001, patience=15):
    """Train LSTM model with early stopping"""
    
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=10, factor=0.5, verbose=False
    )
    
    best_loss = float('inf')
    patience_counter = 0
    train_losses = []
    val_losses = []
    
    print("Training started...")
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        
        # Mini-batch training
        for i in range(0, len(X_train), batch_size):
            batch_X = X_train[i:i+batch_size]
            batch_y = y_train[i:i+batch_size]
            
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            epoch_loss += loss.item()
        
        avg_train_loss = epoch_loss / (len(X_train) / batch_size)
        train_losses.append(avg_train_loss)
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_test)
            val_loss = criterion(val_outputs, y_test).item()
            val_losses.append(val_loss)
        
        scheduler.step(val_loss)
        
        # Early stopping
        if val_loss < best_loss:
            best_loss = val_loss
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1
        
        if (epoch + 1) % 20 == 0:
            print(f'  Epoch [{epoch+1}/{epochs}], Train Loss: {avg_train_loss:.6f}, Val Loss: {val_loss:.6f}')
        
        if patience_counter >= patience:
            print(f"  Early stopping at epoch {epoch+1}")
            break
    
    # Load best model
    model.load_state_dict(best_model_state)
    
    return model, train_losses, val_losses

# ========================================
# PART 5: Train Both Models
# ========================================

print("\n" + "=" * 70)
print("Training Model 1: Time Features Only")
print("=" * 70)

input_size_time = X_train_time.shape[2]
model_time = LSTMPredictor(input_size=input_size_time, hidden_size=64, num_layers=2)
print(f"Model parameters: {sum(p.numel() for p in model_time.parameters())}")

model_time, train_loss_time, val_loss_time = train_model(
    model_time, X_train_time, y_train_time, X_test_time, y_test_time,
    epochs=100, batch_size=32, lr=0.001, patience=15
)

print("\n" + "=" * 70)
print("Training Model 2: Time + Crypto Features")
print("=" * 70)

input_size_crypto = X_train_crypto.shape[2]
model_crypto = LSTMPredictor(input_size=input_size_crypto, hidden_size=64, num_layers=2)
print(f"Model parameters: {sum(p.numel() for p in model_crypto.parameters())}")

model_crypto, train_loss_crypto, val_loss_crypto = train_model(
    model_crypto, X_train_crypto, y_train_crypto, X_test_crypto, y_test_crypto,
    epochs=100, batch_size=32, lr=0.001, patience=15
)

# ========================================
# PART 6: Make Predictions
# ========================================

print("\n" + "=" * 70)
print("Making Predictions")
print("=" * 70)

model_time.eval()
model_crypto.eval()

with torch.no_grad():
    # Time-only predictions
    pred_time_list = model_time(X_test_time).tolist()
    pred_time_scaled = np.array(pred_time_list)
    pred_time = scaler_tgt_time.inverse_transform(pred_time_scaled)
    
    actual_time_list = y_test_time.tolist()
    actual_time_scaled = np.array(actual_time_list)
    actual_time = scaler_tgt_time.inverse_transform(actual_time_scaled)
    
    # Time+Crypto predictions
    pred_crypto_list = model_crypto(X_test_crypto).tolist()
    pred_crypto_scaled = np.array(pred_crypto_list)
    pred_crypto = scaler_tgt_crypto.inverse_transform(pred_crypto_scaled)
    
    actual_crypto_list = y_test_crypto.tolist()
    actual_crypto_scaled = np.array(actual_crypto_list)
    actual_crypto = scaler_tgt_crypto.inverse_transform(actual_crypto_scaled)


# ========================================
# PART 7: Calculate Metrics
# ========================================

def calculate_all_metrics(actual, predicted, model_name):
    """Calculate comprehensive metrics"""
    mae = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    mape = np.mean(np.abs((actual - predicted) / actual)) * 100
    r2 = r2_score(actual, predicted)
    corr, _ = pearsonr(actual.flatten(), predicted.flatten())
    
    # Directional accuracy
    actual_direction = np.diff(actual.flatten()) > 0
    pred_direction = np.diff(predicted.flatten()) > 0
    dir_acc = np.mean(actual_direction == pred_direction) * 100
    
    # Variance ratio
    var_ratio = np.var(predicted) / np.var(actual)
    
    return {
        'Model': model_name,
        'MAE': mae,
        'RMSE': rmse,
        'MAPE_%': mape,
        'R²': r2,
        'Correlation': corr,
        'Dir_Accuracy_%': dir_acc,
        'Variance_Ratio': var_ratio
    }

metrics_time = calculate_all_metrics(actual_time, pred_time, 'LSTM_TimeOnly')
metrics_crypto = calculate_all_metrics(actual_crypto, pred_crypto, 'LSTM_Time+Crypto')

results_df = pd.DataFrame([metrics_time, metrics_crypto])

print("\n" + "=" * 70)
print("DEEP LEARNING RESULTS COMPARISON")
print("=" * 70)
print(results_df.to_string(index=False))

# ========================================
# PART 8: Determine Winner
# ========================================

print("\n" + "=" * 70)
print("🎯 WHICH MODEL IS BETTER?")
print("=" * 70)

improvement_r2 = ((metrics_crypto['R²'] - metrics_time['R²']) / abs(metrics_time['R²'] + 0.0001)) * 100
improvement_mae = ((metrics_time['MAE'] - metrics_crypto['MAE']) / metrics_time['MAE']) * 100

print(f"\nR² Score:")
print(f"  Time-Only: {metrics_time['R²']:.4f}")
print(f"  Time+Crypto: {metrics_crypto['R²']:.4f}")
print(f"  Improvement: {improvement_r2:+.1f}%")

print(f"\nMAE:")
print(f"  Time-Only: ${metrics_time['MAE']:.2f}")
print(f"  Time+Crypto: ${metrics_crypto['MAE']:.2f}")
print(f"  Improvement: {improvement_mae:+.1f}%")

print(f"\nCorrelation with Actual:")
print(f"  Time-Only: {metrics_time['Correlation']:.4f}")
print(f"  Time+Crypto: {metrics_crypto['Correlation']:.4f}")

print(f"\nDirectional Accuracy:")
print(f"  Time-Only: {metrics_time['Dir_Accuracy_%']:.1f}%")
print(f"  Time+Crypto: {metrics_crypto['Dir_Accuracy_%']:.1f}%")

if metrics_crypto['R²'] > metrics_time['R²'] and metrics_crypto['Correlation'] > metrics_time['Correlation']:
    print(f"\n✅ CONCLUSION: Crypto features IMPROVE deep learning predictions!")
    print(f"   R² improved by {improvement_r2:.1f}%")
    print(f"   Model with crypto captures {metrics_crypto['R²']*100:.1f}% of variance")
else:
    print(f"\n❌ CONCLUSION: Crypto features do NOT improve predictions")

# ========================================
# PART 9: Visualizations
# ========================================

print("\n" + "=" * 70)
print("Creating Visualizations")
print("=" * 70)

# Create date index for test period
test_dates = time_df['ds'].iloc[-len(actual_time):].values

# Plot 1: Training History Comparison
fig, axes = plt.subplots(1, 2, figsize=(15, 5))

axes[0].plot(train_loss_time, label='Train', color='blue', alpha=0.7)
axes[0].plot(val_loss_time, label='Validation', color='red', alpha=0.7)
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss')
axes[0].set_title('Time-Only Model Training', fontweight='bold')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot(train_loss_crypto, label='Train', color='blue', alpha=0.7)
axes[1].plot(val_loss_crypto, label='Validation', color='red', alpha=0.7)
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Loss')
axes[1].set_title('Time+Crypto Model Training', fontweight='bold')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('results/dl_training_history.png', dpi=300, bbox_inches='tight')
plt.show()

# Plot 2: Predictions Comparison
fig, ax = plt.subplots(figsize=(16, 8))

ax.plot(test_dates, actual_time, label='Actual', 
        color='black', linewidth=3, marker='o', markersize=6, zorder=10)

ax.plot(test_dates, pred_time, label=f'LSTM Time-Only (R²={metrics_time["R²"]:.3f})', 
        color='blue', linewidth=2, linestyle='--', marker='s', markersize=5, alpha=0.7)

ax.plot(test_dates, pred_crypto, label=f'LSTM Time+Crypto (R²={metrics_crypto["R²"]:.3f})', 
        color='red', linewidth=2, linestyle='-', marker='x', markersize=6, alpha=0.8)

ax.set_xlabel('Date', fontsize=13, fontweight='bold')
ax.set_ylabel('Gold Price (xAU)', fontsize=13, fontweight='bold')
ax.set_title(f'Deep Learning Model Comparison\nCrypto Features Improve R² by {improvement_r2:+.1f}%', 
             fontsize=15, fontweight='bold')
ax.legend(fontsize=11, loc='best')
ax.grid(True, alpha=0.3, linestyle=':')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('results/dl_predictions_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

# Plot 3: Scatter Plots
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Time-Only
axes[0].scatter(actual_time, pred_time, alpha=0.6, s=80, color='blue')
axes[0].plot([actual_time.min(), actual_time.max()], 
             [actual_time.min(), actual_time.max()], 
             'r--', linewidth=2, label='Perfect Prediction')
axes[0].set_title(f'Time-Only Model\nR² = {metrics_time["R²"]:.4f}', fontweight='bold', fontsize=12)
axes[0].set_xlabel('Actual Gold Price', fontsize=11)
axes[0].set_ylabel('Predicted Gold Price', fontsize=11)
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Time+Crypto
axes[1].scatter(actual_crypto, pred_crypto, alpha=0.6, s=80, color='red')
axes[1].plot([actual_crypto.min(), actual_crypto.max()], 
             [actual_crypto.min(), actual_crypto.max()], 
             'r--', linewidth=2, label='Perfect Prediction')
axes[1].set_title(f'Time+Crypto Model\nR² = {metrics_crypto["R²"]:.4f}', fontweight='bold', fontsize=12)
axes[1].set_xlabel('Actual Gold Price', fontsize=11)
axes[1].set_ylabel('Predicted Gold Price', fontsize=11)
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('results/dl_scatter_comparison.png', dpi=300, bbox_inches='tight')
plt.show()
