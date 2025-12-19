# XAUxBTC
Compare exchange rates of gold and cryptocurrencies to find out any correlation

## 🚀 How to Run

### Prerequisites
```bash
pip install pandas numpy matplotlib scipy
pip install yfinance statsforecast xgboost torch scikit-learn
```

### Step 1: Generate Datasets
```bash
python main.py
```
**Output:**
- `model_datasets/dataset.csv` - full dataset downloaded from Kaggle
- `model_datasets/dataset_reduced.csv` - dataset containing only tickers correlated with gold
- `model_datasets/timeseries.csv` - same dataset converted to a time series with gold as an exogenous value
- `model_datasets/gold_raw.csv` - real gold price data
- `model_datasets/time_features.csv` - Time features only
- `model_datasets/features_complete.csv` - Time + crypto features

### Step 2: Run Individual Models

**Naive Model:**
```bash
python naive_model.py
```
**Output:**
- Predictions comparison plot
- Metrics comparison (MAE, R², Correlation)
- Answer: The baseline we will use to evaluate.

**ARIMA/SARIMA Model:**
```bash
python time_model.py
```
**Output:**
- Predictions comparison plot
- Metrics comparison (MAE, R², Correlation)
- Answer: Do crypto features help ARIMA/SARIMA?

**LSTM Model:**
```bash
python deeplearning_model.py
```
**Output:**
- Training history plots
- Predictions comparison
- Answer: Do crypto features help LSTM?

**Ensemble Model:** (already included in `main.py`)
```bash
python combined_model.py
```
**Output:**
- All 4 model predictions (ARIMA, LSTM, Avg, Weighted)
- Performance comparison
- Best model identification
