#python3 -m pip install
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from utilsforecast.plotting import plot_series
from utilsforecast.evaluation import evaluate
from statsforecast import StatsForecast
from statsforecast.models import WindowAverage, Naive, SeasonalNaive, HistoricAverage, AutoARIMA

from utilsforecast.evaluation import evaluate
from utilsforecast.losses import mae
from sklearn.metrics import mean_squared_error

# Load data
df = pd.read_csv("dataset.csv")

# Keep tickers only have correlation > 80%
tickers=['XAU', 'TRX', 'BTC', 'BNB', 'LEO', 'GT', 'SUN', 'OKB', 'XRP']
df=df[df['ticker'].isin(tickers)]

df = df[(df['date'] >= '2023-01-01') & (df['date'] <= '2025-06-01')]

df = df.rename(columns={
    'date': 'ds',           
    'ticker': 'unique_id',  
    'closeNormalized': 'y'            
})

#Convert ds(date) type to Datatime
df['ds'] = pd.to_datetime(df['ds'])


from statsforecast import StatsForecast
from statsforecast.models import Naive, HistoricAverage, WindowAverage, SeasonalNaive
#################### Naive #####################
df_au= df[df['unique_id'] == 'XAU'].copy()
horizon = 30   #Predict 30 days
naive_models = [
    Naive(),                    # Always predicts the last observed value
    HistoricAverage(),          # Predicts the average of all past values
    WindowAverage(window_size=30),  # Predicts the average of the last 7 days
    SeasonalNaive(season_length=30)  # Predicts the value from the same weekday last week

]


sf = StatsForecast(models=naive_models, freq="D")
sf.fit(df=df_au)

#  Generate predictions for the next 30 days (our horizon).
pred_naive = sf.predict(h=horizon)

print(pred_naive.head())

#sf.plot(df, preds_naive)
#plt.show(block=True)


plot_series(
    df=df_au,
    forecasts_df=pred_naive,
    ids=["XAU"],  
    max_insample_length=28,
)
plt.show()

# Test set: last 30 days
test = df_au.groupby("unique_id").tail(30)

# Train set: every time before last 30 days
train = df_au.drop(test.index).reset_index(drop=True)

# 3) Re-fit the baseline models on TRAIN ONLY
sf.fit(df=train)

# Predict the next 7 days
preds = sf.predict(h=horizon)

# 5) Prepare a single table that contains:
#    - the TRUE test values (from 'test') and
#    - the PREDICTED values (from 'preds') aligned by 'unique_id' and 'ds'
eval_df = pd.merge(test, preds, how='left', on=['ds', 'unique_id'])

# 6) Evaluate all metrics
from utilsforecast.evaluation import evaluate
from utilsforecast.losses import mae, mse, rmse, mape

metrics = [mae, mse, rmse, mape]

evaluation = evaluate(
    eval_df,
    metrics=metrics,
)


evaluation_summary = (
    evaluation    .drop(['unique_id'], axis=1)
    .groupby('metric')
    .mean()
    .reset_index()
)

print(evaluation_summary)


'''
################## Mutivariate regression ########################
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

# Prepare data
#X = df[['btc_price', 'eth_price']]  # Crypto prices
#y = df['gold_price']  # Target: gold price

# Split
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Model
#model = LinearRegression()
#model.fit(X_train, y_train)

# Predict
predictions = model.predict(X_test)

# Evaluate
from sklearn.metrics import mean_squared_error, r2_score
print(f"RMSE: {mean_squared_error(y_test, predictions, squared=False)}")
print(f"R²: {r2_score(y_test, predictions)}")



##################### ARIMA/SARIMA ######################
from statsmodels.tsa.statespace.sarimax import SARIMAX

# Use crypto as exogenous variables
model = SARIMAX(
    df['gold_price'],
    exog=df[['btc_price', 'eth_price']],
    order=(1, 1, 1)  # ARIMA parameters
)

results = model.fit()
print(results.summary())


'''