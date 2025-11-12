#Make packages are download in the terminal
#python3 -m pip install yfinance
#python3 -m pip install pandas
#python3 -m pip install matplotlib
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

df_BTC=yf.download("BTC-USD", start='2010-07-13',end='2025-11-10')

df_BTC["AvgDay"] = (df_BTC["High"] + df_BTC["Low"] + df_BTC["Open"] + df_BTC["Close"]) / 4

print(df_BTC.head())



btc_prices = df_BTC[("AvgDay")]  

#plt.figure(figsize=(10, 5))
#plt.plot(btc_prices.index, btc_prices)
#plt.xlabel("Date")
#plt.ylabel("BTC Price (USD)")
#plt.title("BTC-USD average day Price Over Time")
#plt.grid(True)
#plt.tight_layout()
#plt.show()


df_Au=yf.download("GC=F", start='2010-07-13',end='2025-11-10')
df_Au["AvgDay"] = (df_Au["High"] + df_Au["Low"] + df_Au["Open"] + df_Au["Close"]) / 4
print(df_Au.head())

au_prices = df_Au[("AvgDay")]

#plt.figure(figsize=(10, 5))
#plt.plot(au_prices.index, au_prices)
#plt.xlabel("Date")
#plt.ylabel("Au Price (USD)")
#plt.title("Au average day Price Over Time")
#plt.grid(True)
#plt.tight_layout()
#plt.show()



fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

ax1.plot(btc_prices.index, btc_prices)
ax1.set_xlabel("Date")
ax1.set_ylabel("BTC Price (USD)")
ax1.set_title("BTC-USD average day Price Over Time")
ax1.grid(True)

ax2.plot(au_prices.index, au_prices)
ax2.set_xlabel("Date")
ax2.set_ylabel("Au Price (USD)")
ax2.set_title("Au average day Price Over Time")
ax2.grid(True)

fig.tight_layout()
plt.show()


# Combine and align on common dates
df_trend = pd.concat(
    [btc_prices.rename("BTC"), au_prices.rename("AU")],
    axis=1,
    join="inner"
).dropna()

# Option 1: Min-max normalization (compare shape in [0, 1])
btc_norm = (df_trend["BTC"] - df_trend["BTC"].min()) / (df_trend["BTC"].max() - df_trend["BTC"].min())
au_norm  = (df_trend["AU"]  - df_trend["AU"].min())  / (df_trend["AU"].max()  - df_trend["AU"].min())

# Option 2 (alternative): z-score (comment out min-max above if you use this)
# btc_norm = (df_trend["BTC"] - df_trend["BTC"].mean()) / df_trend["BTC"].std()
# au_norm  = (df_trend["AU"]  - df_trend["AU"].mean())  / df_trend["AU"].std()

fig, ax = plt.subplots(figsize=(12, 5))

ax.plot(df_trend.index, btc_norm, label="BTC (normalized)")
ax.plot(df_trend.index, au_norm, label="Au (normalized)")

ax.set_xlabel("Date")
ax.set_ylabel("Normalized Price (trend only)")
ax.set_title("BTC vs Au - Trend Comparison (Normalized)")
ax.grid(True)
ax.legend()

plt.tight_layout()
plt.show()
