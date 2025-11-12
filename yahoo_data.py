#Make packages are download in the terminal
#python3 -m pip install yfinance
#python3 -m pip install pandas
#python3 -m pip install matplotlib
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt




ticker_list=['BTC-USD','GC=F','ETH-USD','USDT-USD','XPR-USD']

frames = []
for i in ticker_list:
    df=yf.download(i,start='2010-07-13',end='2025-11-10')
    df=df[['Close']]
    base = i.split('-')[0]  
    df=df.rename(columns={'Close':base})
    frames.append(df)

    print(i)#To tell us the process is in which ticker now

result = pd.concat(frames, axis=1)
print(result.head())

fig, axes = plt.subplots(3, 2, figsize=(12, 8), sharex=False)
axes = axes.ravel()

for ax, col in zip(axes, result.columns):
    # each subplot: plot one series; dropna to avoid gaps at the start
    result[col].dropna().plot(ax=ax)
    ax.set_title(col)
    ax.set_title(f"{col} close price trend over time")
    ax.set_xlabel("Date")
    ax.set_ylabel("Close Price")
    ax.grid(True, which='major', linestyle='--', alpha=0.5)
    ax.grid(True, which='minor', linestyle=':', alpha=0.3)

plt.tight_layout()
plt.show()


################ Normalization #####################
myticker=['BTC','GC=F','ETH']
#z-score
z = (result[myticker] - result[myticker].mean()) / result[myticker].std()   

plt.figure(figsize=(12, 6))
for t in z.columns:
    plt.plot(z.index, z[t], label=t)

plt.title("Z-score standardized prices")
plt.xlabel("Date")
plt.ylabel("Z-score")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()


