import kagglehub
from kagglehub import KaggleDatasetAdapter
import pandas
import matplotlib.pyplot as plt


# Set the path to the file you'd like to load from Kaggle's dataset
files = [
    "1INCH.csv",
    "AAVE.csv",
    "ADA.csv",
    "ALGO.csv",
    "AMP.csv",
    "APE.csv",
    "AR.csv",
    "ATOM.csv",
    "AVAX.csv",
    "AXS.csv",
    "BAT.csv",
    "BCH.csv",
    "BNB.csv",
    "BSV.csv",
    "BTC.csv",
    "BTT.csv",
    "CAKE.csv",
    "CFX.csv",
    "CHZ.csv",
    "COMP.csv",
    "CRO.csv",
    "CRV.csv",
    "CVX.csv",
    "DAI.csv",
    "DASH.csv",
    "DCR.csv",
    "DOGE.csv",
    "DOT.csv",
    "DYDX.csv",
    "EGLD.csv",
    "ENS.csv",
    "ETC.csv",
    "ETH.csv",
    "FET.csv",
    "FIL.csv",
    "FLOW.csv",
    "FTT.csv",
    "GALA.csv",
    "GLM.csv",
    "GNO.csv",
    "GRT.csv",
    "GT.csv",
    "HBAR.csv",
    "HNT.csv",
    "ICP.csv",
    "IMX.csv",
    "INJ.csv",
    "IOTA.csv",
    "JST.csv",
    "KCS.csv",
    "KSM.csv",
    "LDO.csv",
    "LEO.csv",
    "LINK.csv",
    "LPT.csv",
    "LTC.csv",
    "MANA.csv",
    "MINA.csv",
    "MX.csv",
    "NEAR.csv",
    "NEO.csv",
    "NEXO.csv",
    "NFT.csv",
    "OKB.csv",
    "PAXG.csv",
    "POL.csv",
    "QNT.csv",
    "QTUM.csv",
    "RAY.csv",
    "RSR.csv",
    "RUNE.csv",
    "SAND.csv",
    "SHIB.csv",
    "SNX.csv",
    "SOL.csv",
    "STX.csv",
    "SUN.csv",
    "SUPER.csv",
    "SYRUP.csv",
    "TFUEL.csv",
    "THETA.csv",
    "TRAC.csv",
    "TRX.csv",
    "TUSD.csv",
    "TWT.csv",
    "UNI.csv",
    "USDC.csv",
    "USDD.csv",
    "USDT.csv",
    "VET.csv",
    "WEMIX.csv",
    "XAUt.csv",
    "XCN.csv",
    "XDC.csv",
    "XEC.csv",
    "XLM.csv",
    "XMR.csv",
    "XRP.csv",
    "XTZ.csv",
    "YFI.csv",
    "ZEC.csv",
    "ZEN.csv",
    "ZRX.csv",
]

for i in range(len(files)):
    if files[i] == "BTC.csv":
        break

# Download latest cryptocurrency exchange rates
file_path = files[i]
dfBTC = kagglehub.load_dataset(
    KaggleDatasetAdapter.PANDAS,
    "svaningelgem/crypto-currencies-daily-prices",
    file_path,
    # Provide any additional arguments
    # See the documenation for more information:
    # https://github.com/Kaggle/kagglehub/blob/main/README.md#kaggledatasetadapterpandas
)

# Download latest gold exchange rates
file_path = "Gold_Spot_historical_data.csv"
dfXAU = kagglehub.load_dataset(
    KaggleDatasetAdapter.PANDAS,
    "isaaclopgu/gold-historical-data-daily-updated",
    file_path,
)


# We wants columns to be: "ticker", "date" (yyyy-mm-dd), "open", "high", "low", "close"
dfXAU["date"] = dfXAU["Date"].astype(str).str[:10]
dfXAU["ticker"] = "XAU"
dfXAU = dfXAU.rename(
    columns={"Open": "open", "High": "high", "Low": "low", "Close": "close"}
)
dfXAU = dfXAU.drop(columns=["Date", "Volume", "name"])


# Sort values by dates
dfXAU = dfXAU.sort_values(by="date")
dfBTC = dfBTC.sort_values(by="date")


# Get the first common date
firstDateGold = dfXAU["date"].iloc[0]
firstDateCrypto = dfBTC["date"].iloc[0]

if firstDateGold > firstDateCrypto:
    firstDate = firstDateGold
else:
    firstDate = firstDateCrypto
print(firstDate)


# Get the last common date
lastDateGold = dfXAU["date"].iloc[-1]
lastDateCrypto = dfBTC["date"].iloc[-1]

if lastDateGold < lastDateCrypto:
    lastDate = lastDateGold
else:
    lastDate = lastDateCrypto
print(lastDate)


# Create a new dataframe with only common columns
# df = pandas.DataFrame(data=None, index=None, columns=None, dtype=None, copy=None)
df = pandas.concat([dfBTC, dfXAU])
df = df[df["date"] >= firstDate]
df = df[df["date"] <= lastDate]
df = df.dropna()
df = df.sort_values(by="date")
print(df.head)


# Draw values on a graph
xau_data = df[df["ticker"] == "XAU"]
btc_data = df[df["ticker"] == "BTC"]
plt.figure(figsize=(12, 6))
plt.plot(
    xau_data["date"],
    xau_data["open"],
    marker="o",
    linewidth=2,
    markersize=4,
    label="XAU",
    color="yellow",
)
plt.plot(
    btc_data["date"],
    btc_data["open"],
    marker="s",
    linewidth=0,
    markersize=4,
    label="BTC",
    color="blue",
)
plt.xlabel("Date", fontsize=12)
plt.ylabel("Open prices", fontsize=12)
plt.title("Gold / Bitcoin comparison", fontsize=14, fontweight="bold")
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
