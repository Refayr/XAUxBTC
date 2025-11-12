import matplotlib.pyplot as plt
from dataset import Dataset
from dataset import DatasetProvider


# Set the path to the file you'd like to load from Kaggle's dataset
files = [
    "1INCH.csv",
    "AAVE.csv",
    "ADA.csv",
    "ALGO.csv",
    "AMP.csv",
    "APE.csv",#start 2020-10-01
    "AR.csv",#start 2024-02-07
    "ATOM.csv",
    "AVAX.csv",#2020-07-13
    "AXS.csv",#2020-08-31
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

# TODO: remove this overload of files (gaining time in DEBUG mode)
files = ["BTC.csv"]

data = Dataset(columns={"ticker", "date", "open", "high", "low", "close"}, trim=False)
data.setDateFormat("yyyy-mm-dd")
for file in files:
    data.addDataset(
        source=DatasetProvider.KAGGLE,
        repo="svaningelgem/crypto-currencies-daily-prices",
        file=file,
    )
data.addDataset(
    source=DatasetProvider.KAGGLE,
    repo="isaaclopgu/gold-historical-data-daily-updated",
    file="Gold_Spot_historical_data.csv",
    ticker="XAU",
)
data.exportDataset("csv")
# data.exportDataset("parquet")
# data.exportDataset("excel")

print(data.df.head)


# Draw values on a graph
plt.figure(figsize=(12, 6))

nbCols = 2
nbRows = 2
columns = ["open", "close", "low", "high"]
for i, column in enumerate(columns):
    plt.subplot(nbRows * 100 + nbCols * 10 + i + 1)
    plt.plot(
        data.getTicker("XAU")["date"],
        data.getTicker("XAU")[column],
        marker="o",
        linestyle="-",
        linewidth=2,
        markersize=2,
        label="XAU",
        color="gold",
    )
    plt.plot(
        data.getTicker("BTC")["date"],
        data.getTicker("BTC")[column],
        marker="s",
        linestyle="--",
        linewidth=2,
        markersize=2,
        label="BTC",
        color="blue",
    )
    plt.xlabel("dates", fontsize=12)
    plt.ylabel(f"{column} prices", fontsize=12)
    plt.title(
        f"Gold / Bitcoin comparison ({column} prices)", fontsize=14, fontweight="bold"
    )
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
