import time
import random
import matplotlib.pyplot as plt
import seaborn
from dataset import Dataset
from dataset import DatasetProvider


# Set the path to the file you'd like to load from Kaggle's dataset
csvFiles = [
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
    # "MINA.csv",  # Bronken link
    "MX.csv",
    "NEAR.csv",
    "NEO.csv",
    "NEXO.csv",
    "NFT.csv",
    "OKB.csv",
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
    # "SUPER.csv",  # Broken link
    "SYRUP.csv",
    "TFUEL.csv",
    "THETA.csv",
    "TRAC.csv",
    "TRX.csv",
    "TUSD.csv",
    "TWT.csv",
    "UNI.csv",
    "VET.csv",
    "WEMIX.csv",
    "XCN.csv",
    "XDC.csv",
    "XEC.csv",
    "XLM.csv",
    "XMR.csv",
    "XRP.csv",
    "XTZ.csv",
    # "YFI.csv",  # Broken link
    "ZEC.csv",
    "ZEN.csv",
    # "ZRX.csv",  # Broken link
]

# TODO: remove this overload of files (gaining time in DEBUG mode)
# csvFiles = ["BTC.csv"]

# data = Dataset(columns={"ticker", "date", "open", "high", "low", "close"}, trim=False)
data = Dataset(columns={"ticker", "date", "close"}, trim=False)

data.setDateFormat("yyyy-mm-dd")

localFile = True
if localFile:
    data.addDataset(source=DatasetProvider.CSV, file="dataset.csv")
else:
    for i, csvFile in enumerate(csvFiles):
        data.addDataset(
            source=DatasetProvider.KAGGLE,
            repo="svaningelgem/crypto-currencies-daily-prices",
            file=csvFile,
        )
        # Pause 30s every 40 files to avoid download limit
        if i % 40 == 0:
            time.sleep(30)

    goldProvider = "Kaggle"
    match goldProvider:
        case "Kaggle":
            data.addDataset(
                source=DatasetProvider.KAGGLE,
                repo="isaaclopgu/gold-historical-data-daily-updated",
                file="Gold_Spot_historical_data.csv",
                ticker="XAU",
            )
        case "Yahoo":
            data.addDataset(source=DatasetProvider.YAHOO, file="GC=F", ticker="XAU")
        case _:
            raise ValueError("Bad provider")

data.normalize("close")

data.exportDataset("csv")

print(data.df.head)


column = "closeNormalized"
colors = [
    "dimgrey",
    "rosybrown",
    "lightcoral",
    "red",
    "coral",
    "sienna",
    # "seashell",  # Not much saturated
    "chocolate",
    "darkorange",
    "tan",
    "khaki",
    "beige",
    "olive",
    "greenyellow",
    "palegreen",
    "darkgreen",
    "aquamarine",
    "lightseagreen",
    "teal",
    "cyan",
    "deepskyblue",
    "royalblue",
    "lavender",
    "navy",
    "slateblue",
    "blueviolet",
    "plum",
    "magenta",
    "deeppink",
    "crimson",
]
random.Random(42).shuffle(colors)

methods = ["pearson", "kendall", "spearman"]
minCorr = 0.65
maxCorr = 0.99

tickers = ["XAU"]

fig, axes = plt.subplots(3, 2, figsize=(18, 12))
for i, method in enumerate(methods):
    axes[i, 0].clear()
    corrMatrix = data.corr(values=column, method=method, min=minCorr, max=maxCorr)
    if not corrMatrix.empty:
        tickers.extend(corrMatrix.T.index.tolist())
        seaborn.heatmap(
            corrMatrix,
            annot=True,
            annot_kws={"size": 10},
            fmt=".2f",
            cmap="coolwarm",
            center=0,
            square=True,
            linewidths=0.5,
            cbar_kws={"shrink": 0.8},
            ax=axes[i, 0],
        )
        axes[i, 0].set_title(
            f"Most correlated (>{minCorr * 100:.0f}%) wrt {method}",
            fontsize=14,
            fontweight="bold",
            pad=20,
        )

        for j, ticker in enumerate(corrMatrix.columns):
            axes[i, 1].plot(
                data.getTicker(ticker)["date"],
                data.getTicker(ticker)[column],
                marker="s",
                linestyle="--",
                linewidth=1,
                markersize=1,
                label=ticker,
                color=colors[j],
            )
        axes[i, 1].plot(
            data.getTicker("XAU")["date"],
            data.getTicker("XAU")[column],
            marker="o",
            linestyle="-",
            linewidth=1,
            markersize=1,
            label="XAU",
            color="gold",
        )
        axes[i, 1].set_xlabel("dates", fontsize=12)
        axes[i, 1].set_ylabel(f"{column} prices", fontsize=12)
        axes[i, 1].set_title(
            f"Gold / Cryptos comparison ({column} prices)",
            fontsize=14,
            fontweight="bold",
        )
        axes[i, 1].legend(fontsize=11)
        axes[i, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.subplots_adjust(wspace=0.3, hspace=0.3)
plt.show()
# fig.savefig("chart.png")

# Keep unique ticker names
tickers = list(set(tickers))
print(tickers)

data.df = data.dropTickers(keep=tickers)
data.trimDates()
data.normalize("close")
# print(data.df.columns.tolist())
print(data.df.head)
data.exportDataset("csv", "dataset_reduced.csv")


methods = ["pearson", "kendall", "spearman"]
minCorr = 0.70
maxCorr = 0.99

tickers = ["XAU"]

fig, axes = plt.subplots(3, 2, figsize=(18, 12))
for i, method in enumerate(methods):
    axes[i, 0].clear()
    corrMatrix = data.corr(values=column, method=method, min=minCorr, max=maxCorr)
    if not corrMatrix.empty:
        tickers.extend(corrMatrix.T.index.tolist())
        seaborn.heatmap(
            corrMatrix,
            annot=True,
            annot_kws={"size": 10},
            fmt=".2f",
            cmap="coolwarm",
            center=0,
            square=True,
            linewidths=0.5,
            cbar_kws={"shrink": 0.8},
            ax=axes[i, 0],
        )
        axes[i, 0].set_title(
            f"Most correlated (>{minCorr * 100:.0f}%) wrt {method}",
            fontsize=14,
            fontweight="bold",
            pad=20,
        )

        for j, ticker in enumerate(corrMatrix.columns):
            axes[i, 1].plot(
                data.getTicker(ticker)["date"],
                data.getTicker(ticker)[column],
                marker="s",
                linestyle="--",
                linewidth=1,
                markersize=1,
                label=ticker,
                color=colors[j],
            )
        axes[i, 1].plot(
            data.getTicker("XAU")["date"],
            data.getTicker("XAU")[column],
            marker="o",
            linestyle="-",
            linewidth=1,
            markersize=1,
            label="XAU",
            color="gold",
        )
        axes[i, 1].set_xlabel("dates", fontsize=12)
        axes[i, 1].set_ylabel(f"{column} prices", fontsize=12)
        axes[i, 1].set_title(
            f"Gold / Cryptos comparison ({column} prices)",
            fontsize=14,
            fontweight="bold",
        )
        axes[i, 1].legend(fontsize=11)
        axes[i, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.subplots_adjust(wspace=0.3, hspace=0.3)
plt.show()
print(colors[3])
