import random
import matplotlib.pyplot as plt
import seaborn
import pandas
from utilsforecast.evaluation import evaluate
from utilsforecast.losses import *
from statsforecast import StatsForecast
from statsforecast.models import Naive, HistoricAverage, WindowAverage, SeasonalNaive
from statsforecast.models import AutoARIMA
from dataset import Dataset
from dataset import DatasetProvider


# Define a random color list
colors = [
    "dimgrey",
    "rosybrown",
    "lightcoral",
    "red",
    "coral",
    "sienna",
    # "seashell",  # Too clear
    "chocolate",
    "darkorange",
    "tan",
    "khaki",
    # "beige",  # Too clear
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


def drawCorrelationMatricesAndPlot(corrMatrices, methods, minCorr, df, column):
    """Draw a subplot of correlation matrices with associated plot

    corrMatrices -- a list of correlation matrices
    methods -- method names of the computation of the correlation matrices
    minCorr -- minimal corralation percentage used in the computation of the correlation matrices
    df -- dataframe used to plot
    column -- column from the dataframe used in the plot (with "date")
    """
    fig, axes = plt.subplots(3, 2, figsize=(18, 12))
    if len(corrMatrices) != len(methods):
        raise (
            Exception(
                f"The number of correlation matrices is not the same as the number of methods to calculate them ({len(corrMatrices)}!={len(methods)})"
            )
        )

    for i, corrMatrix in enumerate(corrMatrices):
        axes[i, 0].clear()
        if not corrMatrix.empty:
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
                f"Most correlated (>{minCorr * 100:.0f}%) wrt {methods[i]}",
                fontsize=14,
                fontweight="bold",
                pad=20,
            )

            for j, ticker in enumerate(corrMatrix.columns):
                dfTicker = df[df["ticker"] == ticker]
                axes[i, 1].plot(
                    dfTicker["date"],
                    dfTicker[column],
                    marker="s",
                    linestyle="--",
                    linewidth=1,
                    markersize=1,
                    label=ticker,
                    color=colors[j],
                )
            dfTicker = df[df["ticker"] == "XAU"]
            axes[i, 1].plot(
                dfTicker["date"],
                dfTicker[column],
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
    # "PAXG.csv",  # Too correlated
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
    "TEL.csv",
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
    # "XAUt.csv",  # Too correlated
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
    # "ZRX.csv",  # Broken link
]


# data = Dataset(columns={"ticker", "date", "open", "high", "low", "close"})
data = Dataset(columns={"ticker", "date", "close"})
data.setDateFormat("yyyy-mm-dd")

# Don't download the dataset and use the local .csv to gain time if localFile==True
localFile = True
if localFile:
    data.addDataset(source=DatasetProvider.CSV, file="dataset.csv")
else:
    for csvFile in csvFiles:
        data.addDataset(
            source=DatasetProvider.KAGGLE,
            repo="svaningelgem/crypto-currencies-daily-prices",
            file=csvFile,
        )

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


# First correlation matrices to select the most relevant tickers before preprocessing
column = "closeNormalized"

methods = ["pearson", "kendall", "spearman"]
minCorr = 0.65
maxCorr = 0.99

tickers = ["XAU"]

corrMatrices = []
for method in methods:
    corrMatrices.append(
        data.corr(values=column, method=method, min=minCorr, max=maxCorr)
    )
    if not corrMatrices[-1].empty:
        tickers.extend(corrMatrices[-1].T.index.tolist())
drawCorrelationMatricesAndPlot(corrMatrices, methods, minCorr, data.df, column)

# Keep unique ticker names
tickers = list(set(tickers))
print(tickers)


data.df = data.dropTickers(keep=tickers)
data.trimDates()
# Remove all data after the hole from 2024-12-10 to 2025-01-26
data.end = "2024-12-09"
data.df = data.df[data.df["date"] <= data.end]
data.normalize("close")
# print(data.df.columns.tolist())
# print(data.df.head)
data.exportDataset("csv", "dataset_reduced.csv")


# Now we have only the most relevant tickers in the dataset, with the same time period


# data2 = Dataset(columns={"ticker", "date", "close"})
# data2.setDateFormat("yyyy-mm-dd")
# data2.start = data.start
# data2.end = data.end
# for ticker in tickers:
#    if ticker != "XAU":
#        data2.addDataset(source=DatasetProvider.YAHOO, file=ticker, ticker=ticker)
# data2.addDataset(
#    source=DatasetProvider.KAGGLE,
#    repo="isaaclopgu/gold-historical-data-daily-updated",
#    file="Gold_Spot_historical_data.csv",
#    ticker="XAU",
# )
# data2.trimDates()
# data2.normalize("close")
# data2.exportDataset("csv", "dataset_reduced.csv")


# Second correlation matrices computation with usable values (the dataset is preprocessed)
column = "closeNormalized"

methods = ["pearson", "kendall", "spearman"]
minCorr = 0.65
maxCorr = 0.99

tickers = ["XAU"]

corrMatrices = []
for method in methods:
    corrMatrices.append(
        data.corr(values=column, method=method, min=minCorr, max=maxCorr)
    )
    if not corrMatrices[-1].empty:
        tickers.extend(corrMatrices[-1].T.index.tolist())
drawCorrelationMatricesAndPlot(corrMatrices, methods, minCorr, data.df, column)


print(data.df.head)
# Keep unique ticker names
tickers = list(set(tickers))
print(f"{len(tickers)} selected tickers: {tickers}")
# print(f"{data.df.shape[0]} rows x {data.df.shape[1]} columns")


# TODO: machine learning (time based)
print("\nTime series Forecasting...")
# TODO: Only BTC is taken, need to use every tickers with added "XAU" as exogenous variable
dfCV = data.getTicker("BTC")[["date", "ticker", "closeNormalized"]]
dfXAU = data.getTicker("XAU")[["date", "closeNormalized"]].rename(
    columns={"closeNormalized": "XAU"}
)
# dfCV = pandas.merge(dfCV, dfXAU, how="inner", on="date")
dfCV = pandas.merge(data.df, dfXAU, how="inner", on="date")
dfCV = dfCV.rename(
    columns={"date": "ds", "ticker": "unique_id", "closeNormalized": "y"}
)
dfCV = dfCV[dfCV["unique_id"] != "XAU"]
dfCV.to_csv("timeseries.csv")
print(dfCV.head)
horizon = 7
test = dfCV.tail(7)
train = dfCV.drop(test.index).reset_index(drop=True)
models = [
    Naive(),
    HistoricAverage(),
    WindowAverage(window_size=7),
    SeasonalNaive(season_length=7),
    AutoARIMA(seasonal=False, alias="ARIMA"),
    AutoARIMA(season_length=7, alias="SARIMA"),
]
sf = StatsForecast(models=models, freq="D")
# sf.fit(df=train)

cv_df = sf.cross_validation(
    h=horizon, df=dfCV, n_windows=8, step_size=horizon, refit=True
)

metrics = [mae, mse, rmse, mape]
cv_eval = evaluate(cv_df.drop(columns=["cutoff"]), metrics=metrics)
cv_summary = (
    cv_eval.drop(columns=["unique_id"]).groupby("metric", as_index=False).mean()
)
print(cv_summary)
