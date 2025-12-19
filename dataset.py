#####################################################
##### Class ordering all datasets in one usable #####
#####################################################
# python3 -m pip install kagglehub yfinance pandas
import kagglehub
from kagglehub import KaggleDatasetAdapter
import pandas

import yfinance
import enum
import time

import random
import matplotlib.pyplot as plt
import seaborn


class DatasetProvider(enum.Enum):
    CSV = 1
    KAGGLE = 1 << 1
    YAHOO = 1 << 2


class Dataset:
    def __init__(self, columns):
        self.columns = columns
        self.dateFormat = None
        self.trim = False
        self.df = None
        self.start = -1
        self.end = -1

    def setDateFormat(self, dateFormat="yyyy-mm-dd"):
        self.dateFormat = dateFormat

    def getDateFormat(self):
        return self.dateFormat

    def cleanColumns(self, df, ticker):
        if self.columns is not None:
            referenceLower = {col.lower() for col in self.columns}

            # def shouldDrop(col):
            #    return col.lower() not in referenceLower

            def shouldRename(col):
                return col.lower() != col and col.lower() in referenceLower

            # drop = [col for col in df.columns if shouldDrop(col)]
            drop = filter(lambda x: x.lower() not in referenceLower, df.columns)
            rename = {col: col.lower() for col in df.columns if shouldRename(col)}

            df = df.drop(columns=drop).rename(columns=rename)

        df["date"] = df["date"].astype(str).str[: len(self.dateFormat)]
        df["date"] = pandas.to_datetime(df["date"])
        if ticker is not None:
            df["ticker"] = ticker

        return df

    def trimDates(self):
        """Only keeps the smallest common range of dates in the dataset"""
        self.trim = True
        df = self.df.sort_values(by="date")
        dfPivoted = df.pivot(index="date", columns="ticker", values="close")
        dfPivoted = dfPivoted.dropna()
        self.start = dfPivoted.index[0]
        self.end = dfPivoted.index[-1]
        df = df[(df["date"] >= self.start) & (df["date"] <= self.end)]
        self.df = df.dropna()

        return self.df

    def _importFromYahoo(self, ticker, start=-1, end=-1):
        if start == -1 or end == -1:
            df = yfinance.download(ticker, period="max")
        else:
            df = yfinance.download(ticker, start=start, end=end)

        df = df.stack().reset_index().rename(columns={"index": "date"})

        return df

    def _importFromKaggle(self, repo, file=None):
        # Download latest dataset from Kaggle
        result = None
        try:
            result = kagglehub.load_dataset(
                KaggleDatasetAdapter.PANDAS,
                repo,
                file,
                # Provide any additional arguments
                # See the documenation for more information:
                # https://github.com/Kaggle/kagglehub/blob/main/README.md#kaggledatasetadapterpandas
            )
        except Exception:
            # Sleeps 30sec when the download limit is reached
            print(f"Waiting 30s before downloading {file}")
            time.sleep(30)
            try:
                result = kagglehub.load_dataset(
                    KaggleDatasetAdapter.PANDAS,
                    repo,
                    file,
                )
            except Exception:
                raise Exception(
                    f"KAGGLE ERROR: {file} not found in the following repository: {repo}"
                )

        return result

    def _importFromCsv(self, file):
        return pandas.read_csv(file)

    def addDataset(self, source, repo=None, file=None, ticker=None):
        # TODO: reject importing an already present ticker
        # and handle exceptions raised by import error from Yahoo and Kaggle
        match source:
            case DatasetProvider.KAGGLE:
                # Download latest dataset from Kaggle
                dfTmp = self._importFromKaggle(repo, file)
            case DatasetProvider.YAHOO:
                # Download latest dataset from Yahoo Finance
                dfTmp = self._importFromYahoo(file, self.start, self.end)
            case DatasetProvider.CSV:
                # Use a local .csv file
                dfTmp = self._importFromCsv(file)
            case _:
                raise ValueError("Bad provider")
                return self.df

        dfTmp = self.cleanColumns(dfTmp, ticker)

        dfTmp = dfTmp.sort_values(by="date")

        if self.df is None:
            self.df = dfTmp
        else:
            self.df = pandas.concat([self.df, dfTmp])

        return self.df

    def getTicker(self, ticker=""):
        if ticker == "":
            return self.df
        else:
            return self.df[self.df["ticker"] == ticker]

    def exportDataset(self, format="csv", file=None):
        if self.df is not None:
            if format == "parquet":
                if file is None:
                    self.df.to_parquet("dataset.parquet")
                else:
                    self.df.to_parquet(file)
            elif format == "excel":
                if file is None:
                    self.df.to_excel("dataset.xlsx", sheet_name="Sheet1")
                else:
                    self.df.to_excel(file, sheet_name="Sheet1")
            else:
                if file is None:
                    self.df.to_csv("dataset.csv")
                else:
                    self.df.to_csv(file)

    def normalize(self, column, newColumnSuffix="Normalized"):
        """Normalize a column to columnNormalized with parameters: mean = 0, standard deviation = 1

        Keyword arguments:
        column -- name of the column containing data to be normalized newColumnSuffix -- suffix appended to "column" in order to create the column containing normalized values
        """
        normColumn = column + newColumnSuffix
        stats = self.df.groupby("ticker")[column].agg(["mean", "std"])

        def normalizeValue(value, ticker):
            mean = stats.loc[ticker, "mean"]
            std = stats.loc[ticker, "std"]
            if std != 0 and not pandas.isna(value):
                return (value - mean) / std
            else:
                return pandas.NA

        self.df[normColumn] = self.df.apply(
            lambda row: normalizeValue(row[column], row["ticker"]), axis=1
        )

        return self.df

    # TODO: normalized over min-max method

    def corr(self, values, method="pearson", min=0, max=1, display=True):
        """Build the correlation matrix of the dataset (self.df) according to the column "values"

        Keyword arguments:
        values -- column to use in the computation of the correlation matrix
        method -- "pearson" = Expectation((X-mean(X))*(Y-mean(Y))) / (std(X)*std(Y)) = Cov(X,Y) / (std(X)*std(Y))
                  "kendall" = ((nb concordant pairs) - (nb discordent pairs)) / (nb pairs)
                  "spearman" = pearson(rank(X), rank(Y))
        min -- minimal absolute value of correlation required
        max -- maximal absolution value of correlation required (to remove overfitted data)
        display --- display correlation values
        """
        dfPivoted = self.df.pivot(index="date", columns="ticker", values=values)
        corrMatrix = dfPivoted.corr(method=method)
        corrMatrix = corrMatrix[["XAU"]].drop(index="XAU").T
        # print("\nCorrelation Matrix")
        # print(corrMatrix)

        row = corrMatrix.iloc[0]
        sortedCols = row.abs().sort_values(ascending=False).index
        corrMatrix = corrMatrix[sortedCols]
        # print("\nSorted Correlation Matrix")
        # print(corrMatrix)

        corrMatrix = corrMatrix.where(corrMatrix.abs() > min)
        corrMatrix = corrMatrix.where(corrMatrix.abs() < max)
        corrMatrix = corrMatrix.dropna(axis=1, how="all")
        if display:
            print(f"\nSorted Correlation Matrix > {min * 100:.0f}%")
            print(corrMatrix)
            for ticker in corrMatrix.columns:
                print(f"{ticker}: {corrMatrix.at['XAU', ticker]} correlated with XAU")

        return corrMatrix

    def toTimeSeries(self, column, exogenValue="XAU", ticker=None):
        """Prepare the dataset to be usable with a Time Series algorithm, i.e. with columns "ds" (as "date"), "unique_id" (as "ticker") and "y"

        column -- column used as the "y" column
        exogenValue -- a ticker used as an exogen value
        ticker -- limit the dataset to only one specific ticker
        """

        if ticker is None:
            dfTimeSeries = self.df
        else:
            dfTimeSeries = self.getTicker(ticker)[["date", "ticker", column]]

        if exogenValue is not None:
            dfExo = self.getTicker(exogenValue)[["date", column]].rename(
                columns={column: exogenValue}
            )
            dfTimeSeries = dfTimeSeries[dfTimeSeries["ticker"] != exogenValue]
            dfTimeSeries = pandas.merge(dfTimeSeries, dfExo, how="inner", on="date")

        dfTimeSeries = dfTimeSeries.rename(
            columns={"date": "ds", "ticker": "unique_id", column: "y"}
        )

        return dfTimeSeries

    def selectTickers(
        self, column="closeNormalized", minCorr=0.65, maxCorr=0.99, display=True
    ):
        """Select the most relevant tickers according to correlation matrices with gold price

        Keyword arguments:
        column --- column used to compare tickers
        minCorr --- minamal correlation wanted
        maxCorr --- maximal correlation wanted
        display --- display the correlation matrix and a plot if true
        """
        methods = ["pearson", "kendall", "spearman"]
        tickers = ["XAU"]

        corrMatrices = []
        for method in methods:
            corrMatrices.append(
                self.corr(
                    values=column,
                    method=method,
                    min=minCorr,
                    max=maxCorr,
                    display=display,
                )
            )
            if not corrMatrices[-1].empty:
                tickers.extend(corrMatrices[-1].T.index.tolist())
        if display:
            drawCorrelationMatricesAndPlot(
                corrMatrices, methods, minCorr, self.df, column
            )

        return tickers

    def dropTickers(self, keep):
        """Drop all the tickers that are not in the list

        Keyword arguments:
        keep -- list of tickers to keep
        """
        df = self.df

        df = df[df["ticker"].isin(keep)]

        return df

    def __getitem__(self, index):
        return self.df[index]

    def __setitem__(self, index, value):
        self.df[index] = value


# Define a random color list
colors = [
    "dimgrey",
    "rosybrown",
    "lightcoral",
    "red",
    "coral",
    "sienna",
    "chocolate",
    "darkorange",
    "tan",
    "khaki",
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
                f"Most correlated (>{minCorr * 100:.0f}%) wrt {methods[i].capitalize()}",
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
kaggleCsvFiles = [
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
]
