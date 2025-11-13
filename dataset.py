#####################################################
##### Class ordering all datasets in one usable #####
#####################################################
# python3 -m pip install kagglehub yfinance pandas
import kagglehub
from kagglehub import KaggleDatasetAdapter
import pandas

import yfinance
import enum


class DatasetProvider(enum.Enum):
    CSV = 1
    KAGGLE = 1 << 1
    YAHOO = 1 << 2


class Dataset:
    def __init__(self, columns=None, trim=False):
        self.columns = columns
        self.trim = trim
        self.dateFormat = None
        # self.df = pandas.DataFrame(data=None, index=None, columns=None, dtype=None, copy=None)
        self.df = None
        self.begin = -1
        self.end = -1

    def setDateFormat(self, dateFormat="yyyy-mm-dd"):
        self.dateFormat = dateFormat

    def getDateFormat(self):
        return self.dateFormat

    def cleanColumns(self, df, ticker):
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

    def updateDateRange(self, df):
        dfBegin = df["date"].iloc[0]
        dfEnd = df["date"].iloc[-1]

        def updateBound(current, new, isLowerBound):
            if current == -1:
                return new
            if (isLowerBound and new > current) or (not isLowerBound and new < current):
                return new
            else:
                return current

        self.begin = updateBound(self.begin, dfBegin, isLowerBound=True)
        self.end = updateBound(self.end, dfEnd, isLowerBound=False)

    def _importFromYahoo(self, ticker, start=-1, end=-1):
        if start == -1 or end == -1:
            df = yfinance.download(ticker, period="max")
        else:
            df = yfinance.download(ticker, start=start, end=end)

        df = df.stack().reset_index().rename(columns={"index": "date"})

        return df

    def _importFromKaggle(self, repo, file):
        # Download latest dataset from Kaggle
        return kagglehub.load_dataset(
            KaggleDatasetAdapter.PANDAS,
            repo,
            file,
            # Provide any additional arguments
            # See the documenation for more information:
            # https://github.com/Kaggle/kagglehub/blob/main/README.md#kaggledatasetadapterpandas
        )

    def _importFromCsv(self, file):
        # TODO: import file from hard drive
        pass

    def addDataset(self, source, file, repo=None, ticker=None):
        match source:
            case DatasetProvider.KAGGLE:
                # Download latest dataset from Kaggle
                dfTmp = self._importFromKaggle(repo, file)
            case DatasetProvider.YAHOO:
                # Download latest dataset from Yahoo Finance
                dfTmp = self._importFromYahoo(file, self.begin, self.end)
            case DatasetProvider.CSV:
                # Use a local .csv file
                dfTmp = self._importFromCsv(file)
            case _:
                # TODO: raise an exception
                pass

        # TODO: take care if self.columns == None
        dfTmp = self.cleanColumns(dfTmp, ticker)

        # Sort values by dates
        dfTmp = dfTmp.sort_values(by="date")

        # Limit to a common range of dates
        self.updateDateRange(dfTmp)

        if self.df is None:
            self.df = dfTmp
        else:
            self.df = pandas.concat([self.df, dfTmp])

        # Drop values out of date range
        if self.trim:
            self.df = self.df[
                (self.df["date"] >= self.begin) & (self.df["date"] <= self.end)
            ]
            self.df = self.df.dropna()

        return self.df

    def getTicker(self, ticker=""):
        if ticker == "":
            return self.df
        else:
            return self.df[self.df["ticker"] == ticker]

    def exportDataset(self, format="csv"):
        if self.df is not None:
            if format == "parquet":
                self.df.to_parquet("dataset.parquet")
            elif format == "excel":
                self.df.to_excel("dataset.xlsx", sheet_name="Sheet1")
            else:
                self.df.to_csv("dataset.csv")

    # Create a new column base on the normalization of another column based on a ticker
    def normalize(self, column):
        normColumn = column + "Normalized"
        stats = self.df.groupby("ticker")[column].agg(["mean", "std"])

        def normalizeGroup(group):
            ticker = group.name
            mean = stats.loc[ticker, "mean"]
            std = stats.loc[ticker, "std"]
            if std != 0:
                return (group[column] - mean) / std
            else:
                return pandas.NA

        self.df[normColumn] = self.df.groupby("ticker")[column].transform(
            lambda x: (x - stats.loc[x.name, "mean"]) / stats.loc[x.name, "std"]
            if stats.loc[x.name, "std"] != 0
            else pandas.NA
        )

        # tickers = self.df["ticker"].unique()

        # for ticker in tickers:
        #    mask = self.df["ticker"] == ticker
        #    dfTmp = self.df[mask][column].dropna()
        #    if len(dfTmp) > 0:
        #        mean = dfTmp.mean()
        #        std = dfTmp.std()
        #        self.df.loc[mask, normColumn] = (self.df.loc[mask, column] - mean) / std
        #    else:
        #        self.df.loc[mask, normColumn] = pandas.NA

        return self.df

    def __getitem__(self, index):
        return self.df[index]

    def __setitem__(self, index, value):
        self.df[index] = value
