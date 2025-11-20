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
    def __init__(self, columns, trim=False):
        self.columns = columns
        self.trim = trim
        self.dateFormat = None
        # self.df = pandas.DataFrame(data=None, index=None, columns=None, dtype=None, copy=None)
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

    def updateDateRange(self, df):
        dfStart = df["date"].iloc[0]
        dfEnd = df["date"].iloc[-1]

        def updateBound(current, new, isLowerBound):
            if current == -1:
                return new
            if (isLowerBound and new > current) or (not isLowerBound and new < current):
                return new
            else:
                return current

        self.start = updateBound(self.start, dfStart, isLowerBound=True)
        self.end = updateBound(self.end, dfEnd, isLowerBound=False)

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

        # Limit to a common range of dates
        self.updateDateRange(dfTmp)

        if self.df is None:
            self.df = dfTmp
        else:
            self.df = pandas.concat([self.df, dfTmp])

        # Drop values out of date range
        if self.trim:
            self.df = self.df[
                (self.df["date"] >= self.start) & (self.df["date"] <= self.end)
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

    def normalize(self, column, newColumnSuffix="Normalized"):
        """Normalize a column to columnNormalized with parameters: mean = 0, standard deviation = 1

        Keyword arguments:
        column -- name of the column containing data to be normalized
        newColumnSuffix -- suffix appended to "column" in order to create the column containing normalized values
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

    def __getitem__(self, index):
        return self.df[index]

    def __setitem__(self, index, value):
        self.df[index] = value
