#####################################################
##### Class ordering all datasets in one usable #####
#####################################################
import kagglehub
from kagglehub import KaggleDatasetAdapter
import pandas


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

        def shouldDrop(col):
            return col.lower() not in referenceLower

        def shouldRename(col):
            return col.lower() != col and col.lower() in referenceLower

        drop = [col for col in df.columns if shouldDrop(col)]
        rename = {col: col.lower() for col in df.columns if shouldRename(col)}

        df = df.drop(columns=drop).rename(columns=rename)

        df["date"] = df["date"].astype(str).str[:10]
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

    def addDataset(self, repo, file, ticker=None):
        # Download latest dataset from Kaggle
        dfTmp = kagglehub.load_dataset(
            KaggleDatasetAdapter.PANDAS,
            repo,
            file,
            # Provide any additional arguments
            # See the documenation for more information:
            # https://github.com/Kaggle/kagglehub/blob/main/README.md#kaggledatasetadapterpandas
        )

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

    def __getitem__(self, index):
        return self.df[index]

    def __setitem__(self, index, value):
        self.df[index] = value
