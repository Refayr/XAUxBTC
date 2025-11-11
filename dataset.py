#####################################################
##### Class ordering all datasets in one usable #####
#####################################################
import kagglehub
from kagglehub import KaggleDatasetAdapter
import pandas


class Dataset:
    def __init__(self, columns=None):
        self.columns = columns
        self.dateFormat = None
        # self.df = pandas.DataFrame(data=None, index=None, columns=None, dtype=None, copy=None)
        self.df = None
        self.begin = -1
        self.end = -1

    def setDateFormat(self, dateFormat="yyyy-mm-dd"):
        self.dateFormat = dateFormat

    def getDateFormat(self):
        return self.dateFormat

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
        columns = dfTmp.columns
        rename = {}
        drop = []
        for column in columns:
            lower = column.lower()
            delete = True
            for inCols in self.columns:
                if lower == inCols:
                    delete = False
            if delete:
                drop.append(column)
            elif lower != column:
                rename[column] = lower

        # dfTmp = dfTmp.rename(
        #    columns={
        #        "Date": "date",
        #        "Open": "open",
        #        "High": "high",
        #        "Low": "low",
        #        "Close": "close",
        #    }
        # )
        dfTmp = dfTmp.drop(columns=drop)
        dfTmp = dfTmp.rename(columns=rename)
        dfTmp["date"] = dfTmp["date"].astype(str).str[:10]
        dfTmp["date"] = pandas.to_datetime(dfTmp["date"])
        if ticker is not None:
            dfTmp["ticker"] = ticker

        # Sort values by dates
        dfTmp = dfTmp.sort_values(by="date")

        # Limit to a common range of dates
        dfBegin = dfTmp["date"].iloc[0]
        dfEnd = dfTmp["date"].iloc[-1]
        if self.begin == -1:
            self.begin = dfBegin
        elif dfBegin > self.begin:
            self.begin = dfBegin
        if self.end == -1:
            self.end = dfEnd
        elif dfEnd < self.end:
            self.end = dfEnd

        if self.df is None:
            self.df = dfTmp
        else:
            self.df = pandas.concat([self.df, dfTmp])

        # Drop values out of date range
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
