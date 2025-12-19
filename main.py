import os

from dataset import *
from generate_modeldata import *
from combined_model import *


# ========================================
# PART 1: Preprocess Data
# ========================================
print("\n" + "=" * 70)
print("Downloading and preparing data")
print("=" * 70)
data = Dataset(columns={"ticker", "date", "close"})
data.setDateFormat("yyyy-mm-dd")

localFile = "model_datasets/dataset.csv"
if os.path.isfile(localFile):
    data.addDataset(source=DatasetProvider.CSV, file=localFile)
else:
    for csvFile in kaggleCsvFiles:
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
if not os.path.isfile(localFile):
    data.exportDataset("csv")

# First ticker selection to know the date range
tickers = data.selectTickers(
    column="closeNormalized", minCorr=0.65, maxCorr=0.99, display=False
)
# Keep unique ticker names
tickers = list(set(tickers))

data.df = data.dropTickers(keep=tickers)
data.trimDates()
# Remove all data after the hole from 2024-12-10 to 2025-01-26
data.end = "2024-12-09"
data.df = data.df[data.df["date"] <= data.end]

# Second ticker selection with the new date range
data.normalize("close")
tickers = data.selectTickers(column="closeNormalized", minCorr=0.65, maxCorr=0.99)
tickers = list(set(tickers))
data.df = data.dropTickers(keep=tickers)

data.exportDataset("csv", "model_datasets/dataset_reduced.csv")
print("\n" + "=" * 70)
print(
    "Kept only " + str(len(tickers)) + " tickers over " + str(len(kaggleCsvFiles)) + ":"
)
print(tickers)
print("=" * 70)

dfCV = data.toTimeSeries(column="closeNormalized", exogenValue="XAU")
dfCV.to_csv("model_datasets/timeseries.csv")


# ========================================
# PART 2: Generate data models
# ========================================
generate_modeldata()


# ========================================
# PART 3: Training / Testing
# ========================================
combined_model()
