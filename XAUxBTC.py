import kagglehub
from kagglehub import KaggleDatasetAdapter
import pandas


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


# Sort values by dates
dfXAU.sort_values(by="Date")
dfBTC.sort_values(by="date")


# Get the first common date
firstDateGold = dfXAU["Date"].iloc[0]
firstDateCrypto = dfBTC["date"].iloc[0]

if firstDateGold > firstDateCrypto:
    firstDate = firstDateGold
else:
    firstDate = firstDateCrypto
print(firstDate)


# Get the last common date
lastDateGold = dfXAU["Date"].iloc[-1]
lastDateCrypto = dfBTC["date"].iloc[-1]

if lastDateGold < lastDateCrypto:
    lastDate = lastDateGold
else:
    lastDate = lastDateCrypto
print(lastDate)


# Print the columns
print(dfXAU.columns)
print(dfBTC.columns)
print(dfXAU.head)
print(dfBTC.head)

# Create a new dataframe with only common columns
df = pandas.DataFrame(data=None, index=None, columns=None, dtype=None, copy=None)
