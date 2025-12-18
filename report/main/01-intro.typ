#import "@preview/hei-synd-report:0.1.1": *
#import "/metadata.typ": *
#pagebreak()

= Introduction

== Context
Gold has been used as currency since the 6#super[th] century BC and has played an important role throughout world history. Especially after the establishment of the gold standard, gold becomes the basis for maintaining the world monetary system. Later, although the gold standard system was gradually abolished in the 20#super[th] century due to the Great Depression, gold still occupies an important position in the modern economy. Conversely, cryptocurrencies are much more recent and have been launched in the 21#super[th] century on the Blockchain.
Even if the gold standard system has been abolished, does any correlation between gold and cryptocurrencies could be found? An analysis on several period of time in order to find out one (or more) year when gold stop (or begin) to act as a standard for cryptocurrencies. Our research will aim in a procedure to find it out by defining objectives, enumerating methods needed and the toolkit compulsory to compute data.

== Aims and Objectives

=== Aims: We want to find out a correlation between gold and any cryptocurrency exchange rates.
#ideabox()[At this point, there may not be any correlation between both exchanges rates as one is very stable and the other is very volatile.]
+ Objective 1: Compare the exchange rates, find out if there is a comparable period of time when the cryptocurrency and gold are correlated.
+ Objective 2: Identify some ranges which contain (or do not) periods of correlation between gold and cryptocurrencies.

== Implementation

- Naive model: This is chosen as a baseline, the logic here is tomorrow's price = today's price. Because gold prices fluctuate very little in the short term,
I think this is a good benchmark.
- ARIMA: Because gold prices tend to revert to the mean in the long run, prices fluctuate over time, and past prices influence future prices. ARIMA can capture
the linear trend and autocorrelation of gold price changes.
- SARIMA: SARIMA and ARIMA operate on similar principles, but SARIMA incorporates seasonal patterns. Because gold prices are linked to seasonal demand
(e.g., as holiday gifts or wedding jewelry) and cultural patterns (e.g., the Indian wedding season and Chinese New Year), we believe economic cycles may exist.
Additionally, the seasons in mining areas also affect production, thus influencing prices.
- Deep learning :

Evaluation:
Naive model:
ARIMA:
SARIMA:

Then we add exgeous features to improve the predict, the exgeous feature we generated are
- day of week (0:monday - 6:sunday), because gold trading might have weekly pattern
- which week in the year(1-52), because gold jewelry trading might increase in certain months (e.g.chinese new year, indian wedding season)
- month of year(1-12), days, months, years
- month since start: As the data is started from 2022-05-16 to 2024-12-09, so this is how many months have passed since our data is started. Because it is obvious that gold price
have long-term upward trend. And models gradual inflation effects over time.
- if that day is a holiday / how many days to holiday: Since markets are closed for holidays and there is no trading data, we use the average price before and after the holidays to fill
the gap. Additionally, many investors adjust their positions before holidays, which also contributes to gold price fluctuations. The uncertainty during major
holiday closures also affects gold prices.
