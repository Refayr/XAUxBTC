#import "@preview/hei-synd-report:0.1.1": *
#import "/metadata.typ": *
#pagebreak()

= Implementation

#option-style(type:option.type)[
  This chapter details the development and execution of the project. It describes actual implementation from a top-down or bottom-up approach depending on the project.

  Key elements:
  - *Describe the implementation* – How was the project developed?
  - *Explain technical decisions* – What tools, frameworks, or methods were used and why?
  - *Highlight key components* – What are the most important parts of the implementation?
  - *Address challenges* – What difficulties arose, and how were they solved?
]

#lorem(200)
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

