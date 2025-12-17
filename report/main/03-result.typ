#import "@preview/hei-synd-report:0.1.1": *
#import "/metadata.typ": *
#pagebreak()

= Result

#option-style(type: option.type)[
  This chapter defines the overall system architecture, core components, and interactions between different parts. The goal is to translate the specifications into a concrete, actionable blueprint for development.

  Key elements:
  - *Define the architecture* – What are the main components of the system, and how do they interact?
  - *Select technologies* – What programming languages, tools, or frameworks will be used? Why?
  - *Design data structures* – How will data be organized and processed?
  - *Create process flow* – What are the key steps or workflows in the system?
  - *Plan modularity & scalability* – How can the design adapt to future needs or extensions?
  - *Address constraints* – What design choices were made due to performance, security, or usability considerations?
]


#figure(
  table(
    columns: 7,
    stroke: none,
    align: (left + horizon),
    //inset: 3pt,
    table.vline(x: 1, start: 0, end: 2, stroke: 0.5pt), table.header(
      [*metric*],
      [*Naive*],
      [*HistoricAverage*],
      [*WindowAverage*],
      [*SeasonalNaive*],
      [*ARIMA*],
      [*SARIMA*],
    ),
    table.hline(stroke: 0.5pt),                         [MAE], [57.471655], [598.324263], [83.421234], [86.408374], [71.508333], [71.508333],
    table.hline(stroke: 0.5pt),                         [MSE], [4161.423763], [359019.774221], [7986.952859], [9092.011023], [5916.408276], [5916.408276],
  ),
  caption: "Baseline Model comparison",
)


From the baseline models, we found that ARIMA and SARIMA have the best performance although it is worse than Naive model. That's may because Naive model use yesterday's price as today's price 
and the flactuation for gold price between a short periode will only have small change. So we use ARIMA and SARIMA to do further analysis.

#figure(
  table(
    columns: 3,
    stroke: none,
    align: (left + horizon),
    //inset: 3pt,
    table.vline(x: 1, start: 0, end: 2, stroke: 0.5pt), table.header(
      [*Model*],
      [*MAE*],
      [*RMSE*],
    ),
    table.hline(stroke: 0.5pt),                         [ARIMA_TimeOnly], [74.022], [79.428],
    table.hline(stroke: 0.5pt),                         [SARIMA_TimeOnly], [74.022], [79.428],
    table.hline(stroke: 0.5pt),                         [ARIMA_Crypto+Time], [92.358], [101.845],
    table.hline(stroke: 0.5pt),                         [SARIMA_Crypto+Time], [117.592], [132.474],
  ),
  caption: "ARIMA/SARIMA comparison between different features",
)

#figure(
  image("../results/gold_prediction_comparison.png", width: 80%),
  caption: [Visuliazition ARIMA/SARIMA],
) 

Although from the table above, it seems crypto features do not improve the prediction, the error for models with time features is smaller than those with time 
features and crypto features, that because time-only model predicts near average. But when we look at the visuliazition for these models, we can see that time-model 
is basically a flat line, just like naive model. So it is not useful for trading prediction. And time+crypto model captures actual price movements, it has flactuations
like real market price on the graph, so we think time+crypto model si more valuable even if point estimate are slightly off and have bigger error.

And in time+crypto models, ARIMA is better than SARIMA. So we decided to use ARIMA and another complex model(deep learning) to do further analysis.


#figure(
  table(
    columns: 4,
    stroke: none,
    align: (left + horizon),
    //inset: 3pt,
    table.vline(x: 1, start: 0, end: 2, stroke: 0.5pt), table.header(
      [*Model*],
      [*MAE*],
      [*RMSE*],
      [*$R^2$*],
    ),
    table.hline(stroke: 0.5pt),                         [ARIMA_TimeOnly], [45.66], [61.00], [-0.358],
    table.hline(stroke: 0.5pt),                         [LSTM_TimeOnly], [669.21], [671.32], [-163.430],
    table.hline(stroke: 0.5pt),                         [Ensemble_Avg_TimeOnly], [342.73], [347.25], [-42.995],
    table.hline(stroke: 0.5pt),                         [Ensemble_Weighted_TimeOnly], [61.45], [82.27], [-1.470],
  ),
  caption: "Time-Only Dataset Result",
)


#figure(
  table(
    columns: 4,
    stroke: none,
    align: (left + horizon),
    //inset: 3pt,
    table.vline(x: 1, start: 0, end: 2, stroke: 0.5pt), table.header(
      [*Model*],
      [*MAE*],
      [*RMSE*],
      [*$R^2$*],
    ),
    table.hline(stroke: 0.5pt),                         [ARIMA_Time+Crypto], [47.48], [58.48], [-0.248],
    table.hline(stroke: 0.5pt),                         [LSTM_Time+Crypto], [90.84], [112.25], [-3.597],
    table.hline(stroke: 0.5pt),                         [Ensemble_Avg_Time+Crypto], [55.13], [74.46], [-1.023],
    table.hline(stroke: 0.5pt),                         [Ensemble_Weighted_Time+Crypto], [47.33], [65.94], [-0.586],
  ),
  caption: "Time + Crypto Dataset Result",
)



#figure(
  image("../results/ensemble_time_only.png", width: 80%),
  caption: [Visuliazition time-only models],
) 

#figure(
  image("../results/ensemble_time_crypto.png", width: 80%),
  caption: [Visuliazition time+crypto models],
)

Overall, all of our models have negative $R^2$, which means our model are all failing. But we believe this is normal, because market trading prices themselves 
cannot be predicted by our simple models. However, our goal is to find out if crypto features can help with prediction. The models here are ARIMA, deep 
laearning and the model combined by them. Same as before the visuliazition for time-only models are all like flat line, but the models with time and crypto prices 
exhibit fluctuations similar to those in the real market. So our conclusion is crpto will improve the model.

We adjust test period for ARIMA in combined model comparision(from 30 days to 60 days), and the performance improved significantly.