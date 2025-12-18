#import "@preview/hei-synd-report:0.1.1": *
#import "/metadata.typ": *
#pagebreak()

= Result

#figure(
  table(
    columns: 7,
    stroke: none,
    align: (left + horizon),
    //inset: 3pt,
    table.vline(x: 1, start: 0, end: 3, stroke: 0.5pt), table.header(
      [*metric*],
      [*Naive*],
      [*HistoricAverage*],
      [*WindowAverage*],
      [*SeasonalNaive*],
      [*ARIMA*],
      [*SARIMA*],
    ),
    table.hline(stroke: 0.5pt),                         [MAE],                      [57.471655], [598.324263],  [83.421234],     [86.408374],   [71.508333],
    [71.508333],                                        table.hline(stroke: 0.5pt), [MSE],       [4161.423763], [359019.774221], [7986.952859], [9092.011023],
    [5916.408276],                                      [5916.408276],
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
    table.vline(x: 1, start: 0, end: 5, stroke: 0.5pt), table.header(
      [*Model*],
      [*MAE*],
      [*RMSE*],
    ),
    table.hline(stroke: 0.5pt),                         [ARIMA_TimeOnly],           [74.022],
    [79.428],                                           table.hline(stroke: 0.5pt), [SARIMA_TimeOnly],
    [74.022],                                           [79.428],                   table.hline(stroke: 0.5pt),
    [ARIMA_Crypto+Time],                                [92.358],                   [101.845],
    table.hline(stroke: 0.5pt),                         [SARIMA_Crypto+Time],       [117.592],
    [132.474],
  ),
  caption: "ARIMA/SARIMA comparison between different features",
)

#figure(
  image("/resources/img/gold_prediction_comparison.png", width: 80%),
  caption: [Visuliazition ARIMA/SARIMA],
)

Although from the table above, it seems cypto features do not improve the prediction, the error for models with time features is smaller than those with time
features and cypto features, that because time-only model predicts near average. But when we look at the visuliazition for these models, we can see that time-model
is basically a flat line, just like naive model. So it is not useful for trading prediction. And time+cypto model captures actual price movements, it has flactuations
like real market price on the graph, so we think time+cypto model si more valuable even if point estimate are slightly off and have bigger error.

And in time+cypto models, ARIMA is better than SARIMA. So we decided to use ARIMA and another complex model(deep learning) to do further analysis.

#figure(
  table(
    columns: 4,
    stroke: none,
    align: (left + horizon),
    //inset: 3pt,
    table.vline(x: 1, start: 0, end: 5, stroke: 0.5pt), table.header(
      [*Model*],
      [*MAE*],
      [*RMSE*],
      [*$R^2$*],
    ),
    table.hline(stroke: 0.5pt),                         [ARIMA_TimeOnly],           [45.66],                    [61.00],
    [-0.358],                                           table.hline(stroke: 0.5pt), [LSTM_TimeOnly],            [669.21],
    [671.32],                                           [-163.430],                 table.hline(stroke: 0.5pt), [Ensemble_Avg_TimeOnly],
    [342.73],                                           [347.25],                   [-42.995],                  table.hline(stroke: 0.5pt),
    [Ensemble_Weighted_TimeOnly],                       [61.45],                    [82.27],                    [-1.470],
  ),
  caption: "Time-Only Dataset Result",
)

#figure(
  table(
    columns: 4,
    stroke: none,
    align: (left + horizon),
    //inset: 3pt,
    table.vline(x: 1, start: 0, end: 5, stroke: 0.5pt), table.header(
      [*Model*],
      [*MAE*],
      [*RMSE*],
      [*$R^2$*],
    ),
    table.hline(stroke: 0.5pt),                         [ARIMA_TimeOnly],           [47.48],                    [58.48],
    [-0.248],                                           table.hline(stroke: 0.5pt), [LSTM_TimeOnly],            [90.84],
    [112.25],                                           [-3.597],                   table.hline(stroke: 0.5pt), [Ensemble_Avg_TimeOnly],
    [55.13],                                            [74.46],                    [-1.023],                   table.hline(stroke: 0.5pt),
    [Ensemble_Weighted_TimeOnly],                       [47.33],                    [65.94],                    [-0.586],
  ),
  caption: "Time + Cypto Dataset Result",
)

#figure(
  image("/resources/img/ensemble_time_only.png", width: 80%),
  caption: [Visuliazition time-only models],
)

#figure(
  image("/resources/img/ensemble_time_crypto.png", width: 80%),
  caption: [Visuliazition time+crypto models],
)

The models here are ARIMA, deep laearning and the model combined by them. Same as before the visuliazition for time-only models are all like flat line. So not useful for
market prediction. But in time+cypto model, all models performs better.
