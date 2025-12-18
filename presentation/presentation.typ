#import "@preview/typographix-polytechnique-slides:0.2.0" as template

/// Available options:
/// - ratio: a float, typically 16/9 or 4/3
/// - h1-theme: the theme for section title slides, one of
///   - "light"
///   - "light-dark": light with a dark frame
///   - "dark"
///   - "dark-light": dark with a light frame
///  - frame-theme: either "light" or "dark".
/// More information and previews on the README:
/// https://github.com/remigerme/typst-polytechnique/tree/main/slide
#show: template.apply.with(ratio: 16 / 9, h1-theme: "light", frame-theme: "light")

/// Cover page (optional), available options:
/// - title
/// - speaker
/// - date
/// - theme: "light" or "dark"
/// - background-image: should be `image("path/to/image.png", width: 100%, height: 100%)` if provided


#template.cover(
  title: "Gold and cryptocurrencies exchange rates correlation",
  speaker: [
    #grid(
      columns: (1fr, auto),
      [Zhousiying Wu, Nicolas Valety],
      [
        #box[
          #image("../report/resources/img/goldcrypto.png", width: 150pt)
          #place(top + right, dx: 5pt, dy: -5pt)[
            #text(size: 100pt, fill: red)[?]
          ]
        ]
      ]
    )
  ],
  date: "19/12/2025",
  theme: "light",
)


#outline(title: [
  #grid(
    columns: (1fr, auto),
    align: (left, right),
    "Content",
    #pause
    image("../resources/img/goldcrypto.png", width: 50pt)
  )
])

#outline(title: "Content")

= Method

== Method

- Tickers with >65% of correlation with gold (from Kaggle and Yahoo! Finance)
- Date ranges from 16/05/2022 to 09/12/2024 (hole in dataset from 10/12/2024)
- Tickers are exogenous data for gold time series

#v(1cm)
Un espace vertical pour aérer le tout. Et maintenant une grille avec deux éléments (ici, des tableaux):

#grid(
  columns: (1fr, 1fr),
  align: horizon + center,
  table(
    columns: (2fr, 1fr, 1fr),
    inset: 20pt,
    [*Volume horaire*],     [*Fun*], [*Ennui*],
    [au moins 3h par jour], [oui],   [non],
    [au moins 7h par jour], table.cell(colspan: 2, rect(fill: red, width: 100%)),
  ), table(
    stroke: (x, y) => if y == 0 { (bottom: 1pt) } else { none },
    [*Top cinq des gares*],
    [L'Argentière-la-Bessée],
    [Paris Gare de Lyon],
    [Cassis],
    [Saint-Pierre des Corps],
    [Montparnasse],
  ),
)

= Result

== Result
#grid(
  columns: (1fr, 1fr),
  rows: (auto, auto, auto),
  gutter: 10pt,
  align: (center, center),
  
  // Row 1: Tables
  [
    #figure(
      text(size: 8pt)[
        #table(
          columns: 4,
          stroke: none,
          align: (left + horizon),
          table.vline(x: 1, start: 0, end: 5, stroke: 0.5pt), 
          table.header([*Model*], [*MAE*], [*RMSE*], [*$R^2$*]),
          table.hline(stroke: 0.5pt),                         
          [ARIMA_TimeOnly], [45.66], [61.00], [-0.358],                                           
          table.hline(stroke: 0.5pt), 
          [LSTM_TimeOnly], [540.974], [543.786], [-106.889],                 
          table.hline(stroke: 0.5pt), 
          [Ensemble_Avg_TimeOnly], [278.607], [284.345], [-28.499],                  
          table.hline(stroke: 0.5pt),
          [Ensemble_Weighted_TimeOnly], [60.585], [81.713], [-1.436],
        )
      ],
      caption: text(size: 9pt)["Time-Only Dataset Result"],
    )
  ],
  [
    #figure(
      text(size: 8pt)[
        #table(
          columns: 4,
          stroke: none,
          align: (left + horizon),
          table.vline(x: 1, start: 0, end: 5, stroke: 0.5pt), 
          table.header([*Model*], [*MAE*], [*RMSE*], [*$R^2$*]),
          table.hline(stroke: 0.5pt),                         
          [ARIMA_Time+Crypto], [47.485], [58.477], [-0.248],                                           
          table.hline(stroke: 0.5pt), 
          [LSTM_Time+Crypto], [59.376], [70.860], [0.0239],                   
          table.hline(stroke: 0.5pt), 
          [Ensemble_Avg_Time+Crypto], [43.320], [57.908], [-0.223],                   
          table.hline(stroke: 0.5pt),
          [Ensemble_Weighted_Time+Crypto], [42.270], [57.227], [-0.195],
        )
      ],
      caption: text(size: 9pt)["Time + Crypto Dataset Result"],
    )
  ],
  
  // Row 2: Images  
  [
    #figure(
      image("/img/ensemble_time_only.png", width: 90%),
      caption: text(size: 9pt)["Time-only models visualization"],
    )
  ],
  [
    #figure(
      image("/img/ensemble_time_crypto.png", width: 90%),
      caption: text(size: 9pt)["Time+crypto models visualization"],
    )
  ]
)

== Conclusion

#align(center + horizon, text(size: 40pt, "Correlation but hard to predict gold exchange rate."))

#align(center, text(fill: template.PALETTE.gold, "Question time!"))
