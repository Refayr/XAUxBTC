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
  speaker: "Zhousiying Wu, Nicolas Valety",
  date: "19/19/2025",
  theme: "light",
)

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
Mon dieu, qu'ai-je fait ?

== Conclusion

#align(center + horizon, text(size: 40pt, "Correlation but hard to predict gold exchange rate."))

#align(center, text(fill: template.PALETTE.gold, "Question time!"))
