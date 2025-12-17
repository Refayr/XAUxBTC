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

== Dataset

=== Dataset structure
#figure(
  table(
    columns: 5,
    stroke: none,
    align: (left + horizon),
    //inset: 3pt,
    table.vline(x: 1, start: 0, end: 2, stroke: 0.5pt), table.header(
      [],
      [*date*],
      [*ticker*],
      [*close*],
      [*closeNormalized*],
    ),
    table.hline(stroke: 0.5pt),                         [index], [date], [currency short name], [price at closing time in \$],
    [normalized (by ticker) price],
  ),
  caption: "Dataset structure as a Pandas Dataframe",
)

=== Tickers

The following tickers have more than 80% of correlation (with respect to Pierson's method) with gold over the whole period:
+ PAXG: 0.9997041186031184 correlated with XAU
+ XAUt: 0.9991536823382102 correlated with XAU
+ TRX: 0.914206355226965 correlated with XAU
+ BTC: 0.9043753594253107 correlated with XAU
+ BNB: 0.871339510212942 correlated with XAU
+ LEO: 0.8616999596788716 correlated with XAU
+ GT: 0.8477381081302898 correlated with XAU
+ SUN: 0.847138790223619 correlated with XAU
+ OKB: 0.8155979268149238 correlated with XAU
+ XRP: 0.8146858576249392 correlated with XAU
