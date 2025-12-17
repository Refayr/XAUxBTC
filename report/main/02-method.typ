#import "@preview/hei-synd-report:0.1.1": *
#import "/metadata.typ": *
#pagebreak()

= Method

#option-style(type: option.type)[
  This chapter defines the project requirements and presents the theoretical background. If applicable, it should also include a *State of the art* review to compare existing solutions and justify the chosen approach.

  Key elements:
  - *Define the requirements* – What must the system/process be able to do?
  - *Describe the architecture* – What are the key components of the solution?
  - *Review existing solutions* – What approaches already exist? How does yours compare?
  - *Explain design choices* – Why did you choose this approach? What alternatives were considered?
]

== Requirements

=== Dataset restrictions
- Due to the high number of tickers, we have only used the crytos with at least 65% of correlation with gold (whatever the method used).
- There is a hole in the dataset downloaded from Kaggle: no data are available between the December 10th 2024 and January 26th 2025. So the data analysis will occur from the first common date with all the interesting tickers and December 9th 2024.
