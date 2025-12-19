#import "@preview/hei-synd-report:0.1.1": *
#import "/metadata.typ": *
#pagebreak()

= Introduction

== Context
In an era dominated by AI models, predicting trading markets has become a goal for many. Our project explores the relationship
between cryptocurrency and gold prices, attempting to answer whether cryptocurrency can help predict gold prices.

The reason why we chose that is Gold has been used as currency since the 6#super[th] century BC and has played an important
role throughout world history. Especially after the establishment of the gold standard, gold becomes the basis for
maintaining the world monetary system. Later, although the gold standard system was gradually abolished in the 20#super[th]
century due to the Great Depression, gold still occupies an important position in the modern economy. Conversely, cryptocurrencies
are much more recent and have been launched in the 21#super[th] century on the Blockchain. So we guess there might some relation
between them.

Even if the gold standard system has been abolished, does any correlation between gold and cryptocurrencies could be found? An analysis on several period of time in order to find out one (or more) year when gold stop (or begin) to act as a standard for cryptocurrencies. Our research will aim in a procedure to find it out by defining objectives, enumerating methods needed and the toolkit compulsory to compute data.

== Aims and Objectives

=== Aims: We want to find out a correlation between gold and any cryptocurrency exchange rates.
#ideabox()[At this point, there may not be any correlation between both exchanges rates as one is very stable and the other is very volatile.]
+ Objective 1: Compare the exchange rates, find out if there is a comparable period of time when the cryptocurrency and gold are correlated.
+ Objective 2: Identify some ranges which contain (or do not) periods of correlation between gold and cryptocurrencies.

== Starting point
By launching the `main.py` file, you will be able to reproduce all the results explained in the following chapters.
