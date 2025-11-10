// common CONFIG  : #set
#set document(
  title: "Python Project",
)
/*#show title: set text(size: 17pt)
#show title: set align(center)
#show title: set block(below: 1.2em)*/

#set page(
  paper: "a4",
  margin: (x: 1.0cm, y: 1.0cm),
  /*header: align(
    right + horizon,
    context document.title,
  ),*/
  //numbering: "1 / 1"
  numbering: "1",
)

// display numbers of pages :
#counter(page).update(1)

#set text(
  font: "New Computer Modern",
  size: 11pt,
)

// style of numbering of headers :
#set heading(numbering: "I. 1. A. a.")

//#show heading: it => {
//  if (it.level == 1) {
//    block(it.body)
//  } else {
//    block(counter(heading).display() + " " + it.body)
//  }
//}

// style of numbering of  enum : lists with numbers :
#set enum(numbering: "1.a)")


// BODY TEXT  :
//#title[Python Project]
= Introduction
Gold has been used as currency since the 6#super[th] century BC and has played an important role throughout world history. Especially after the establishment of the gold standard, gold becomes the basis for maintaining the world monetary system. Later, although the gold standard system was gradually abolished in the 20#super[th] century due to the Great Depression, gold still occupies an important position in the modern economy. Conversely, cryptocurrencies are much more recent and have been launched in the 21#super[th] century on the Blockchain.
Even if the gold standard system has been abolished, does any correlation between gold and cryptocurrencies could be found? An analysis on several period of time in order to find out one (or more) year when gold stop (or begin) to act as a standard for cryptocurrencies. Our research will aim in a procedure to find it out by defining objectives, enumerating methods needed and the toolkit compulsory to compute data.

= Aims and Objectives
== Aims: We want to find out a correlation between gold and any cryptocurrency exchange rates.
+ Objective 1: Compare the exchange rates, find out if there is a comparable period of time when the cryptocurrency and gold are correlated.
+ Objective 2: Identify some ranges which contain (or do not) periods of correlation between gold and cryptocurrencies.

= Methods
Data we will use is:
+ Date of observation
+ Name if currency
+ Currency/gold price at open market, close market and average day price
Additionally we want to find out other parameters (mean, variance, etc.) in order to compare the different exchanges using mathematical tools.

= Toolkit
We will use some libraries for doing mathematical computations and to draw graphs (eg. Matplotlib).
== Libraries used
+ kagglehub (for keeping using the latest datasets)
+ pandas (to manipulate the dataframes)
+ matplotlib
== Datasets used
+ #link(
    "https://www.kaggle.com/datasets/svaningelgem/crypto-currencies-daily-prices/data?select=BTC.csv",
  )[Crypto cryptocurrencies daily prices]
+ #link(
    "https://www.kaggle.com/datasets/isaaclopgu/gold-historical-data-daily-updated",
  )[Gold historical data daily updated]
=== Dataset modifications
+ Dates before 17/07/2010 must be droped as no cryptocurrency were available before
+ Columns keeped :
  Every columns from the gold dataset must be renamed with a lowercase name.
  - ticker: "XAU" value added to gold exchange rates
  - date: date format must be converted to yyyy-mm-dd for gold exchange rates dataset
  - open
  - high
  - low
  - close
== Usefull tools
+ Git
  - git add newFile
  - git commit -a -m "comment"
  - git push -u origin main
  - git pull git\@github.com:user/repository.git
