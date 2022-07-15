# Network-Analysis-R

# [A]. Graph Centrality Measures



## References


# [B]. Adjacency Matrices

There are various R packages that allow to perform Network Analysis and simulate Adjacency matrices. Some examples are provided below. 

```R

# Install R package
install.packages("network")
library(network)

# Examples of constructing Adjacency Matrices
num_nodes   <- 10
sociomatrix <- matrix(round(runif(num_nodes*num_nodes)), nrow = num_nodes, ncol = num_nodes)

> sociomatrix

      [,1] [,2] [,3] [,4] [,5] [,6] [,7] [,8] [,9] [,10]
 [1,]    0    0    1    1    0    0    1    1    1     1
 [2,]    1    0    1    0    0    0    0    0    0     1
 [3,]    1    0    0    1    0    0    0    0    1     1
 [4,]    1    1    1    0    0    1    0    1    1     1
 [5,]    1    0    1    1    0    0    1    1    0     0
 [6,]    1    1    0    1    1    1    0    0    1     0
 [7,]    1    0    0    0    1    0    0    0    0     0
 [8,]    1    1    1    1    0    0    1    0    0     1
 [9,]    1    0    0    0    0    1    0    0    0     0
[10,]    0    1    1    0    0    1    1    0    1     1

```

# Example 1

Using the R package ['NetworkRiskMeasures'](https://cran.r-project.org/web/packages/NetworkRiskMeasures/index.html) simulate a set of variables that represent 

```R

# Install R packages
install.packages("NetworkRiskMeasures")
library(NetworkRiskMeasuress)

# SIMULATING A BANKING NETWORK#
set.seed(1234)

# Heavy tailed assets
assets <- rlnorm(125, 0, 2)
assets[assets < 4] <- runif(length(assets[assets < 4]))

# Heavy tailed liabilities
liabilities <- rlnorm(125, 0, 2)
liabilities[liabilities < 4] <- runif(length(liabilities[liabilities < 4]))

# Making sure assets = liabilities
assets <- sum(liabilities) * (assets/sum(assets))

# Buffer as a function of assets
buffer <- pmax(0.01, runif(length(liabilities))*liabilities + abs(rnorm(125, 4, 2.6)))

# Weights as a function of assets, buffer and liabilities
weights <- (assets + liabilities + buffer + 1) + rlnorm(125, 0, 1)

# creating data.frame
sim_data <- data.frame(bank=paste0("b", 1:125),assets=assets,liabilities=liabilities,buffer=buffer,weights=weights)

> sim_data
    bank      assets  liabilities     buffer    weights
1     b1  0.85649141 8.712943e-01  4.7351453   7.800199
2     b2  0.75493765 5.263533e+00 11.0522047  18.847993
3     b3  0.51493440 2.213045e+01 23.9234389  48.567933
4     b4 14.46652184 6.448131e-01  0.1834904  16.802932
5     b5  7.24256925 4.510640e-01  7.4512208  17.111658
6     b6  0.52927302 6.957761e-01  5.8826386   9.897335
7     b7  0.68818971 1.632177e-01  6.2494310   9.161813
8     b8  0.14030514 9.555963e-01  5.8162012   8.386880
9     b9  0.20020252 3.688959e-01  7.8806287  10.551445
10   b10  8.35134980 4.699216e-01  3.3008525  13.321270

```


## References

On Financial Networks and Time Series Networks

- Barigozzi, M., & Brownlees, C. (2019). Nets: Network estimation for time series. Journal of Applied Econometrics, 34(3), 347-364.
- Barigozzi, M., & Hallin, M. (2017). A network analysis of the volatility of high dimensional financial series. Journal of the Royal Statistical Society: Series C (Applied Statistics), 66(3), 581-605.
- Baruník, J., & Křehlík, T. (2018). Measuring the frequency dynamics of financial connectedness and systemic risk. Journal of Financial Econometrics, 16(2), 271-296.



# Reading List

[1] Newman, M. (2018). Networks. Oxford University Press.
