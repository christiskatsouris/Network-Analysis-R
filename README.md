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
sociomatrix <- matrix(round(runif(num_nodes*num_nodes)), # edge values
                         nrow = num_nodes, #nrow must be same as ncol
                         ncol = num_nodes)

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


## References

On Financial Networks and Time Series Networks

- Barigozzi, M., & Brownlees, C. (2019). Nets: Network estimation for time series. Journal of Applied Econometrics, 34(3), 347-364.
- Barigozzi, M., & Hallin, M. (2017). A network analysis of the volatility of high dimensional financial series. Journal of the Royal Statistical Society: Series C (Applied Statistics), 66(3), 581-605.
- Baruník, J., & Křehlík, T. (2018). Measuring the frequency dynamics of financial connectedness and systemic risk. Journal of Financial Econometrics, 16(2), 271-296.



# Reading List

[1] Newman, M. (2018). Networks. Oxford University Press.
