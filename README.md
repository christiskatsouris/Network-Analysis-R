# Network-Analysis-R

# [A]. Graph Centrality Measures

To obtain Centrality measures from a graph we first need to obtain the adjacency matrix. Then, the centrality measures, which appear in the form of estimated vectrors for instance, can be calculated using either 'build-in' functions from R packages or by developing our own estimation procedure using a function. 

### Katz centrality 

Proposed by Katz (1953), for a symmetric adjacency matrix $\mathbf{A}$ with a vector of centrality scores for the nodes of the network given by
$$ KC_i(\alpha ) = $$

$$\sum_{j=0}^{\infty} \alpha^j \sum_{i=1}^N \lambda_i^k v_i v_i^{\top} \mathbf{1} $$.

The benefits of Katz centrality is that it the centrality score of nodes can be decomposed into two components, i.e., the idiosyncratic centrality and the system-related centrality, that is, the centrality passed to it in proportion to how important its neighbours are. Due to the construction of katz centrality of capturing the influence of nodes (i.e., the nodes which a node is connected to) is considered as a robust centrality measure in capturing financial contagion and risk transmission within the network, since we are also capturing the influence of financial institutions due to connectedness induced by the underline network topology. 


```R

# Constructing the graph matrix
G <- as.undirected( graph.adjacency( correlation.matrix, weighted = T) )

# Estimate closeness centrality
closeness.centrality.vector <- betweeness(G, mode="in")
closeness.centrality.vector <- as.vector(closeness.centrality.vector)



```

Therefore, as we see above there are various measures which capture the main features of the network topology. 

## Assignment 1



## References

- Huang, Q., Zhao, C., Zhang, X., Wang, X., & Yi, D. (2017). Centrality measures in temporal networks with time series analysis. EPL (Europhysics Letters), 118(3), 36001.
- Liao, H., Mariani, M. S., Medo, M., Zhang, Y. C., & Zhou, M. Y. (2017). Ranking in evolving complex networks. Physics Reports, 689, 1-54.


# [B]. Adjacency Matrices and Network Data

There are various R packages that allow to perform Network Analysis and simulate Adjacency matrices. 

In this teaching page we provide some key examples. In particular, we mainly focus on the aspects of simulating network data as well as the construction of adjacency matrix from simulated data based on the notion of Granger Causality. Furthermore, a second aspect of interest is the notion of 'tail dependency' in graphs or more generally when considering network data. For instance the framework proposed by [Katsouris (2021)](https://arxiv.org/abs/2112.12031) introduces a network-driven tail risk matrix (or financial connectedness matrix) which is constructed based on risk measures that capture the effects of financial contagion and systemic risk in financial markets (see, also Härdle et al. (2016)). Specifically in financial econometrics an important research question from the newtork analysis perspective is the mechanism which contributes to the transmission of systemic risk as well as the spillover effects of financial contagion.  

```R

# Install R package
install.packages("network")
library(network)

# Examples of constructing Adjacency Matrices
num_nodes      <- 10
network_matrix <- matrix(round(runif(num_nodes*num_nodes)), nrow = num_nodes, ncol = num_nodes)

> network_matrix

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

## Example 1

Using the R package ['NetworkRiskMeasures'](https://cran.r-project.org/web/packages/NetworkRiskMeasures/index.html) simulate a set of variables that represent a financial network. 

```R

# Install R package
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

## Example 2

Using the R package ['igraph'](https://cran.r-project.org/web/packages/nets/index.html) construct an adjacency matrix based on a Granger Causality network.  

```R

# Install R package
install.packages("nets")
library(nets)

# estimate a VAR model
> lambda  <- 0.01
> system.time( mdl <- nets(y,CN=FALSE,p=P,lambda=lambda*T,verbose=TRUE) )

iter 1 of 1 
 Iter: 0001 RSS: 6.4447 Pen: 0.0000 Obj: 6.4447 Spars: 1.0000
 Iter: 0002 RSS: 1.4296 Pen: 0.1151 Obj: 1.5447 Spars: 0.4800 Delta: 6.3879
 Iter: 0003 RSS: 0.8885 Pen: 0.0257 Obj: 0.9141 Spars: 0.6933 Delta: 0.0126
 Converged! RSS: 0.8885 Pen: 0.0257 Obj: 0.9141 Spars: 0.6933 Delta: 0.0001
   user  system elapsed 
   0.23    0.00    0.25 
   
> g.adj.hat <- mdl$g.adj
> g.adj.hat 
   V1 V2 V3 V4 V5
V1  0  1  0  0  0
V2  1  0  1  1  0
V3  0  1  0  0  1
V4  1  0  1  0  1
V5  0  1  0  1  0

```


## Example 3

Using the R package ['nets'](https://cran.r-project.org/web/packages/nets/index.html) simulate time series data with network dependence. Notice that the notion of 'network dependence' is not formally mathematically defined in the literature, so currently there is no formal definition (to the best of my knowledge). This is an active research field, so many open research questions remain.

```R

# Install R package
install.packages("nets")
library(nets)

N  <- 5
P  <- 3
T  <- 1000
A  <- array(0,dim=c(N,N,P))
C  <- matrix(0,N,N)

A[,,1]   <- 0.7 * diag(N)
A[,,2]   <- 0.2 * diag(N)
A[1,2,1] <- 0.2
A[4,3,2] <- 0.2

C       <- diag(N)
C[1,1]  <- 2
C[4,2]  <- -0.2
C[2,4]  <- -0.2
C[1,3]  <- -0.1
C[1,3]  <- -0.1

Sig <- solve(C)
L   <- t(chol(Sig))
y   <- matrix(0,T,N)
eps <- rep(0,N)

for( t in (P+1):T )
{
  z <- rnorm(N)
  for( i in 1:N )
  {
    eps[i] <- sum( L[i,] * z )
  }
  
  for( l in 1:P )
  {
    for( i in 1:N )
    {
      y[t,i] <- y[t,i] + sum(A[i,,l] * y[t-l,])
    }
  }
  y[t,] <- y[t,] + eps
}
lambda <- c(1,2)

system.time( mdl <- nets(y,P,lambda=lambda*T,verbose=TRUE) )
mdl

> mdl
 Time Series Panel Dimension: T=1000 N=5
 VAR Lags P=1
 RSS 1.009657 Num Par 8 Lasso Penalty:  1000 2000

```

## Assignment 2

Using a modelling approach of your choice, simulate network data and obtain the adjacency matrix of the induced graph. Furthermore, consider a suitable econometric specification for modelling the dependence structre of the time series data under network dependence. Assume that the simulated data generating process is correctly specified.  

## References

On Financial Networks using Times Series Data:
(Systemic Risk in Financial Markets)

- Allen, F., & Gale, D. (2000). Financial contagion. Journal of political economy, 108(1), 1-33.
- Acharya, V. V., Pedersen, L. H., Philippon, T., & Richardson, M. (2017). Measuring systemic risk. The review of financial studies, 30(1), 2-47.
- Acemoglu, D., Ozdaglar, A., & Tahbaz-Salehi, A. (2015). Systemic risk and stability in financial networks. American Economic Review, 105(2), 564-608.
- Anufriev, M., & Panchenko, V. (2015). Connecting the dots: Econometric methods for uncovering networks with an application to the Australian financial institutions. Journal of Banking & Finance, 61, S241-S255.
- Balboa, M., López-Espinosa, G., & Rubia, A. (2015). Granger causality and systemic risk. Finance Research Letters, 15, 49-58.
- Barigozzi, M., & Brownlees, C. (2019). Nets: Network estimation for time series. Journal of Applied Econometrics, 34(3), 347-364.
- Barigozzi, M., & Hallin, M. (2017). A network analysis of the volatility of high dimensional financial series. Journal of the Royal Statistical Society: Series C (Applied Statistics), 66(3), 581-605.
- Baruník, J., & Křehlík, T. (2018). Measuring the frequency dynamics of financial connectedness and systemic risk. Journal of Financial Econometrics, 16(2), 271-296.
- Baumöhl, E., Kočenda, E., Lyócsa, Š., & Výrost, T. (2018). Networks of volatility spillovers among stock markets. Physica A: Statistical Mechanics and its Applications, 490, 1555-1574.
- Borri, N. (2019). Conditional tail-risk in cryptocurrency markets. Journal of Empirical Finance, 50, 1-19.
- Billio, M., Getmansky, M., Lo, A. W., & Pelizzon, L. (2012). Econometric measures of connectedness and systemic risk in the finance and insurance sectors. Journal of financial economics, 104(3), 535-559.
- Chen, C. Y. H., Härdle, W. K., & Okhrin, Y. (2019). Tail event driven networks of SIFIs. Journal of Econometrics, 208(1), 282-298.
- Diebold, F. X., & Yılmaz, K. (2014). On the network topology of variance decompositions: Measuring the connectedness of financial firms. Journal of econometrics, 182(1), 119-134.
- Härdle, W. K., Wang, W., & Yu, L. (2016). Tenet: Tail-event driven network risk. Journal of Econometrics, 192(2), 499-513.
- Hashem, S. Q., & Giudici, P. (2016). NetMES: a network based marginal expected shortfall measure. The Journal of Network Theory in Finance, 2(3), 1-36.
- Huang, W. Q., & Wang, D. (2018). A return spillover network perspective analysis of Chinese financial institutions’ systemic importance. Physica A: Statistical Mechanics and its Applications, 509, 405-421.
- Katsouris, C. (2021). Optimal Portfolio Choice and Stock Centrality for Tail Risk Events. [arXiv preprint arXiv:2112.12031](https://arxiv.org/abs/2112.12031).
- Tobias, A., & Brunnermeier, M. K. (2016). CoVaR. The American Economic Review, 106(7), 1705.
- Yang, C., Chen, Y., Niu, L., & Li, Q. (2014). Cointegration analysis and influence rank—A network approach to global stock markets. Physica A: Statistical Mechanics and its Applications, 400, 168-185.



# [C]. High Dimensional Network Data Analysis 

(Advanced Topics in Modelling High Dimensional Time Series Data) 


## References

On Time Series Specifications for Network Data:

- Ando, T., Greenwood-Nimmo, M., & Shin, Y. (2022). Quantile Connectedness: Modeling Tail Behavior in the Topology of Financial Networks. Management Science, 68(4), 2401-2431.
- Zhu, X., & Pan, R. (2020). Grouped network vector autoregression. Statistica Sinica, 30(3), 1437-1462.
- Zhu, X., Wang, W., Wang, H., & Härdle, W. K. (2019). Network quantile autoregression. Journal of econometrics, 212(1), 345-358.
- Zhu, X., Wang, W., Wang, H., & Härdle, W. K. (2019). Network quantile autoregression. Journal of econometrics, 212(1), 345-358.
- Xu, X., Wang, W., Shin, Y., & Zheng, C. (2022). Dynamic Network Quantile Regression Model. Journal of Business & Economic Statistics, (just-accepted), 1-36.
- Krackhardt, D. (1988). Predicting with networks: Nonparametric multiple regression analysis of dyadic data. Social networks, 10(4), 359-381.


# Reading List

[1] Newman, M. (2018). Networks. Oxford University Press.

[2] Bühlmann, P., & Van De Geer, S. (2011). Statistics for high-dimensional data: methods, theory and applications. Springer Science & Business Media. 

# Disclaimer

The author (Christis G. Katsouris) declares no conflicts of interest.

The proposed Course Syllabus is currently under development and has not been officially undergone quality checks. All rights reserved.

Any errors or omissions are the responsibility of the author.


# How to Cite a Website

See: https://www.mendeley.com/guides/web-citation-guide/
