# Network-Data-Analysis-R

Teaching page on aspects related to Network Analysis using the Statistical Software R (Drafted July 2022).

### Course Overview:

The main philosophy with this course is to combine traditional network analysis (graph theory) with more formal econometric specifications suitable for network data as well as computational statistics aspects. Emphasis is given on modelling systemic risk, financial contagion as well as tail interdependancies using graph representations. Although this course does not cover formal probability theory for network analysis (to be covered in a different course), we introduce state-of-the-art techniques and programming capabilities with R for each topic covered.


# Introduction 

The notion of financial connectedness which arises as an underline mechanism due to the complex structure of financial systems, implies that to be able to accuratly  and consistently monitor financial contagion, spillover effects and the amplification of systemic risk, it requires to be incorporated within the econometric identification stage. In particular, the rise of systemic risk is considered to be due to shocks transmitted via system-wide connectedness. Moreover, such type of interdependancies can be traced in the long memory properties of time series under a more general form of dependence  (such as network information) and therefore appropriate modelling methodologies require to capture the evolution of spillover effects and the network's spectral response within a infinite time-horizon. Different approaches in the literature provide a different representation and identification strategy. For example, a suitable methodology for modelling financial contagion could depend less on the network topology and the process of link formation. 

# [A]. Graph Centrality Measures

Centrality measures are often used as a way to provide a statistical representation of how connected a node is and to access spillover effects within the network. The introduction of centrality measures can answer relevant questions within the framework of financial networks such as: "Who is the key player?" (see, [Ballester et al. (2006)](https://www.jstor.org/stable/3805930#metadata_info_tab_contents)), "What is the most vulnerable to economic shocks node?", or "What is the level of finacial connectedness of core versus periphery nodes in the graph?". Furthermore, [Drehmann and Tarashev (2013)](https://www.sciencedirect.com/science/article/pii/S1042957313000326), asks "Who is more systemically important the lender or the borrower in an interbank market transaction?". Further, related research questions to the systemic risk side of things include: What is the influence of financial connectedness on the systemic risk transmission mechanism? Does systemic risk create arbitrage opportunities and how can we minimize the negative spillovers to the financial system and its stability? Does the structure of the network matter in economic and investment behaviour such as portfolio allocation decision making? Can financial contagion and correlation externalities be reduced with appropriate portfolio allocation strategies? Therefore, overall the exact network topology at certain periods of time (e.g., such as periods of finacial turbulence or market exuberance) can amplify the transmission of financial contagion and thus is important to develop robust econometric methodologies for systemic risk monitoring. 

Generally speaking, the use of centrality measures can provide network information related to:

(a) the properties of local topology via measures such as degree centrality and page rank, and 

(b) global information such as closeness centrality and betweenness centrality. 

Let $\mathcal{G} = ( \mathcal{E}, \mathcal{N} )$ be a non-empty graph consisting of a set of nodes where $\mathcal{E}$ represents the set of edges and $\mathcal{N}$ represents the set of nodes such that $| \mathcal{N} | = N$, the number of nodes in the graph. 

In other words, the network under examination represents an economy of N economic agents (nodes in the graph) with $\mathcal{N} ∈ (1, ..., N)$ which have financial interactions in pairs represented by the connected edges of the network. We begin by defining the main centrality measures. 

### Katz centrality. 

Proposed by Katz (1953), for a symmetric adjacency matrix $\mathbf{A}$ with a vector of centrality scores for the nodes of the network given by

$$K_i = \sum_{j=0}^{\infty} \alpha^j \sum_{i=1}^N \lambda_i^k v_i v_i^{\top} \mathbf{1}.$$

The benefits of Katz centrality is that it the centrality score of nodes can be decomposed into two components, i.e., the idiosyncratic centrality and the system-related centrality, that is, the centrality passed to it in proportion to how important its neighbours are. Due to the construction of katz centrality of capturing the influence of nodes (i.e., the nodes which a node is connected to) is considered as a robust centrality measure in capturing financial contagion and risk transmission within the network, since we are also capturing the influence of financial institutions due to connectedness induced by the underline network topology. 

### Page rank centrality. 

A modification of the katz centrality is the page rank centrality, which corrects for the contribution of neighbouring nodes on the impact each node has within the network. More specifically, with the eigenvector and katz centrality, there is no distinction between degree centrality and the level of connectedness of these neighbouring nodes. For example, low degree nodes may receive a high score because they are connected to very high degree nodes, even though they may have low degree centrality. Thus page rank centrality scales the contribution of node $i$'s neighbours, $j$, to the centrality of node $i$ by the degree of i. Thus, the page rank centrality is given by

 $$PR_i = \alpha \sum_{j=1}^N A_{ji} \frac{v_j}{d_j} + \beta = \mathbf{\beta}(\mathbf{I} - \alpha\mathbf{D}^{-1} \mathbf{A})^{-1}.$$
 
 where $d_j$ the degree centrality of node $j$. Notice that the weighted characteristic path length measures the average shortest path, from one node to any other node in the network. Thus, it can be interpreted as the shortest number of connections a shock from a node needs to reach another connected node in the network. 
 
 ### Closeness centrality. 
 
 t access the centrality of a node at the local neighbourhood level. For example, the larger the closeness centrality of an institution the faster the influence in the other nodes of the network since it requires fewer steps for an impact to reach other nodes. The normalized closeness centrality of a node is computed as

$$CC_i= N-1 / \sum_{j=1}^N d_{ij}.$$

### Betweenness centrality. 

Considered for example, two financial institutions which have large betweenness centrality, this implies that the pair is important is the transmission of shocks. It is defined as the ratio of the total number of all shortest paths in the network that go via this node and the number of all other shortest paths that do not pass this node. 

$$CB_i= \sum_{s \neq t \neq j} \frac{\sigma_{st}(i)}{\sigma_{st}}.$$

### Leverage centrality. 

Leverage centrality considers the degree of a node relative to its neighbours and is based on the principle that a node in a network is central if its immediate neighbours rely on that node for information\footnote{A node with negative leverage centrality is influenced by its neighbors, as the neighbors connect and interact with far more nodes. A node with positive leverage centrality, on the other hand, influences its neighbors since the neighbors tend to have far fewer connections (see, Vargas (2017)). The leverage centrality is computed as 

$$LC_i =  \frac{1}{d_i} \sum_{i \in N_i } \frac{ d_i - d_j }{ d_i + d_j }.$$

### Eigenvector centrality. 

The eigenvector of node $i$ is equal to the leading eigenvector $\mathbf{v}_i$ and is computed using the characteristic equation of the adjacency matrix. Thus, the EC is defined

$$v_i = \sum_{ j \in N(i) } v_j = \sum A_{ij} v_j.$$

Thus, we can see that the above definition of the eigenvector centrality implies that it depends on both the number of neighbours $|N(i)|$ and the quality of its connections $\mathbf{v}_j$, for $j \in N(i)$.  

### Square eigenvector centrality. 

Since we are particularly interested to assess the network dynamics in financial networks as well as the robustness and resilience of a network from economic shocks and spillovers, the square eigenvector centrality reflects the impact of the removal of node $j$ from the graph at an eigen-frequency/eigenvalue $\lambda_k$ as presented by Van Mieghem (2014). This is given by the expression

$$(x_k)^2 = - \frac{1}{ c^{\prime}(\lambda_k) } \mathsf{det} ( A_{ \backslash \(j ) } - \lambda_k \mathbf{I}).$$

where 

$$ c_A(\lambda) = \text{det} ( A - \lambda I ), \ \ \  c_A^{\prime} (\lambda) = \frac{ d c_A^{\prime} (\lambda) }{ d \lambda } ,$$

$\mathbf{A}_{ \backslash (j) }$ is obtained from $\mathbf{A}$ by removal of row $j$ and column $j$. 

To obtain Centrality measures from a graph we first need to obtain the adjacency matrix. Then, the centrality measures, which appear in the form of estimated vectrors for instance, can be calculated using either 'build-in' functions from R packages or by developing our own estimation procedure using a function.

```R

# Install R package 
install.packages("igraph")
library(igraph)

# Constructing the graph matrix
G <- as.undirected( graph.adjacency( correlation.matrix, weighted = T) )

# Estimate closeness centrality
closeness.centrality.vector <- betweeness(G, mode="in")
closeness.centrality.vector <- as.vector(closeness.centrality.vector)

# Estimate Eigenvector Centrality
eigenvector.centrality.vector <- eigen_centrality(G)$vector
eigenvector.centrality.vector <- as.matrix( as.vector( eigenvector.centrality.vector) )

```
Furthermore, we can also construct an R function to estimate the leverage centrality measure from a graph as below. Let A be the square adjacency matrix with elements being either 1 or 0. Furthermore, consider the corresponding weighted version of the adjacency matrix.   

```R

leverage <- function (A, weighted = TRUE)
{
  if(nrow(A)!=ncol(A))
  {stop("Input not an adjacency matrix")}
  
  binarize <- function (A)
  {
    bin <- ifelse(A!=0,1,0)
    row.names(bin) <- colnames(A)
    colnames(bin) <- colnames(A)
    
    return(bin)
  }
  
  if(!weighted)
  {B<-binarize(A)
  }else{B<-A}
  
  con <- colSums(B)
  lev <- matrix(1,nrow=nrow(B),ncol=1)
  
  for(i in 1:ncol(B))
  {
  lev[i]<-(1/con[i])*sum((con[i]-con[which(B[,i]!=0)])/(con[i]+con[which(B[,i]!=0)]))
  }
  
  for(i in 1:nrow(lev))
    if(is.na(lev[i,]))
    {
    lev[i,]<-0
    }
  
  lev        <- as.vector(lev)
  names(lev) <- colnames(A)
  
  return(lev)
}  

```

Therefore, this Section introduces various centrality measures which can be employed in order to identify the main features regarding the network topology of the graph with a given structure. Lastly, we introduce a useful result, that is, the Perron-Frobenius Theorem.

$\textbf{Lemma:}$ A vector $\mathbf{x}=(x_1,...,x_n)$ is said to be an eigenvctor of an (N x N matrix) $\mathbf{A}$ with an eigenvalue $\lambda$ if for each $i$ it satisfies the following expression  
$$\sum_{j=1}^N a_{ij}x_j = \lambda x_i.$$ 
Then, the eigenvalues of a matrix $\mathbf{A}$ are roots of the characteristic equation of the matrix $| \mathbf{A} - \lambda \mathbf{I} | = 0$.   

$\textbf{Theorem:}$ (Perron-Frobenius theorem) If A is a positive matrix, there is a unique characteristic root of A, $\lambda(A)$, which has the greatest absolute value. This root is positive and simple, and its associated characteristic vector may be taken to be positive. 

If $\lambda(A) = \{ x : det( \mathbf{A} - x \mathbf{I} ) = 0 \}$ then $\lambda_1(A) \geq ... \geq \lambda_n(A)$ where e.g., $\lambda_{max}(A) = \lambda_1(A)$ and $\lambda_{min }(A) = \lambda_1(A)$.

### Remarks

- Notice that in general a matrix can have complex eigenvalues and eigenvectors, but an adjacency matrix of a graph is a non-negative matrix. Thus, for any non-negative matrix, the Perron-Frobenius theorem guarantees that there exists an eigenvalue which is real and larger than or equal to all other eigenvalues in magnitude. The largest eigenvalue is called the Perron-Frobenius eigenvalue of the matrix, which we will denote by $\lambda_1(\mathbf{A})$. Furthermore, the theorem states that there exists an eigenvector of $\mathbf{A}$ corresponding to $\lambda_1(\mathbf{A})$ all of which components are real and non-negative.
- Furthermore, according to [Katsouris (2021a)](https://arxiv.org/abs/2112.12031) a simple application of the Spectral Vector Decomposition (SVD) on the covariance matrix yields the following expression 

$$\lambda_k = \sum_{q=1}^n v_{kh} v_{qk} \sigma_{q}^2 + \sum_{i=1}^n \overset{n}{\underset{\underset{j \neq i}{j=1}}{\sum}} v_{kj} v_{ik} \sigma_{ji}.$$

Interestingly, this expression shows that each eigenvalue is the sum of a weighted average of the variance of all assets in the portfolio plus the sum of the contributions of the different covariance terms. Hence, we can decompose the contribution of an eigenvalue $\lambda_k$ to the optimal weight, into two components. A first component which arises from the aggregate effect of the idiosyncratic variance of each asset in the portfolio, and a second component which arises from the dependence between the asset $i$ and the remaining assets in the portfolio. 

## Example 1

A related example from financial economics is to investigate the relation between portfolio performance measures and network topology. In other words, the underline graph structure at a given time period provides a representation of the degree of interdependencies in the financial network and consequently a good way for measuring phenomena such as financial contagion, volatility spillover effects as well as systemic risk. A related empirical study to this direction is to investigate the relation between portfolio performance measures and network centrality. In other words, the portfolio optimization problem under appropriate modifications (e.g., on the optimization function and the constraints) can be employed as a shrinkage methodology (see, Assignment 1 on the optimal portfolio problem). 

A second aspect related to this stream of literature, focuses more on the financial theory side of things and more specifically how firm structure (i.e., related firm organization of any form) affects both network centrality as well as investment performance (see, Rossi et al. (2018) and Larckeret al. (2013)). Furthermore, considering that firm organization theory has been developed based on discussions that provide a parallelism of decision making to how the human brain works (related literature is beyond the scope of this teaching page), then one might even consider a related neural network application to investigate the sensitivity of the 'operating environment' (i.e., decision making in organizations in the case of firm theory versus the structural/underline features of the neural network), to out-of-sample performance (similar to the portfolio performance using a forecasting scheme). A related study that explores this concept using a formal statistical framework with neural networks is proposed by [Mancini et al. (2021b)](https://www.sciencedirect.com/science/article/pii/S0893608021003257).    

### Optimal Portfolio weights on Centrality Regressions
 
$$ w_t = \alpha + \beta c_t + \gamma \sigma_t + \epsilon_t.$$


```R

# Examples in R on Firm Centrality and Portfolio Performance


```

### Centrality Measure on Firm Characteristics

In this case, we focus on the induced graph topology, as a measure of financial connectedness, on a given 'snapshot window' (i.e., a static model that corresponds to a specific time within the out-of-sample period). From the statistics point of view, the response variable is endogenously generated from the network structure, based on the corresponding adjacency matrix, estimated from the risk matrix proposed by Katsouris (2021). Then, the explanatory variables correspond to the firm characteristics (four different variables or factors) that can explain overall firm performance and fitness.   

$$ c_t = \delta_0 + \boldsymbol{\delta}_1^{\prime} \boldsymbol{f}_t + u_t.$$

```R

Call:
lm(formula = cc ~ BS_firm_roll_means)

Residuals:
       Min         1Q     Median         3Q        Max 
-1.104e-04 -2.700e-05 -7.586e-06  3.193e-05  8.321e-05 

Coefficients:
                     Estimate Std. Error t value Pr(>|t|)    
(Intercept)          0.446423   0.159877   2.792 0.007077 ** 
BS_firm_roll_means1 -0.010590   0.002903  -3.648 0.000568 ***
BS_firm_roll_means2  0.253659   0.069666   3.641 0.000580 ***
BS_firm_roll_means3 -0.006650   0.002878  -2.311 0.024410 *  
BS_firm_roll_means4 -0.028810   0.011109  -2.593 0.012008 *  
---
Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1

Residual standard error: 4.068e-05 on 58 degrees of freedom
Multiple R-squared:  0.4572,	Adjusted R-squared:  0.4198 
F-statistic: 12.22 on 4 and 58 DF,  p-value: 2.868e-07

```

## Assignment 1

Using the centrality measures introduced above and based on a partial-correlation network as presented in the framework of Anufriev and Panchenko (2015)   examine the effect of network topology to the optimal portfolio choice problem. Specifically, based on the GMVP (min-variance optimization problem), set-up the optimization function using the full investment constraint as well as a constraint on weights being strictly in the interval (0,1). Then, in order to research the question on how centrality could be affecting the min-variance optimization problem use a suitable methodology that either removes central nodes automatically or via an appropriate penalization/shrinkage method. See also the [Computational-Econometrics-R](https://github.com/christiskatsouris/Computational-Econometrics-R) page which has helpful examples on optimization techniques. 

## References

- Abduraimova, K. (2022). Contagion and tail risk in complex financial networks. Journal of Banking & Finance, 106560.
- Anufriev, M., & Panchenko, V. (2015). Connecting the dots: Econometric methods for uncovering networks with an application to the Australian financial institutions. Journal of Banking & Finance, 61, S241-S255.
- Ballester, C., Calvó‐Armengol, A., & Zenou, Y. (2006). Who's who in networks. Wanted: The key player. Econometrica, 74(5), 1403-1417.
- Drehmann, M., & Tarashev, N. (2013). Measuring the systemic importance of interconnected banks. Journal of Financial Intermediation, 22(4), 586-607.
- Huang, Q., Zhao, C., Zhang, X., Wang, X., & Yi, D. (2017). Centrality measures in temporal networks with time series analysis. EPL (Europhysics Letters), 118(3), 36001.
- Katsouris, C. (2021a). Optimal Portfolio Choice and Stock Centrality for Tail Risk Events. arXiv preprint arXiv:2112.12031.
- Katsouris C. (2021b). A Graph Topology Measure for a Time Series Regression-based Covariance Matrix with Tail Estimates. Working paper. University of Southampton.
- Katz, L. (1953). A new status index derived from sociometric analysis. Psychometrika, 18(1):39–43
- Larcker, D. F., So, E. C., & Wang, C. C. (2013). Boardroom centrality and firm performance. Journal of Accounting and Economics, 55(2-3), 225-250.
- Liao, H., Mariani, M. S., Medo, M., Zhang, Y. C., & Zhou, M. Y. (2017). Ranking in evolving complex networks. Physics Reports, 689, 1-54.
- Le, L. T., Eliassi-Rad, T., & Tong, H. (2015, June). MET: A fast algorithm for minimizing propagation in large graphs with small eigen-gaps. In Proceedings of the 2015 SIAM International Conference on Data Mining (pp. 694-702). Society for Industrial and Applied Mathematics.
- Rossi, A. G., Blake, D., Timmermann, A., Tonks, I., & Wermers, R. (2018). Network centrality and delegated investment performance. Journal of Financial Economics, 128(1), 183-206.
- Mancini, T., Calvo-Pardo, H., & Olmo, J. (2021b). Extremely randomized neural networks for constructing prediction intervals. Neural Networks, 144, 113-128.
- Vargas Jr, R., Waldron, A., Sharma, A., Flórez, R., and Narayan, D. A. (2017). A graph theoretic analysis of leverage centrality. AKCE International Journal of Graphs and Combinatorics, 14(3):295–306.
- Van Mieghem, P. (2014). Graph eigenvectors, fundamental weights and centrality metrics for nodes in networks. arXiv preprint arXiv:1401.4580.

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

### Remarks

- Linear Granger causality as first introduced by Granger (1969, 1980) has been further tested in various frameworks (e.g., VAR(p) models). Specifically, let two time series be $Y_{i,t}$ and $Y_{j,t}$, for $t = 1,...,T$ with the corresponding information set available at time $t-1$ be $I_{t-1} \equiv ( I_{i, t-1}, I_{j, t-1} )$. Then we say that if there are enough statistical evidence to support the hypothesis 

$$ H_1: F( Y_{i,t} | I_{i, t-1} ) \neq  F( Y_{i,t} | I_{t-1} ) \ \  \text{for} \ \forall \  \ F( .| I_{t-1}) \in \mathbb{R}.$$

- Then there is additional information in the set of $Y_{j,t-1}$  which helps predict the future values of $Y_{i,t-1}$, that is, time series $Y_{j,t}$ is said to "Granger-cause" $Y_{i,t}$, where $t = 1,...,T$. 

- Generally, the notion of Granger Causality provides a suitable mechanism for the identification of causal links in graph structures, using a formal statistical methodology.  Another example, of a suitable such mechanism is to apply a Granger Causality statistic in the tails of the conditional distributions, testing this way  the occurrence of a left or right tail event of a random variable is systematically followed (in a statistical sense) by the occurrence of a left or right tail event of a second random variable. A GC test is based on the following econometric specification 

$$ \Delta Y_t = \alpha + \sum_{i=1}^p \beta_{yi} \Delta Y_{t-1} + \sum_{i=1}^p \beta_{xi} \Delta X_{t-1} + \epsilon_t,$$

where $\Delta X$ and $\Delta Y$ are stationary time series. Thus, when the coefficients $\beta_{xi}$ are jointly significant we reject the null hypothesis that $X_t$ does not Granger cause $Y_t$ based on the standard F-test.

- Lastly, from a more probabilistic point of view, for example, Valjarević and Petrović (2012) examine the equivalence of statistical causality and the orthogonality property of martingales. Their theorem proves that two independent martingales are orthogonal to each other if and only if each of them is caused by itself. The application of this result can be found in trading strategies. In particular, a trading strategy is locally risk-minimizing if and only if the cost process of the trading strategy and the unpredictable, martingale part of the price fluctuation of a stock are independent and caused by themselves. 


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

### Remarks

Further examples of algorithms can be found on various 'R help' websites which explain step-by-step the syntax of main R commands and provide illustrations of statistical procedures via output examples using R code fragements (see, [UCLA Advanced Research Computing: Statistical Methods and Data Analytics](https://stats.oarc.ucla.edu/r/codefragments/simulating_network/)). 


## Assignment 2

Using a modelling approach of your choice, simulate network data and obtain the adjacency matrix of the induced graph. Furthermore, consider a suitable econometric specification for modelling the dependence structre of the time series data under network dependence. Assume that the simulated data generating process is correctly specified.  

## References

On Financial Networks using Times Series Data:
(Systemic Risk in Financial Markets)

- Ando, T., Greenwood-Nimmo, M., & Shin, Y. (2022). Quantile Connectedness: Modeling Tail Behavior in the Topology of Financial Networks. Management Science, 68(4), 2401-2431.
- Allen, F., & Gale, D. (2000). Financial contagion. Journal of political economy, 108(1), 1-33.
- Acharya, V. V., Pedersen, L. H., Philippon, T., & Richardson, M. (2017). Measuring systemic risk. The review of financial studies, 30(1), 2-47.
- Acharya, V., Engle, R., & Richardson, M. (2012). Capital shortfall: A new approach to ranking and regulating systemic risks. American Economic Review, 102(3), 59-64.
- Acemoglu, D., Ozdaglar, A., & Tahbaz-Salehi, A. (2015). Systemic risk and stability in financial networks. American Economic Review, 105(2), 564-608.
- Anufriev, M., & Panchenko, V. (2015). Connecting the dots: Econometric methods for uncovering networks with an application to the Australian financial institutions. Journal of Banking & Finance, 61, S241-S255.
- Balboa, M., López-Espinosa, G., & Rubia, A. (2015). Granger causality and systemic risk. Finance Research Letters, 15, 49-58.
- Barigozzi, M., & Brownlees, C. (2019). Nets: Network estimation for time series. Journal of Applied Econometrics, 34(3), 347-364.
- Barigozzi, M., & Hallin, M. (2017). A network analysis of the volatility of high dimensional financial series. Journal of the Royal Statistical Society: Series C (Applied Statistics), 66(3), 581-605.
- Baruník, J., & Křehlík, T. (2018). Measuring the frequency dynamics of financial connectedness and systemic risk. Journal of Financial Econometrics, 16(2), 271-296.
- Baumöhl, E., Kočenda, E., Lyócsa, Š., & Výrost, T. (2018). Networks of volatility spillovers among stock markets. Physica A: Statistical Mechanics and its Applications, 490, 1555-1574.
- Borri, N. (2019). Conditional tail-risk in cryptocurrency markets. Journal of Empirical Finance, 50, 1-19.
- Billio, M., Getmansky, M., Lo, A. W., & Pelizzon, L. (2012). Econometric measures of connectedness and systemic risk in the finance and insurance sectors. Journal of financial economics, 104(3), 535-559.
Calabrese, R., Elkink, J. A., & Giudici, P. S. (2017). Measuring bank contagion in Europe using binary spatial regression models. Journal of the Operational Research Society, 68(12), 1503-1511.
- Cai, J., Eidam, F., Saunders, A., & Steffen, S. (2018). Syndication, interconnectedness, and systemic risk. Journal of Financial Stability, 34, 105-120.
- Cui, X., & Yang, L. (2022). Systemic risk and idiosyncratic networks among global systemically important banks. International Journal of Finance & Economics.
- Chen, C. Y. H., Härdle, W. K., & Okhrin, Y. (2019). Tail event driven networks of SIFIs. Journal of Econometrics, 208(1), 282-298.
- Diebold, F. X., & Yılmaz, K. (2014). On the network topology of variance decompositions: Measuring the connectedness of financial firms. Journal of econometrics, 182(1), 119-134.
- Härdle, W. K., Wang, W., & Yu, L. (2016). Tenet: Tail-event driven network risk. Journal of Econometrics, 192(2), 499-513.
- Hashem, S. Q., & Giudici, P. (2016). NetMES: a network based marginal expected shortfall measure. The Journal of Network Theory in Finance, 2(3), 1-36.
- Huang, W. Q., & Wang, D. (2018). A return spillover network perspective analysis of Chinese financial institutions’ systemic importance. Physica A: Statistical Mechanics and its Applications, 509, 405-421.
- Katsouris, C. (2021). Optimal Portfolio Choice and Stock Centrality for Tail Risk Events. [arXiv preprint arXiv:2112.12031](https://arxiv.org/abs/2112.12031).
- Tobias, A., & Brunnermeier, M. K. (2016). CoVaR. The American Economic Review, 106(7), 1705.
- Yang, C., Chen, Y., Niu, L., & Li, Q. (2014). Cointegration analysis and influence rank—A network approach to global stock markets. Physica A: Statistical Mechanics and its Applications, 400, 168-185.
- Valjarević, D., & Petrović, L. (2012). Statistical causality and orthogonality of local martingales. Statistics & Probability Letters, 82(7), 1326-1330.


# [C]. High Dimensional Network Data Analysis 

(Advanced Topics in Modelling High Dimensional Time Series Data) 

We begin by focusing on the graphical lasso shrinkage. The corresponding nodewise regression is defined as below 

$$\hat{ \gamma }_j := \underset{ \gamma \in \mathbb{R}^{p-1} }{ \text{arg min} } \left( || r_j^{*} - r^{*}(-j) \gamma ||_2^2 / n + 2 \lambda_j ||  \gamma ||_1 \right).$$

Then, the nodewise regression estimator $\hat{\Theta}$ of the precision matrix $\Theta$ is constructed as follow. 

$$
\hat{\Theta} := 
\begin{bmatrix}
1 \ & \  -\gamma_{1,2} \ & \  \ & \ -\gamma_{1,p} 
\\
-\gamma_{1,2} \ & \ 1  \ & \  \ & \ 
\\
\vdots \ & \ \vdots  \ & \ \vdots \ & \ \vdots
\\
-\gamma_{p,1} \ & \  -\gamma_{p,2} \ & \  \ & \ 1
\end{bmatrix}.
$$

Further examples we can examine include the statistical estimation and inference for the Network Vector Autoregression model (see, [Zhu et al. (2017)](https://projecteuclid.org/journals/annals-of-statistics/volume-45/issue-3/Network-vector-autoregression/10.1214/16-AOS1476.full)); although these aspects correspond to more advanced topics in Network Analysis Using R. Furthermore, an extension of the particular framework is presented by [Zhu et al. (2020)](https://www.jstor.org/stable/26968936#metadata_info_tab_contents) who consider the estimation and inference problem for the NVAR model under grouped structure. 

## Example 4

```R

# Install R packages
install.packages("MASS")
library(MASS)

betaOLS<-function(Ymat, W, Z)                        ### OLS estimation for theta
{

  Ymat1 <- W%*%Ymat                                  ### obtain WY
  Time  <- ncol(Ymat)-1                                                                                         
  if (is.null(Z))
    X <- cbind(rep(1, nrow(Ymat)*Time),              ### the intercept
               as.vector(Ymat1[,-ncol(Ymat)]),       ### WY_{t-1}
               as.vector(Ymat[,-ncol(Ymat)]))
  else
    X <- cbind(rep(1, nrow(Ymat)*Time),              ### the intercept
               as.vector(Ymat1[,-ncol(Ymat)]),       ### WY_{t-1}
               as.vector(Ymat[,-ncol(Ymat)]),        ### Y_{t-1}
               do.call("rbind", rep(list(Z), Time))) ### nodal covariates
  
  invXX     <- solve(crossprod(X))                   ### {t(X)X}^{-1}
  Yvec      <- as.vector(Ymat[,-1])                  ### the response vector
  thetaEst  <- invXX%*%colSums(X*Yvec)                                                                          
  sigmaHat2 <- mean((Yvec - X%*%thetaEst)^2)         ### estimation for hat sigma^2
  covHat    <- invXX*sigmaHat2                       ### covariance for hat theta
  
  return( list( theta = thetaEst, covHat = covHat, sigmaHat = sqrt(sigmaHat2)) )           
}

# Coding Procedure Reference: Associate Professor Xuening Zhu
# Articles References: Zhu et al. (2017), Zhu et al. (2020).

```

## Remarks:

- The above R function provides the estimation procedure for the parameter beta of the NAR model. However, when additional structure is imposed (such as the grouped dependence), then the econometric identification becomes more complex and so for modelling purposes it requires the implementation of a commonly used statistical procedure such as the EM Algorithm (see, Wu C.J.(1983)) and this page [Computational Econometrics](https://github.com/christiskatsouris/Computational-Econometrics-R).


```R

### Implementation of the EM algorithm
# Coding Procedure Reference: Associate Professor Xuening Zhu
# Articles References: Zhu et al. (2017), Zhu et al. (2020).

################################################
### EM algorithm
################################################

EM.NAR<-function( Ymat =  Ymat, W = W, Z = Z, K = K, seed = F )
{
   Ymat =  Ymat
   W = W
   Z = Z
   K = K

  if (is.null(Z))
    npara = 3   # number of parameters
  else
    npara = 3+ncol(Z)
  
  alpha = rep(1/K, K)
  if (seed)
    set.seed(1234)
  
  ### set initial values for the algorithm
  nar_para = betaOLS(Ymat, W, Z)  # generate the initial estimator by the NAR model
  if (K==1)
    return(nar_para)
  theta = t(sapply(nar_para$theta, function(x) runif(K, min = x-0.05, max = x+0.05))) # generate the initial values by small perturbations of NAR estimator
  sigma = rep(nar_para$sigmaHat, K) # for simplicity set sigma the same for initial values
  
  ### get X variables
  WYmat = W%*%Ymat
  N     = nrow(Ymat)
  Time  = ncol(Ymat)

  X = as.matrix(X)         # all predictors
  Y = as.vector(Ymat[,-1]) # the response variable
  
  ### start the EM algorithm
  delta = 1
  ezK   = matrix(0, ncol = K, nrow = nrow(Ymat))
  while(delta > 10^-5)
  {
    ### E-step
    eps_hat = (Y - X%*%theta)%*%diag(x = 1/sigma)
    for (k in 1:K)
    {
      eps_mat = matrix(eps_hat[,k], ncol = Time - 1)
      
      # As a stability condition it requires to calculate the log-transformed probability first
      ezK[,k] = -(Time-1)*log(sigma[k]) - rowSums(eps_mat^2/2) + log(alpha[k]) 
    }
    
    ezK = ezK - rowMeans(ezK)
    ind = apply(ezK, 1, function(x) 
    {
      if (any(x>500)) # if it is large enough (>500), return the largest as 1 (otherwise the R will produce an NaN)
        return(which.max(x)) # return which one is largest
      return(0)
    })
    
    ezK = exp(ezK)/rowSums(exp(ezK)) # exp-transform back to obtain the probability
    if (sum(ind)>0) 
    { # code the values recorded in ind to be 0 and 1
      ii = which(ind>0)
      ezK[ii,] = 0
      ezK[cbind(ii,ind[ii])] = 1
    }
    
    ### M-step
    theta0 = theta; sigma0 = sigma; alpha0 = alpha
    for (k in 1:K)
    {
      zz        = rep(ezK[,k], Time - 1)
      X_new     = zz*X
      theta[,k] = ginv(crossprod(X_new, X))%*%crossprod(X_new, Y)  # obtain the estimator for theta
      sigma[k]  = sqrt(sum(zz*(Y - X%*%theta[,k])^2)/sum(zz))      # obtain the estimator for sigma
      
      }
    }
     alpha = colSums(ezK)/N # obtain the estimator for alpha
    delta  = max(abs(c(theta - theta0, sigma - sigma0, alpha - alpha0)))
  }
  return( list(theta = theta, alpha = alpha, sigma = sigma, ezK = ezK) )
}

```

### Task

Begin by studying the related statistical theory for the EM algorithm. Then, using on a dataset of your choice run the implementation of the EM algorithm for the NAR model step-by-step (as given above) and report related aspects of the computational procedure such as convergence rate (and computational time) with respect to the sample size, description of the updating steps as well as the main findings (e.g., parameter estimates, standard errors, confidence intervals etc.) based on the selected dataset. 

As an additional step, write your own coding procedure (with the help of the above two R functions), that performs the prediction step for the Grouped NAR model.  

HINT: Suitable datasets that can be employed include cross-sectional data of stock returns. In this case the adjacency matrix has to be constructed before proceeding with estimation and inference with a network-based modelling approach, based on a given statistical methodology that specifies the graph structure (see Section 2 above). Furthermore, we can also consider 'natural' network data - plenty of applications can be found on the website of [Network Data Repository](https://networkrepository.com/index.php).  


# [C1]. Further Econometric Specifications

In order to capture the main features of financial networks, one might be interested to examine the simultaneous features of Granger causality and spatial dependence (or graph dependence in the broader sence) with a structural econometrics model as below

$$Y_t = \mathbf{A} . Y_{t-1} + \rho \mathbf{W} . Y_{t} + U_t,$$

where $Y_t = (y_{1t},...,y_{Nt})^{\top}$ is a vector of dependent variables (e.g., excess returns), $\mathbf{W} \in \mathbb{R}$ is an (N x N) non-stochastic spatial matrix with a zero diagonal, $\rho$ is a scalar parameter and $U_t = (u_{1t},...,u_{Nt})^{\top}$ is a vector of i.i.d. disturbances with zero mean and finite variances. Thus, for the correct identification of the above specification we assume that there exists a power representation of $(I-\rho \mathbf{W})^{-1}$ given by  

$$ (\mathbf{I}-\rho \mathbf{W})^{-1} = \sum_{j=0}^{\infty} \rho^j \mathbf{W}^j.$$ 

If $| \rho | < 1$ then $(I-\rho \mathbf{W})$ is nonsingular and has a unique solution, capturing the idea that connections further away are less influential. If $| \rho | > 1$ then the process is explosive, which is interpreted as complete financial collapse. 

Furthermore, under the above specification, the term $A Y_{t-1}$ captures the Granger causality effects between the components of the vector valued series $Y_t$ and its lag term $Y_{t-1}$. Moreover, the second term of the model $\rho W Y_{t}$ captures the spatial dependence (i.e. the simultaneous dependence) of the network since it includes a spatial lagged of the dependent variable along with the weight matrix $\mathbf{W}$ which is exogenously defined in the dynamics of the system (a procedure is followed to determine the elements of $W_{ij}$). We leave the above considerations for future research (see, [Active Research Project](https://www.researchgate.net/project/The-Econometrics-of-Financial-Networks-Theory-and-Applications)). A related framework which focuses on Granger Causality detection using high dimensional systems of neural networks is proposed by [Mancini et al. (2021a)](https://www.sciencedirect.com/science/article/pii/S0169207020301588).

### Remarks

- Notice that when considering the theoretical underpinnings of feedforward algorithms when applying neural network modelling methodologies based on time series data, this can be done within the framework of Hidden Markov Models. Related methodologies and theory can be found on the homepage of [Professor John Aston](http://www.statslab.cam.ac.uk/~jada2/) ([Google Scholar](https://scholar.google.com/citations?user=YPJIv4QAAAAJ&hl=en)). Moreover, the lecture notes of the course: 'Applied Statistics - Part III. Hidden Markov Models' (2011-2012), Department of Statistics, University of Warwick, are also helpful.    

## References

On Time Series Specifications for Network Data:

- Andreou, E., Gagliardini, P., Ghysels, E., & Rubin, M. (2019). Inference in Group Factor Models With an Application to Mixed‐Frequency Data. Econometrica, 87(4), 1267-1305.
- Amillotta, M., Fokianos, K., & Krikidis, I. (2022, February). Generalized Linear Models Network Autoregression. In International Conference on Network Science (pp. 112-125). Springer, Cham.
- Callot, L., Caner, M., Önder, A. Ö., & Ulaşan, E. (2021). A nodewise regression approach to estimating large portfolios. Journal of Business & Economic Statistics, 39(2), 520-531.
- Krackhardt, D. (1988). Predicting with networks: Nonparametric multiple regression analysis of dyadic data. Social networks, 10(4), 359-381.
- Mancini, T., Calvo-Pardo, H., & Olmo, J. (2021a). Granger causality detection in high-dimensional systems using feedforward neural networks. International Journal of Forecasting, 37(2), 920-940.
- Mazumder, R., & Hastie, T. (2012). The graphical lasso: New insights and alternatives. Electronic journal of statistics, 6, 2125.
- Zhu, X., & Pan, R. (2020). Grouped network vector autoregression. Statistica Sinica, 30(3), 1437-1462.
- Zhu, X., Wang, W., Wang, H., & Härdle, W. K. (2019). Network quantile autoregression. Journal of econometrics, 212(1), 345-358.
- Zhu, X., Pan, R., Li, G., Liu, Y., & Wang, H. (2017). Network vector autoregression. The Annals of Statistics, 45(3), 1096-1123.
- Xu, X., Wang, W., Shin, Y., & Zheng, C. (2022). Dynamic Network Quantile Regression Model. Journal of Business & Economic Statistics, (just-accepted), 1-36.
- Witten, D. M., Friedman, J. H., & Simon, N. (2011). New insights and faster computations for the graphical lasso. Journal of Computational and Graphical Statistics, 20(4), 892-900.
- Wu, C. J. (1983). On the convergence properties of the EM algorithm. The Annals of statistics, 95-103.

# Reading List

$\textbf{[1]}$ Newman, M. (2018). Networks. Oxford University Press.

$\textbf{[2]}$ Newman, M. E., Barabási, A. L. E., & Watts, D. J. (2006). The structure and dynamics of networks. Princeton University Press.

$\textbf{[3]}$ Manual, C. A. (2020). The Econometrics of Networks. network, 255, 260.

$\textbf{[4]}$ Graham, B., & De Paula, A. (Eds.). (2020). The Econometric Analysis of Network Data. Academic Press.

$\textbf{[5]}$ Bühlmann, P., & Van De Geer, S. (2011). Statistics for high-dimensional data: methods, theory and applications. Springer Science & Business Media. 

# Disclaimer

The author (Christis G. Katsouris) declares no conflicts of interest.

The proposed Course Syllabus is currently under development and has not been officially undergone quality checks. All rights reserved.

Any errors or omissions are the responsibility of the author.

# Acknowledgments

The author has benefited by participating in workshops and training sessions related to High Performance Computing both at the University of Southampton as well as at University College London (UCL).

Any questions or comments/suggestions are welcome! 

If you are interested to collaborate on the general theme of 'Network Econometrics', don't hesitate to contact me at christiskatsouris@gmail.com

# How to Cite a Website

See: https://www.mendeley.com/guides/web-citation-guide/

# Historical Background

$\textbf{Paul Erdős}$ (26 March 1913 – 20 September 1996) was a Hungarian mathematician. He was one of the most prolific mathematicians and producers of mathematical conjectures of the 20th century. Erdős pursued and proposed problems in discrete mathematics, graph theory, number theory, mathematical analysis, approximation theory, set theory, and probability theory. Much of his work centered around discrete mathematics, cracking many previously unsolved problems in the field. He championed and contributed to Ramsey theory, which studies the conditions in which order necessarily appears. Overall, his work leaned towards solving previously open problems, rather than developing or exploring new areas of mathematics. Erdős published around 1,500 mathematical papers during his lifetime, a figure that remains unsurpassed. He firmly believed mathematics to be a social activity, living an itinerant lifestyle with the sole purpose of writing mathematical papers with other mathematicians. He was known for his social practice of mathematics, working with more than 500 collaborators (Source: Wikipedia). 
