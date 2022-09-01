library(readr)
library(tidyverse)
library(tsibble)
library(hts)

prison <- read_csv("prison_population.csv")
prison <- prison %>% 
  mutate(Quarter = yearquarter(Date)) %>% 
  select(-Date) 
prison %>% 
  mutate(names = paste(State, Gender, Legal, Indigenous, sep = "-")) %>% 
  select(names, Quarter, Count) %>% 
  tsibble(key = names, index = Quarter) %>% 
  as.ts() -> bts

nodes <- list(8, rep(2, 8), rep(2, 16), rep(2, 32)) # hierarchical structure
prison.hts <- hts(bts, nodes)
ally <- aggts(prison.hts)
nr <- nrow(ally) # no. of obs in the data set
nc <- ncol(ally) # no. of series in the data set
nb <- ncol(bts) # no. of bottom level series
h <- 1
nfreq <- 4 # frequency of the data
time.attr <- tsp(ally)

train.size <- 24  # Number of observations for training
n.iter <- nr - train.size  # Total number of iterations
test.size <- nr - train.size # The available obs for testing

# data from original series
original.y <- array(, dim = c(n.iter, nr, nc))
test.y <- array(, dim = c(n.iter, test.size, nc))
fcasts.base <- array(, dim = c(n.iter, h, nc))
fitted.values <- array(, dim = c(n.iter, nr, nc))

for (i in 1L:nc) {
  # Fitting model using the whole sample
  single.y <- ts(ally[, i], start = c(2005, 1), frequency = nfreq)    
  
  for (j in 1:test.size) {
    ending <- time.attr[1L] + (train.size + j - 2L)/nfreq
    
    # Start rolling window forecasting
    original.y[j, 1:(train.size + j - 1), i] <- y.train <- 
      window(single.y, end = ending)
    test.y[j, 1L:(test.size - j + 1), i] <- 
      window(single.y, start = ending + 1/nfreq, end = time.attr[2])
    
    # Fitting ARIMA models
    fit <- auto.arima(y.train)
    fitted.values[j, 1:(train.size + j - 1), i] <- fitted(fit)
    fcasts.base[j, , i] <- forecast(fit, h = h)$mean
  }
  print(paste("Done with variable", i))
}

# summing matrix
gmat <- hts:::GmatrixH(nodes)
s <- hts:::SmatrixM(gmat)

fcasts.bu <- array(, dim = c(n.iter, h, nc))
for (i in 1L:n.iter) {
  bfcasts <- matrix(fcasts.base[i, , (nc - nb + 1):nc], nrow = 1)
  fcasts.hts <- suppressMessages(hts(bfcasts, nodes = nodes))
  fcasts.bu[i, , ] <- allts(fcasts.hts)
}

# OLS
fcasts.ols <- array(, dim = c(n.iter, h, nc))
for (i in 1L:n.iter) {
  gols <- solve(t(s) %*% s) %*% t(s)
  fcasts.ols[i, , ] <- as.vector(s %*% gols %*% (fcasts.base[i, , ]))
}

# WLSv
fcasts.wlsv <- array(, dim = c(n.iter, h, nc))
for (i in 1L:n.iter) {
  resid <- na.omit(original.y[i, , ] - fitted.values[i, , ])
  wvec <- 1 / colMeans(resid^2, na.rm = TRUE)
  w.1 <- diag(wvec)
  gwls <- solve(t(s) %*% w.1 %*% s) %*% t(s) %*% w.1
  fcasts.wlsv[i, , ] <- as.vector(s %*% gwls %*% (fcasts.base[i, , ]))
}

# MinT(Shrink)
fcasts.mints <- array(, dim = c(n.iter, h, nc))
for (i in 1L:n.iter) {
  resid <- na.omit(original.y[i, , ] - fitted.values[i, , ])
  dmat <- diag(colMeans(resid^2))
  w.1s <- hts:::shrink.estim(resid, dmat)[[1]]
  gmint <- solve(t(s) %*% solve(w.1s) %*% s) %*% t(s) %*% solve(w.1s)
  fcasts.mints[i, , ] <- as.vector(s %*% gmint %*% (fcasts.base[i, , ]))
}


# WLSv
fcasts.wlsv2 <- array(, dim = c(n.iter, h, nc))
for (i in 1L:n.iter) {
  if (i %% 10 == 1) {
    resid <- na.omit(original.y[i, , ] - fitted.values[i, , ])
    wvec <- 1 / colMeans(resid^2, na.rm = TRUE)
    w.1 <- diag(wvec)
    gwls <- solve(t(s) %*% w.1 %*% s) %*% t(s) %*% w.1
  }
  fcasts.wlsv2[i, , ] <- as.vector(s %*% gwls %*% (fcasts.base[i, , ]))
}

# MinT(Shrink)
fcasts.mints2 <- array(, dim = c(n.iter, h, nc))
for (i in 1L:n.iter) {
  if (i %% 10 == 1) {
    resid <- na.omit(original.y[i, , ] - fitted.values[i, , ])
    dmat <- diag(colMeans(resid^2))
    w.1s <- hts:::shrink.estim(resid, dmat)[[1]]
    gmint <- solve(t(s) %*% solve(w.1s) %*% s) %*% t(s) %*% solve(w.1s)
  }
  fcasts.mints2[i, , ] <- as.vector(s %*% gmint %*% (fcasts.base[i, , ]))
}

CalcMSE <- function(object, nodes) {
  # object <- fcasts.bu
  h <- ncol(object)
  n.iter <- nrow(object)
  nc <- dim(object)[3]
  
  mse <- matrix(, nrow = length(nodes) + 2, ncol = h) 
  res <- array(, dim = c(n.iter, h, nc))
  for (i in 1L:n.iter) {
    res[i, , ] <- object[i, , ] - test.y[i, 1:h, ]
  }
  cs <- cumsum(c(0, 1, lapply(nodes, sum)))
  if (h > 1) {
    mse[1L, ] <- colMeans(res[, , 1]^2, na.rm = TRUE)
    for (i in 2:(length(nodes) + 1)) {
      end <- cs[i + 1]
      start <- cs[i] + 1
      series <- seq(start, end)
      mse[i, ] <- rowMeans(colMeans(res[, , series]^2, na.rm = T))
    }
    mse[(length(nodes) + 2), ] <- rowMeans(colMeans(res^2, na.rm = T))
  } else {
    mse[1L, ] <- mean(res[, , 1L]^2, na.rm = TRUE)
    for (i in 2L:(length(nodes) + 1)) {
      end <- cs[i + 1L]
      start <- cs[i] + 1L
      series <- seq(start, end)
      mse[i, ] <- mean(colMeans(res[, , series]^2, na.rm = T))
    }
    mse[(length(nodes) + 2), ] <- mean(colMeans(res^2, na.rm = T))
  }
  mse
}


t(100 * (CalcMSE(fcasts.ols, nodes)/CalcMSE(fcasts.base, nodes) - 1))
t(100 * (CalcMSE(fcasts.wlsv, nodes)/CalcMSE(fcasts.base, nodes) - 1))
t(100 * (CalcMSE(fcasts.mints, nodes)/CalcMSE(fcasts.base, nodes) - 1))

t(100 * (CalcMSE(fcasts.wlsv2, nodes)/CalcMSE(fcasts.base, nodes) - 1))
t(100 * (CalcMSE(fcasts.mints2, nodes)/CalcMSE(fcasts.base, nodes) - 1))


# > t(100 * (CalcMSE(fcasts.ols, nodes)/CalcMSE(fcasts.base, nodes) - 1))
# [,1]     [,2]      [,3]      [,4]      [,5]      [,6]
# [1,] -5.1306 -5.46015 -4.473159 -14.96288 -10.50708 -8.187689
# > t(100 * (CalcMSE(fcasts.wlsv, nodes)/CalcMSE(fcasts.base, nodes) - 1))
# [,1]     [,2]      [,3]     [,4]      [,5]     [,6]
# [1,] 34.47318 2.226849 -2.475108 -13.0499 -13.53553 2.871784
# > t(100 * (CalcMSE(fcasts.mints, nodes)/CalcMSE(fcasts.base, nodes) - 1))
# [,1]     [,2]       [,3]     [,4]      [,5]     [,6]
# [1,] 31.02408 2.753409 -0.9185573 -13.2863 -14.22173 2.266204
# > t(100 * (CalcMSE(fcasts.wlsv2, nodes)/CalcMSE(fcasts.base, nodes) - 1))
# [,1]      [,2]      [,3]      [,4]      [,5]    [,6]
# [1,] 33.75779 0.1579313 -4.456538 -14.47859 -14.65988 1.44014
# > t(100 * (CalcMSE(fcasts.mints2, nodes)/CalcMSE(fcasts.base, nodes) - 1))
# [,1]      [,2]      [,3]     [,4]      [,5]     [,6]
# [1,] 30.81135 0.3657686 -3.495602 -14.1161 -14.56478 1.053508