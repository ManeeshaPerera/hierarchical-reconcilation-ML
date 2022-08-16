library(readr)
library(forecast)

model_fitting <- function(data, train.size, h, frequency) {
  # Arguments
  # data: no. of series x length of the series (this assumes data to be without meta information)
  # h: forecast horizon
  # frequency: frequency of the series
  #
  # Return
  # an list of arrays
  
  nr <- nrow(data) # number of series
  nc <- ncol(data) # full length of the series
  
  niter <- nc - train.size - h + 1 # number of iterations to perform
  
  # arrays to store all the necessary information - [number of rolling windows, number of ts, length of the series]
  train.y <- array(, dim = c(niter, nr, nc))
  test.y <- array(, dim = c(niter, nr, h))
  fcasts.base <- array(, dim = c(niter, nr, h))
  fitted.values <- array(, dim = c(niter, nr, nc))
  
  # iterate through each time series
  for (i in 1:nr) {
    
    # whole sample
    single.y <- data[i, ]
    
    # go through all iterations - rolling origin for each series
    for (j in 1:3) {
      print("starting")
      print(j)
      # Start rolling window forecasting
      train.y[j, i, 1:(train.size + j - 1)] <- y.train <-
        ts(single.y[1:(train.size + j - 1)], frequency = frequency)
      test.y[j, i, 1:h] <- single.y[(train.size + j):(train.size + j + h - 1)]
      
      # Fitting ARIMA models
      fit <- auto.arima(y.train)
      fitted.values[j, i, 1:(train.size + j - 1)] <- fitted(fit)
      fcasts.base[j, i, 1:h] <- forecast(fit, h = h)$mean
    }
  }
  list(train = train.y, test = test.y, fcasts = fcasts.base, fitted = fitted.values)
}

# Test code
dataset <- read_csv('labour_small.csv')
ts_df <- dataset[,c(-1, -2)]
ts_meta_df <- dataset[,c(1, 2)]
print(ts_meta_df)

h <- 1
frequency <- 4
train.size <- 68

data_matrix <- data.matrix(ts_df)
out <- model_fitting(data_matrix, train.size, h, frequency)

actual <- out$train
test <- out$test
fcasts <- out$fcasts
fitted <- out$fitted

# t(na.omit(t(actual[1, ,])))# actuals for the first iteration
fitted_new <- t(na.omit(t(fitted[1, ,])))# fitted values for the first iteration
# fcasts[1, ,]
# print(test[1, ,])
