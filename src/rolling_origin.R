library(readr)
library(forecast)

model_fitting <- function(data, train.size, h, frequency, arima_ets) {
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
    print(paste("completed series", i, "out of", nr, "time series", sep = ' '))

    # whole sample
    single.y <- data[i, ]

    # go through all iterations - rolling origin for each series
    for (j in 1:niter) {
      # Start rolling window forecasting
      train.y[j, i, 1:(train.size + j - 1)] <- y.train <-
        ts(single.y[1:(train.size + j - 1)], frequency = frequency)
      test.y[j, i, 1:h] <- single.y[(train.size + j):(train.size + j + h - 1)]

      # Fitting ARIMA/ ETS models
      if (arima_ets == 'arima'){
        fit <- auto.arima(y.train)
      }
      else if (arima_ets == 'ets'){
        fit <- ets(y.train)
      }

      fitted.values[j, i, 1:(train.size + j - 1)] <- fitted(fit)
      fcasts.base[j, i, 1:h] <- forecast(fit, h = h)$mean
    }
  }
  list(train = train.y, test = test.y, fcasts = fcasts.base, fitted = fitted.values, niter = niter)
}

# run code
run_rolling_origin <- function (dataset_name, h, freq, min_train_length, arima_ets, store_train_test){
  dataset <- read.csv(paste0('input_data/', dataset_name, '.csv'))

  ts_df <- dataset[,c(-1, -2)]
  ts_meta_df <- dataset[,c(1, 2)]

  frequency <- freq
  train.size <- min_train_length

  data_matrix <- data.matrix(ts_df)
  out <- model_fitting(data_matrix, train.size, h, frequency, arima_ets)

  actual <- out$train
  test <- out$test
  fcasts <- out$fcasts
  fitted <- out$fitted
  rolling_windows <- out$niter

  # iterated through each rolling window and store the results

  for (itr in 1:rolling_windows){
    actual_itr <- t(na.omit(t(actual[itr, ,])))# actuals for the first iteration
    fitted_itr <- t(na.omit(t(fitted[itr, ,])))# fitted values for the first iteration
    fcasts_itr <- fcasts[itr, ,]
    test_itr <- test[itr, ,]

    # convert to data frame and add meta info
    actual_df <- cbind(ts_meta_df, data.frame(actual_itr))
    fitted_df <- cbind(ts_meta_df, data.frame(fitted_itr))
    fcasts_df <- cbind(ts_meta_df, data.frame(fcasts_itr))
    test_df <- cbind(ts_meta_df, data.frame(test_itr))

  if (store_train_test == TRUE) {
    write.table(actual_df, paste('rolling_window_experiments', dataset_name, paste(paste('actual', itr, sep = '_'), 'csv', sep = '.'), sep = '/'),
              col.names = TRUE, row.names = FALSE, sep = ",")
    write.table(test_df, paste('rolling_window_experiments', dataset_name, paste(paste('test', itr, sep = '_'), 'csv', sep = '.'), sep = '/'),
              col.names = TRUE, row.names = FALSE, sep = ",")
  }
  write.table(fitted_df, paste('rolling_window_experiments', dataset_name, paste(paste(arima_ets, 'fitted', itr, sep = '_'), 'csv', sep='.'), sep = '/'),
              col.names = TRUE, row.names = FALSE, sep = ",")
  write.table(fcasts_df, paste('rolling_window_experiments', dataset_name, paste(paste(arima_ets, 'forecasts', itr, sep = '_'), 'csv', sep = '.'), sep = '/'),
              col.names = TRUE, row.names = FALSE, sep = ",")

}

}
