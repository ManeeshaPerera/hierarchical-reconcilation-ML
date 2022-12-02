library(readr)
library(forecast)
library(hts)

model_fitting <- function(data, nbts, s, train.size, h, frequency, arima_ets) {
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
    single.y <- data[i,]

    # go through all iterations - rolling origin for each series
    for (j in 1:niter) {
      # Start rolling window forecasting
      train.y[j, i, 1:(train.size + j - 1)] <- y.train <-
        ts(single.y[1:(train.size + j - 1)], frequency = frequency)
      test.y[j, i, 1:h] <- single.y[(train.size + j):(train.size + j + h - 1)]

      # Fitting ARIMA/ ETS models
      if (arima_ets == 'arima') {
        fit <- auto.arima(y.train)
      }
      else if (arima_ets == 'ets') {
        fit <- ets(y.train)
      }

      fitted.values[j, i, 1:(train.size + j - 1)] <- fitted(fit)
      fcasts.base[j, i, 1:h] <- forecast(fit, h = h)$mean
    }
  }
  ut.mat <- cbind(diag(nr - nbts), -s[1:(nr - nbts),])

  fitted.values.transformed <- array(, dim = c(niter, nr - nbts, nc))
  fcasts.base.transformed <- array(, dim = c(niter, nr - nbts, h))

  for (i in 1:niter) {
    fitted.values.transformed[i, ,] <- as.matrix(ut.mat %*% fitted.values[i, ,])
    fcasts.base.transformed[i, ,] <- as.matrix(ut.mat %*% fcasts.base[i, ,])
  }

  list(train = train.y, test = test.y, fcasts = fcasts.base,
       fitted = fitted.values, fcasts.transformed = fcasts.base.transformed,
       fitted.transformed = fitted.values.transformed, niter = niter)
}

# run code
run_rolling_origin <- function(dataset_name, h, freq, min_train_length, arima_ets, store_train_test) {
  dataset <- read.csv(paste0('input_data/', dataset_name, '.csv'))

  ts_df <- dataset[, c(-1, -2)]
  ts_meta_df <- dataset[, c(1, 2)]

  frequency <- freq
  train.size <- min_train_length

  data_matrix <- data.matrix(ts_df)

  if (dataset_name == 'prison') {
    hierarchy_nodes <- list(8, rep(2, 8), rep(2, 16), rep(2, 32))
  }
  else if (dataset_name == 'tourism') {
    hierarchy_nodes <- list(7, c(14, 7, 13, 12, 5, 21, 5))
  }
  else if (dataset_name == 'wikipedia') {
    hierarchy_nodes <- list(3, c(2, 1, 2), rep(4, 5), c(rep(9, 3), 6, rep(9, 3), 6, 7, 9, 6, 3, 6, 9, 8, 6, rep(9, 3), 6), c(2, 1, 1, 21, 3, 6, 2, 1, 1, 3, 4, 4, 33, 24, 14, 4, 3,
                                                                                                                             4, 2, 1, 2, 21, 3, 5, 2, 1, 1, 3, 1, 17, 4, 3, 1, 2,
                                                                                                                             1, 1, 21, 3, 6, 2, 1, 1, 3, 4, 4, 33, 24, 14, 4, 3, 4,
                                                                                                                             2, 1, 2, 21, 3, 5, 2, 1, 1, 3, 1, 17, 4, 3, 1, 1, 19,
                                                                                                                             2, 4, 2, 1, 1, 3, 1, 3, 29, 19, 9, 4, 2, 4, 1, 18, 2,
                                                                                                                             2, 1, 1, 12, 1, 1, 1, 17, 2, 2, 1, 1, 3, 4, 4, 33, 24,
                                                                                                                             14, 4, 2, 4, 1, 2, 19, 3, 3, 1, 1, 1, 3, 1, 17, 4, 3,
                                                                                                                             1, 2, 1, 1, 21, 3, 6, 2, 1, 1, 3, 4, 4, 33, 24, 14, 4,
                                                                                                                             3, 4, 2, 1, 2, 21, 3, 5, 2, 1, 1, 3, 1, 17, 4, 3, 1))
  }
  else if (dataset_name == 'labour') {
    hierarchy_nodes <- list(8, rep(2, 8), rep(2, 16))
  }

  gmat <- hts:::GmatrixH(hierarchy_nodes)
  s <- hts:::SmatrixM(gmat)
  nbts <- ncol(s) # number of bottom level series
  nr <- nrow(data_matrix)

  ntop <- nr - nbts

  meta_fitted <- dataset[1:ntop, c(1,2)]

  out <- model_fitting(data_matrix, nbts, s, train.size, h, frequency, arima_ets)

  actual <- out$train
  test <- out$test
  fcasts <- out$fcasts
  fitted <- out$fitted
  fitted_transformed <- out$fitted.transformed
  fcasts_transformed <- out$fcasts.transformed
  rolling_windows <- out$niter

  # iterated through each rolling window and store the results

  for (itr in 1:rolling_windows){
    actual_itr <- t(na.omit(t(actual[itr, ,])))# actuals for the iteration
    fitted_itr <- t(na.omit(t(fitted[itr, ,])))# fitted values for the iteration
    fitted_tranform_itr <- t(na.omit(t(fitted_transformed[itr, ,]))) # fitted transform values for the iteration - transofrmation will not have bottom level series

    fcasts_itr <- fcasts[itr, ,]
    fcasts_itr_transform <- fcasts_transformed[itr, ,]
    test_itr <- test[itr, ,]

    # convert to data frame and add meta info
    actual_df <- cbind(ts_meta_df, data.frame(actual_itr))
    fitted_df <- cbind(ts_meta_df, data.frame(fitted_itr))
    fitted_transform_df <- cbind(meta_fitted, data.frame(fitted_tranform_itr))

    fcasts_df <- data.frame(fcasts_itr)
    fcasts_transform_df <- data.frame(fcasts_itr_transform)

    colnames(fcasts_df) <- seq(1:h)
    colnames(fcasts_transform_df) <- seq(1:h)

    fcasts_df <- cbind(ts_meta_df, fcasts_df)
    fcasts_transform_df <- cbind(meta_fitted, fcasts_transform_df)

    test_df <- data.frame(test_itr)
    colnames(test_df) <- seq(1:h)
    test_df <- cbind(ts_meta_df, test_df)

    if (store_train_test == TRUE) {
    write.table(actual_df, paste('results', dataset_name, paste(paste('actual', itr, sep = '_'), 'csv', sep = '.'), sep = '/'),
              col.names = TRUE, row.names = FALSE, sep = ",")
    write.table(test_df, paste('results', dataset_name, paste(paste('test', itr, sep = '_'), 'csv', sep = '.'), sep = '/'),
              col.names = TRUE, row.names = FALSE, sep = ",")
    }
    write.table(fitted_df, paste('results', dataset_name, paste(paste(arima_ets, 'fitted', itr, sep = '_'), 'csv', sep='.'), sep = '/'),
              col.names = TRUE, row.names = FALSE, sep = ",")
    write.table(fcasts_df, paste('results', dataset_name, paste(paste(arima_ets, 'forecasts', itr, sep = '_'), 'csv', sep = '.'), sep = '/'),
              col.names = TRUE, row.names = FALSE, sep = ",")

    write.table(fitted_transform_df, paste('results', dataset_name, paste(paste(arima_ets, 'fitted_transformed', itr, sep = '_'), 'csv', sep = '.'), sep = '/'),
              col.names = TRUE, row.names = FALSE, sep = ",")
    write.table(fcasts_transform_df, paste('results', dataset_name, paste(paste(arima_ets, 'forecasts_transformed', itr, sep = '_'), 'csv', sep = '.'), sep = '/'),
              col.names = TRUE, row.names = FALSE, sep = ",")

  }
#
}