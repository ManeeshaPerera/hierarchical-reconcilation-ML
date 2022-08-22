# Title     : Benchmark approaches: BU, OLS, WLS, MinT(Sample), MinT(Shrink), ERM
# Created by: Maneesha Perera
# Created on: 12/2/22

#install packages('readr')
#install packages('tidyverse')
#install packages('tsibble')
#install packages('lubridate')
#install packages('gt')
#install packages('hts')

library(readr)
library(tidyverse)
library(tsibble)
library(lubridate)
library(gt)
library(hts)
library(Matrix)

# Notations used below in comments. T - number of observations, M- total number of series, B - number of bottom level series, A - number of top level series, H- forecast horizon


run_benchmarks <- function (dataset_name, actual, fitted, forecasts){
  actual_len <- length(actual)
  base_fitted <- as.matrix(fitted) # M X T matrix
  fitted_len <- length(fitted)
  start_idx <- actual_len - fitted_len + 1
  # for DeepAR and WaveNet fitted values were less than the actual so adding this operation to make the lengths equal
  actual <- actual[, (start_idx:actual_len)]
  actual <- as.matrix(actual)# M X T matrix
  if (dataset_name == 'prison'){
    hierarchy_nodes <- list(8, rep(2, 8), rep(2, 16), rep(2, 32))
  }
  else if (dataset_name == 'tourism'){
    hierarchy_nodes <- list(7, c(14, 7, 13, 12, 5, 21, 5))
  }
  else if (dataset_name == 'wikipedia'){
    hierarchy_nodes <- list(3, c(2, 1, 2), rep(4, 5), c(rep(9,3), 6, rep(9, 3), 6, 7, 9, 6, 3, 6, 9, 8, 6, rep(9, 3), 6), c(2,  1,  1, 21,  3,  6,  2,  1,  1,  3,  4,  4, 33, 24, 14,  4,  3,
                                                                    4,  2,  1,  2, 21,  3,  5,  2,  1,  1,  3,  1, 17,  4,  3,  1,  2,
                                                                    1,  1, 21,  3,  6,  2,  1,  1,  3,  4,  4, 33, 24, 14,  4,  3,  4,
                                                                    2,  1,  2, 21,  3,  5,  2,  1,  1,  3,  1, 17,  4,  3,  1,  1, 19,
                                                                    2,  4,  2,  1,  1,  3,  1,  3, 29, 19,  9,  4,  2,  4,  1, 18,  2,
                                                                    2,  1,  1, 12,  1,  1,  1, 17,  2,  2,  1,  1,  3,  4,  4, 33, 24,
                                                                    14,  4,  2,  4,  1,  2, 19,  3,  3,  1,  1,  1,  3,  1, 17,  4,  3,
                                                                    1,  2,  1,  1, 21,  3,  6,  2,  1,  1,  3,  4,  4, 33, 24, 14,  4,
                                                                    3,  4,  2,  1,  2, 21,  3,  5,  2,  1,  1,  3,  1, 17,  4,  3,  1))
  }
  else if (dataset_name == 'labour'){
    hierarchy_nodes <- list(8, rep(2, 8), rep(2, 16))
  }
  gmat <- hts:::GmatrixH(hierarchy_nodes) # creating grouping structure
  summing_matrix <- hts:::SmatrixM(gmat) # create the summing matrix for the hierarchy - M X B

  all_ts_count <- nrow(summing_matrix) # number of total time series in the heirarchy (M)
  bottom_level_ts_count <- ncol(summing_matrix) # bottom level time series count (B)

  top_ts_count <- all_ts_count - bottom_level_ts_count # top level time series count (A)

  j_matrix <- cbind(matrix(0, bottom_level_ts_count, top_ts_count), diag(bottom_level_ts_count)) # creates a B X M matrix

  residuals <- t(actual - base_fitted)  # residuals for the fitted values (y - y_hat) and transpose - T X M

  covariance_matrix <- crossprod(residuals) / nrow(residuals) # create sample covariance matrix - M X M

  diagonal_covariance_matrix <- diag(x = diag(covariance_matrix)) # create diagonal covariance matrix - M X M

  shrink_estimator <- hts:::shrink.estim(residuals, diagonal_covariance_matrix)[[1]] # calculating the shrinkage estimator - M X M

  # Running benchmark reconciliation methods

  # Bottom Up

  bottom_up_fc <- summing_matrix %*% j_matrix %*% forecasts # M X H

  # OLS
  ols_fc <- summing_matrix %*% solve(t(summing_matrix) %*% summing_matrix) %*% t(summing_matrix) %*% forecasts # M X H

  # WLS
  wls_fc <- summing_matrix %*% solve(t(summing_matrix) %*% solve(diagonal_covariance_matrix) %*% summing_matrix) %*% t(summing_matrix) %*% solve(diagonal_covariance_matrix) %*% forecasts # M X H

  # MinT(Sample)
  mint_sample_fc <- tryCatch({
    summing_matrix %*% solve(t(summing_matrix) %*% solve(covariance_matrix) %*% summing_matrix) %*% t(summing_matrix) %*% solve(covariance_matrix) %*% forecasts
  }, error = function(e) {
    message('Could not calculate MinT Sample')
    return('NA')
  }) # M X H or NA if an error occurs

  # Mint(Shrink)
  mint_shrink_fc <- summing_matrix %*% solve(t(summing_matrix) %*% solve(shrink_estimator) %*% summing_matrix) %*% t(summing_matrix) %*% solve(shrink_estimator) %*% forecasts # M X H

  # Empreical Risk Minimizer (ERM)
  svd_matrix <- svd(t(base_fitted)) # singular decomposition matrix -> give three components u, v, d
  non_negative <- svd_matrix$d[abs(svd_matrix$d) > sqrt(.Machine$double.eps)]
  tmp <- svd_matrix$u[,1:length(non_negative)] %*% diag(x = 1/non_negative) %*% t(svd_matrix$v[, 1:length(non_negative)])

  erm_fc <- summing_matrix %*% j_matrix %*% actual %*% tmp %*% forecasts # M X H

  if (typeof(mint_sample_fc) == 'S4'){
        list(bottomup = as.data.frame(as.matrix(bottom_up_fc)), ols = as.data.frame(as.matrix(ols_fc)), wls = as.data.frame(as.matrix(wls_fc)), mintsample = as.data.frame(as.matrix(mint_sample_fc)), mintshrink = as.data.frame(as.matrix(mint_shrink_fc)), erm = as.data.frame(as.matrix(erm_fc)))
    }
  else {
      list(bottomup = as.data.frame(as.matrix(bottom_up_fc)), ols = as.data.frame(as.matrix(ols_fc)), wls = as.data.frame(as.matrix(wls_fc)), mintsample = FALSE, mintshrink = as.data.frame(as.matrix(mint_shrink_fc)), erm = as.data.frame(as.matrix(erm_fc)))
  }

}

hts_benchmarks <- function (dataset_name, input_file_path, filename_fc, base_model_name, fitted_iter, fc_iter, experiment){
  if (experiment == 'expanding_window') {
    # exapanding window experiments
    fitted <- read_csv(paste0("forecasts/new_data_samples/", filename_fc, "_", base_model_name, "_fitted.csv"))[, -(1:2)]
    forecasts <- as.matrix(read_csv(paste0("forecasts/new_data_samples/", filename_fc, "_", base_model_name, "_forecasts.csv"))[, -(1:2)]) # M X H matrix
    actual <- read_csv(paste0(input_file_path, "_actual.csv"))[, -(1:2)] # M X T matrix
  }
  else if (experiment == 'rolling_window'){
    # rolling window experiments
    fitted <- read_csv(paste0("rolling_window_experiments/", dataset_name, "/", base_model_name, "_fitted", '_', fitted_iter, '.csv'))[, -(1:2)]
    forecasts <- as.matrix(read_csv(paste0("rolling_window_experiments/", dataset_name, "/", base_model_name, "_forecasts", '_', fc_iter, '.csv'))[, -(1:2)]) # M X H matrix
    actual <- read_csv(paste0("rolling_window_experiments/", dataset_name, "/", "actual", '_', fitted_iter, '.csv'))[, -(1:2)] # M X T matrix
  }

  output <- run_benchmarks(dataset_name, actual, fitted, forecasts)
  bottom_up <- output$bottomup
  ols <- output$ols
  wls <- output$wls
  mintshrink <- output$mintshrink
  erm <- output$erm
  mintsample <- output$mintsample

  if (experiment == 'expanding_window'){
    write.table(bottom_up, paste0('results/expanding_window_results/benchmarks/', filename_fc, "_", base_model_name, "_bottomup.csv"),
                col.names = TRUE, sep = ",")
    write.table(ols, paste0('results/expanding_window_results/benchmarks/', filename_fc, "_", base_model_name, "_ols.csv"),
                col.names = TRUE, sep = ",")
    write.table(wls, paste0('results/expanding_window_results/benchmarks/', filename_fc, "_", base_model_name, "_wls.csv"),
                col.names = TRUE, sep = ",")
    if (mintsample != FALSE){
      write.table(mintsample, paste0('results/expanding_window_results/benchmarks/', filename_fc, "_", base_model_name, "_mintsample.csv"),
                  col.names = TRUE, sep = ",")
    }
    write.table(mintshrink, paste0('results/expanding_window_results/benchmarks/', filename_fc, "_", base_model_name, "_mintshrink.csv"),
                col.names = TRUE, sep = ",")
    write.table(erm, paste0('results/expanding_window_results/benchmarks/', filename_fc, "_", base_model_name, "_erm.csv"),
                col.names = TRUE, sep = ",")
  }
  else if (experiment == 'rolling_window'){
    write.table(bottom_up, paste0('rolling_window_experiments/hts/', dataset_name, "/", base_model_name, "_bottomup_", fc_iter, '.csv'),
                col.names = TRUE, sep = ",")
    write.table(ols, paste0('rolling_window_experiments/hts/', dataset_name, "/", base_model_name, "_ols_", fc_iter, '.csv'),
                col.names = TRUE, sep = ",")
    write.table(wls, paste0('rolling_window_experiments/hts/', dataset_name, "/", base_model_name, "_wls_", fc_iter, '.csv'),
                col.names = TRUE, sep = ",")
    if (mintsample != FALSE){
      write.table(mintsample, paste0('rolling_window_experiments/hts/', dataset_name, "/", base_model_name, "_mintsample_", fc_iter, '.csv', sep = ''),
                  col.names = TRUE, sep = ",")
    }
    write.table(mintshrink, paste0('rolling_window_experiments/hts/', dataset_name, "/", base_model_name, "_mintshrink_", fc_iter, '.csv'),
                col.names = TRUE, sep = ",")
    write.table(erm, paste0('rolling_window_experiments/hts/', dataset_name, "/", base_model_name, "_erm_", fc_iter, '.csv'),
                col.names = TRUE, sep = ",")
  }
}

