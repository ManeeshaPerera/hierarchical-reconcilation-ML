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

hts_benchmarks <- function (dataset_name, input_file_path, filename_fc, base_model_name){
  actual <- as.matrix(read_csv(paste(input_file_path, "_actual.csv", sep=''))[, -(1:2)]) # M X T matrix
  base_fitted <- as.matrix(read_csv(paste("forecasts/new_data_samples/", filename_fc, "_", base_model_name, "_fitted.csv", sep=''))[, -(1:2)]) # M X T matrix
  forecasts <- as.matrix(read_csv(paste("forecasts/new_data_samples/", filename_fc, "_", base_model_name, "_forecasts.csv", sep=''))[, -(1:2)]) # M X T matrix
  if (dataset_name == 'prison'){
    hierarchy_nodes <- list(8, rep(2, 8), rep(2, 16), rep(2, 32))
  }
  else if (dataset_name == 'tourism'){
    hierarchy_nodes <- list(7, c(14, 7, 13, 12, 5, 21, 5))
  }
  else if (dataset_name == 'wikipedia'){
    hierarchy_nodes <- list(4, rep(3, 4), rep(c(2,1,2), 4), c(rep(38, 2), 30, 24, 38, rep(93, 2), 74, 92, 93, rep(38, 2), 25, 31, 38, rep(29,2), 14, rep(29,2)))
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

  # save the files
  write.table(as.data.frame(as.matrix(bottom_up_fc)), paste('results/expanding_window_results/benchmarks/', filename_fc, "_", base_model_name, "_bottomup.csv", sep = ''),
              col.names = TRUE, sep = ",")
  write.table(as.data.frame(as.matrix(ols_fc)), paste('results/expanding_window_results/benchmarks/', filename_fc, "_", base_model_name, "_ols.csv", sep = ''),
              col.names = TRUE, sep = ",")
  write.table(as.data.frame(as.matrix(wls_fc)), paste('results/expanding_window_results/benchmarks/', filename_fc, "_", base_model_name, "_wls.csv", sep = ''),
              col.names = TRUE, sep = ",")
  if (typeof(mint_sample_fc) == 'S4'){
    write.table(as.data.frame(as.matrix(mint_sample_fc)), paste('results/expanding_window_results/benchmarks/', filename_fc, "_", base_model_name, "_mintsample.csv", sep = ''),
                col.names = TRUE, sep = ",")
  }
  write.table(as.data.frame(as.matrix(mint_shrink_fc)), paste('results/expanding_window_results/benchmarks/', filename_fc, "_", base_model_name, "_mintshrink.csv", sep = ''),
                col.names = TRUE, sep = ",")
  write.table(as.data.frame(as.matrix(erm_fc)), paste('results/expanding_window_results/benchmarks/', filename_fc, "_", base_model_name, "_erm.csv", sep = ''),
              col.names = TRUE, sep = ",")
}

