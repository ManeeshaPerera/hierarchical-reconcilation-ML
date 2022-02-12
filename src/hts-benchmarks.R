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

actual <- as.matrix(read_csv("input_data/prison_actual.csv")[, -(1:2)])
base_fitted <- as.matrix(read_csv("forecasts/prison_arima_fitted.csv")[, -(1:2)])
forecasts <- as.matrix(read_csv("forecasts/prison_arima_forecasts.csv")[, -(1:2)])


nodes <- list(8, rep(2, 8), rep(2, 16), rep(2, 32)) # create the hierarchy structure
gmat <- hts:::GmatrixH(nodes) # creating grouping structure
summing_matrix <- hts:::SmatrixM(gmat) # create the summing matrix for the hierarchy

all_ts_count <- nrow(summing_matrix) # number of totoal time series in the heirarchy
bottom_level_ts_count <- ncol(summing_matrix) # bottom level time series count

top_ts_count <- all_ts_count - bottom_level_ts_count # top level time series count

residuals <- t(actual - base_fitted)  # residuals for the fitted values (y - y_hat) - two matrices with sizes T X M where T is the number of observations and M is the total number of time series in the heirarchy

covariance_matrix <- crossprod(residuals) / nrow(residuals) # create sample covariance matrix

diagonal_covariance_matrix <- diag(x = diag(covariance_matrix)) # create diagonal covariance matrix

shrink_estimator <- hts:::shrink.estim(residuals, diagonal_covariance_matrix)[[1]] # calculating the shrinkage estimator

# OLS
ols_fc <- summing_matrix %*% solve(t(summing_matrix) %*% summing_matrix) %*% t(summing_matrix) %*% forecasts

# WLS
wls_fc <- summing_matrix %*% solve(t(summing_matrix) %*% solve(diagonal_covariance_matrix) %*% summing_matrix) %*% t(summing_matrix) %*% solve(diagonal_covariance_matrix) %*% forecasts