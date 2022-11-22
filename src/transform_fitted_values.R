library(readr)
library(forecast)
library(hts)

run_transform_fitted <- function(dataset_name, base_model_name, fitted_iter) {
  dataset <- read.csv(paste0('input_data/', dataset_name, '.csv'))
  ts_df <- dataset[, c(-1, -2)]
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

  fitted <- read_csv(paste0("rolling_window_experiments_transformed/", dataset_name, "/", base_model_name, "_fitted", '_', fitted_iter, '.csv'))[, -(1:2)]
  forecast <- read_csv(paste0("rolling_window_experiments_transformed/", dataset_name, "/", base_model_name, "_forecasts", '_', fitted_iter, '.csv'))[, -(1:2)]

  gmat <- hts:::GmatrixH(hierarchy_nodes)
  s <- hts:::SmatrixM(gmat)
  nr <- nrow(data_matrix) # number of series
  nbts <- ncol(s) # number of bottom level series

  ntop <- nr - nbts
  meta_fitted <- dataset[1:ntop, c(1,2)]

  ut.mat <- cbind(diag(nr - nbts), -s[1:(nr - nbts),])
  base_fitted <- as.matrix(fitted)
  forecast <- as.matrix(forecast)

  transformed_fitted <- as.matrix(ut.mat %*% base_fitted)
  fitted_transform_df <- cbind(meta_fitted, data.frame(transformed_fitted))

  transformed_fc <- as.matrix(ut.mat %*% forecast)
  fc_transform_df <- cbind(meta_fitted, data.frame(transformed_fc))

  write.table(fitted_transform_df, paste0('rolling_window_experiments_transformed/', dataset_name,  "/", base_model_name, "_fitted_transformed_", fitted_iter, '.csv'),
                col.names = TRUE, sep = ",", row.names = FALSE)
  write.table(fc_transform_df, paste0('rolling_window_experiments_transformed/', dataset_name,  "/", base_model_name, "_forecasts_transformed_", fitted_iter, '.csv'),
                col.names = TRUE, sep = ",", row.names = FALSE)
}