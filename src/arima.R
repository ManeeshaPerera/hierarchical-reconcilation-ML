# Title     : ARIMA
# Created by: Maneesha Perera
# Created on: 4/12/21
library(readr)
library(forecast)

f_model <- function (x, frequency, horizon){
  model_fitting <- auto.arima(ts(as.numeric(x), frequency = frequency))
  model_residuals <- as.numeric(model_fitting$fitted)
  model_forecasts <- forecast(model_fitting, h =horizon)
  model_forecasts <- as.numeric(model_forecasts$mean)
  model_results <- list(model_residuals,model_forecasts)
  return (model_results)
}

model_results_write <- function(df, data, frequency, horizon){
  ts_df <- df[,c(-1, -2)]
  ts_meta_df <- df[,c(1, 2)]
  ts_df_list <- split(ts_df, seq(nrow(ts_df)))

  model_results_list <- lapply(ts_df_list, function(x) f_model(x, frequency, horizon))

  residual_list <- lapply(model_results_list, `[[`, 1)
  forecast_list <- lapply(model_results_list, `[[`, 2)

  residual_df <- do.call(rbind, residual_list)
  forecasts_df <- do.call(rbind, forecast_list)

  residual_df <- cbind(ts_meta_df,residual_df)
  forecasts_df <- cbind(ts_meta_df, forecasts_df)

  write.table(residual_df, paste(data, "arima_fitted.csv", sep = '_'),
              col.names = TRUE, row.names = FALSE, sep = ",")
  write.table(forecasts_df, paste(data, "arima_forecasts.csv", sep = '_'),
              col.names = TRUE, row.names = FALSE, sep = ",")
}


prison <- c("filename" = "input_data/prison_actual.csv", "freq" = 4, "horizon"= 8, 'name' = 'forecasts/prison')
all_level_ts_train <-read_csv(prison['filename'])
model_results_write(all_level_ts_train, prison['name'], as.integer(prison['freq']), as.integer(prison['horizon']))


tourism <- c("filename" = "input_data/tourism_actual.csv", "freq" = 12, "horizon"= 12, 'name' = 'forecasts/tourism')
all_level_ts_train <-read_csv(tourism['filename'])
model_results_write(all_level_ts_train, tourism['name'], as.integer(tourism['freq']), as.integer(tourism['horizon']))

wikipedia <- c("filename" = "input_data/wikipedia_actual.csv", "freq" = 1, "horizon"= 7, 'name' = 'forecasts/wikipedia')
all_level_ts_train <-read_csv(wikipedia['filename'])
model_results_write(all_level_ts_train, wikipedia['name'], as.integer(wikipedia['freq']), as.integer(wikipedia['horizon']))

