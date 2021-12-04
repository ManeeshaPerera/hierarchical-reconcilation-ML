# Title     : TODO
# Objective : TODO
# Created by: kasun
# Created on: 1/3/21

library(tidyverse)
library(readr)
library(tsibble)
library(lubridate)
library(fabletools)
library(forecast)

set.seed(1234)

setwd("/Users/kasun/PycharmProjects/htsML")

header <- read_csv("tourism.csv", n_max = 3, col_names = FALSE)

header <- header %>%
  t() %>%
  as_tibble() %>%
  fill(V1:V2, .direction = "down") %>%
  unite(name, V1, V2, V3, na.rm = TRUE, sep = "/") %>%
  pull()

tourism <- read_csv("tourism.csv", skip = 3, col_names = header)
tourism <- tourism %>%
  fill(Year, .direction = "down")
tourism <- tourism %>%
  pivot_longer(!Year:Month, names_to = "Variable", values_to = "Trips") %>%
  separate(Variable, c("State", "Region", "Purpose"), sep = "/")

tourism <- tourism %>%
  mutate(Year_month = yearmonth(ym(paste(Year, Month)))) %>%
  as_tsibble(key = c(State, Region, Purpose), index = Year_month) %>%
  select(Year_month, State:Trips)

tourism <- tourism %>%
  mutate(State = recode(State,
    `New South Wales` = "NSW",
    `Northern Territory` = "NT",
    `Queensland` = "QLD",
    `South Australia` = "SA",
    `Tasmania` = "TAS",
    `Victoria` = "VIC",
    `Western Australia` = "WA"
  ))


bottom_level_ts <- tourism %>% group_by(State, Region) %>%
  summarise(Trips = sum(Trips)) %>%
  ungroup() %>% as_tibble() %>%
  unite(col = Region, 1,2, sep = "-", remove =TRUE) %>%
  pivot_wider(names_from = Year_month, values_from = Trips) %>%
  rename(Description = Region) %>% mutate(Level = 3) %>%
  relocate(Level)

second_level_ts <- tourism %>%
  group_by(State) %>%
  summarise(Trips = sum(Trips)) %>%
  ungroup() %>% as_tibble() %>%
  pivot_wider(names_from = Year_month, values_from = Trips) %>%
  rename(Description = State) %>% mutate(Level = 2) %>%
  relocate(Level)

top_ts <- tourism %>% as_tibble() %>%
  group_by(Year_month) %>%
  summarise(Trips = sum(Trips)) %>%
  ungroup() %>%
  pivot_wider(names_from = Year_month, values_from = Trips) %>%
  mutate(Level = 1, Description = "Aggregated") %>%
  relocate(Level, Description)


all_level_ts <- rbind(top_ts, second_level_ts, bottom_level_ts)
all_level_ts_test <- all_level_ts[, (ncol(all_level_ts) - 11):ncol(all_level_ts)]
all_level_ts_train <- all_level_ts[, 1: (ncol(all_level_ts) - 12)]

f_model <- function (x){
  model_fitting <- auto.arima(ts(as.numeric(x), frequency = 12))
  model_residuals <- as.numeric(model_fitting$fitted)
  model_forecasts <- forecast(model_fitting, h =12)
  model_forecasts <- as.numeric(model_forecasts$mean)
  model_results <- list(model_residuals,model_forecasts)
  return (model_results)
}

model_results_write <- function(df){
  ts_df <- df[,c(-1, -2)]
  ts_meta_df <- df[,c(1, 2)]
  #ts_df <- ts(ts_df, frequency = 12)
  ts_df_list <- split(ts_df, seq(nrow(ts_df)))

  model_results_list <- lapply(ts_df_list, function(x) f_model(x))

  residual_list <- lapply(model_results_list, `[[`, 1)
  forecast_list <- lapply(model_results_list, `[[`, 2)

  residual_df <- do.call(rbind, residual_list)
  forecasts_df <- do.call(rbind, forecast_list)

  residual_df <- cbind(ts_meta_df,residual_df)
  forecasts_df <- cbind(ts_meta_df, forecasts_df)

  write.table(residual_df, "tourism_arima_fitted.csv",
              col.names = TRUE, row.names = FALSE, sep = ",")
  write.table(forecasts_df, "tourism_arima_forecasts.csv",
              col.names = TRUE, row.names = FALSE, sep = ",")

  #results_files <- list(residual_df, forecasts_df)
  #return (results_files)
}

model_results_write(all_level_ts_train)

meta_info <- all_level_ts_train[, c(1,2)]

final_test <- cbind(meta_info, all_level_ts_test)
write.table(final_test, "tourism_model_test.csv", col.names = TRUE, row.names = FALSE, sep = ",")



#residual_file = as.data.frame(results_files[[1]])
#residual_file = as.data.frame(results_files[[1]])