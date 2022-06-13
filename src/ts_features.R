library(tsfeatures)
library(reshape2)
library(readr)

get_ts_features <- function (data, freq){
  input_data <- read_csv(paste("input_data/", data, "_actual.csv", sep=''))
  input_data <- input_data[, 3:ncol(input_data)] # remove level and description info
  all_ts <-nrow(input_data)
  ts_list <- list()

  # for each ts calculate the features
  for (ts_i in 1:all_ts) {
    ts_val <- input_data[ts_i, ]
    ts_val <- as.data.frame(ts_val)
    ts_val <- melt(ts_val)
    ts_val <- ts(ts_val['value'], frequency=freq)
    ts_list[ts_i] <- list(ts_val)
  }

  features <-tsfeatures(ts_list)
  # print(ts_list)
  write.table(features, paste("input_data/ts_features/", data, ".csv", sep=''), col.names = TRUE, sep = ",")
}

