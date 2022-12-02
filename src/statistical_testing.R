# install.packages('tsutils')
library(tsutils)
library(readr)

method_errors <- function (dataset_name, method){
  sample_wise_errors <- read_csv(paste0("results/errors/", dataset_name, "/error_percentages/", method, "_sample_wise_errors.csv"))
  sample_wise_errors <- sample_wise_errors[, (2:length(sample_wise_errors))]
  sample_wise_errors <- as.matrix(sample_wise_errors)
  if (dataset_name == 'prison' | dataset_name == 'wikipedia'){
    colnames(sample_wise_errors) <- c('BU', 'OLS', 'WLS', 'MinTShrink', 'ERM', 'NHR-TFNet')
  }
  else {
    colnames(sample_wise_errors) <- c('BU', 'OLS', 'WLS', 'MinTSample', 'MinTShrink', 'ERM', 'NHR-TFNet')
  }
  nemenyi(sample_wise_errors, plottype ='matrix')
  path_matrix <- paste0("paper_figs/", method, "_", dataset_name, "_matrix", ".pdf")
  quartz.save(path_matrix, type="pdf")

  nemenyi(sample_wise_errors, plottype ='mcb')
  path_matrix <- paste0("paper_figs/", method, "_", dataset_name, "_mcb", ".pdf")
  quartz.save(path_matrix, type="pdf")
}


