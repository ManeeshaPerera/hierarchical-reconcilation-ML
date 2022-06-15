library(hts)
library(plyr)
library(foreach)
library(doSNOW)
library(matlib)
library(ggplot2)
library(randomForest)
library(rpart)
library(kernlab)
library(gbm)

setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

###################################################################################################
#Read data & Create validation set
###################################################################################################
#Read file (from 01/1998 to 12/2017)
input <- read.csv("data.csv", stringsAsFactors = F)
#Keep what needed
input <- input[5:(nrow(input)-3),]
input$SuperWEB2.tm. <- NULL
#Define names and ids
TS_id <- as.character(input[1,3:ncol(input)]) #Ids of series at the bottom level
TS_name <- as.character(input[2,3:ncol(input)]) #Names of series at the bottom level
#Exclude dates
input <- input[4:nrow(input),]
input$X = input$X.1 <- NULL
colnames(input) <- TS_id


for (j in 1:ncol(input)){ input[,j] <- as.numeric(input[,j]) }
#Create hts
input <- ts(input, frequency = 12, start = c(1998,1))
colnames(input) <- TS_id
data <- hts(input, characters = c(1, 1, 1))
structure <- window(data, start = c(1998,1))
m_series <- nrow(smatrix(structure)) #Total number of series at all levels
b_series <- length(structure$labels[[(length(structure$labels)-1)+1]]) #Number of series at the bottom level
input <- NULL

#Get base forecasts for training and forecast purposes
fh = 12 #Forecasting horizon considered
origin <- 60 #Starting origin (5 years)
tslength <- length(data$bts[,1]) #Available observations
counter <- 0
Forecast_file = Residuals_file <- NULL
while ((origin+counter+fh) <= tslength){
  
  #Fetch data
  ts_sample <- head(aggts(data), origin+counter+fh)
  train_sample <- head(ts_sample, length(ts_sample[,1])-fh)
  test_sample <- tail(ts_sample, fh)
  
  #Generate base forecasts
  all_mod_ets = all_mod_arima = all_mod_theta <- NULL
  all_for_ets = all_for_arima = all_for_theta <- NULL
  all_fit_ets = all_fit_arima = all_fit_theta <- NULL
  for (tsid in 1:ncol(train_sample)){
    all_mod_ets[length(all_mod_ets)+1] <- list(ets(train_sample[,tsid]))
  }
  
  all_for_ets <- foreach(tsid=1:ncol(train_sample), .combine='cbind') %do% as.numeric(forecast(all_mod_ets[[tsid]], h = fh)$mean)
  
  all_fit_ets <- foreach(tsid=1:ncol(train_sample), .combine='cbind') %do% as.numeric((train_sample[,tsid] - fitted(all_mod_ets[[tsid]])))
  
  #Save results
    
    allf <- data.frame(all_for_ets)
    Residuals_temp <- all_fit_ets
    
    colnames(allf) <- paste0("F_", as.character(colnames(test_sample)))
    test_sample_c <- as.data.frame(test_sample)
    colnames(test_sample_c) <- paste0("A_", as.character(colnames(test_sample_c)))
    #Combine with actual values
    allf <- cbind(allf, test_sample_c)
    #Add info for period
    allf$rep <- counter +1
    allf$fh <- c(1:fh)
    allf$period <- c((length(train_sample[,1])+1) : (length(train_sample[,1])+fh))
    allf$origin <- length(train_sample[,1])
    allf$model <- model_id
    #Combine with the rest
    Forecast_file <- rbind(Forecast_file, allf)
    
    Residuals_temp <- as.data.frame(Residuals_temp)
    colnames(Residuals_temp) <- colnames(test_sample)
    Residuals_temp$rep <- counter +1
    Residuals_temp$period <- c(1:length(train_sample[,1]))
    Residuals_temp$origin <- length(train_sample[,1])
    Residuals_temp$model <- model_id
    Residuals_file <- rbind(Residuals_file, Residuals_temp)
  
  counter <- counter + 1
  
}
ts_sample = train_sample = test_sample = allf = Residuals_temp <- NULL
write.csv(Forecast_file, "Tourism_Forecast_file.csv", row.names = FALSE)
write.csv(Residuals_file, "Tourism_Residual_file.csv", row.names = FALSE)