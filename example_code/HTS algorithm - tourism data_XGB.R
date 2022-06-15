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

error_cal <- function(insample, outsample, forecasts, ppy){
  
  #insample=Insample[,i];outsample=Forecasts_temp[,i];forecasts=Outsample[,i];ppy=12
  masep<-mean(abs(diff(insample, lag = ppy)))
  
  outsample <- as.numeric(outsample)
  forecasts <- as.numeric(forecasts)
  
  mase <- mean(abs(outsample-forecasts))/masep
  amse <- abs(mean(outsample-forecasts))/masep
  rmsse <- sqrt(mean(abs(outsample-forecasts)^2)/(masep^2))
  
  output <- c(mase,amse,rmsse) ; names(output) <- c("MASE","AMSE","RMSSE")
 
  return(output)
}
reconcile <- function(frc_base){
  
  S <- smatrix(structure)
  W <- diag(rowSums(S))
  
  fb <- t(as.matrix(as.data.frame(frc_base)))
  
  frcst <- S%*%inv(t(S)%*%inv(W)%*%S)%*%t(S)%*%inv(W)%*%fb
  return(frcst)
}
bu <- function(frc_base){
  
  S <- smatrix(structure)
  m <- nrow(S) #total number of series
  K <- length(structure$labels)-1 #Numer of levels in the hierarchy
  mK <- length(structure$labels[[K+1]]) #Numer of series at the bottom level of the hierarchy
  P <- cbind(matrix(0, nrow = mK,  ncol = m-mK) ,   diag(1, mK))
  
  fb <- t(as.matrix(as.data.frame(frc_base)))
  frcst <- S%*%P%*%fb
  return(frcst)
}
td <- function(frc_base){
  
  S <- smatrix(structure)
  m <- nrow(S) #total number of series
  K <- length(structure$labels)-1 #Numer of levels in the hierarchy
  mK <- length(structure$labels[[K+1]]) #Numer of series at the bottom level of the hierarchy
  
  weigtsTD <- colSums(allts(structure))/colSums(allts(structure))[1]
  weigtsTD <- matrix(tail(weigtsTD, mK))
  
  P <- cbind( weigtsTD,  matrix(0, nrow = mK,  ncol = m-1))
  
  fb <- t(as.matrix(as.data.frame(frc_base)))
  frcst <- S%*%P%*%fb
  return(frcst)
}

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
    all_mod_arima[length(all_mod_arima)+1] <- list(auto.arima(train_sample[,tsid]))
    all_mod_theta[length(all_mod_theta)+1] <- list(thetaf(train_sample[,tsid]))
  }
  
  all_for_ets <- foreach(tsid=1:ncol(train_sample), .combine='cbind') %do% as.numeric(forecast(all_mod_ets[[tsid]], h = fh)$mean)
  all_for_arima <- foreach(tsid=1:ncol(train_sample), .combine='cbind') %do% as.numeric(forecast(all_mod_arima[[tsid]], h = fh)$mean)
  all_for_theta <- foreach(tsid=1:ncol(train_sample), .combine='cbind') %do% as.numeric(forecast(all_mod_theta[[tsid]], h = fh)$mean)
  
  all_fit_ets <- foreach(tsid=1:ncol(train_sample), .combine='cbind') %do% as.numeric((train_sample[,tsid] - fitted(all_mod_ets[[tsid]])))
  all_fit_arima <- foreach(tsid=1:ncol(train_sample), .combine='cbind') %do% as.numeric((train_sample[,tsid] - fitted(all_mod_arima[[tsid]])))
  all_fit_theta <- foreach(tsid=1:ncol(train_sample), .combine='cbind') %do% as.numeric((train_sample[,tsid] - fitted(all_mod_theta[[tsid]])))
  
  #Save results
  for (model_id in 1:3){
    
    if (model_id==1){
      allf <- data.frame(all_for_ets)
      Residuals_temp <- all_fit_ets
    }else if (model_id==2){
      allf <- data.frame(all_for_arima)
      Residuals_temp <- all_fit_arima
    }else{
      allf <- data.frame(all_for_theta)
      Residuals_temp <- all_fit_theta
    }
    
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
  }
  
  counter <- counter + 1
  
}
ts_sample = train_sample = test_sample = allf = Residuals_temp <- NULL
write.csv(Forecast_file, "Tourism_Forecast_file.csv", row.names = FALSE)
write.csv(Residuals_file, "Tourism_Residual_file.csv", row.names = FALSE)

Forecast_file <- read.csv("Tourism_Forecast_file.csv")
Residuals_file <- read.csv("Tourism_Residual_file.csv")


save.image("WD_ets_arima_theta.Rdata")
#######################################################################
#######################################################################

#######################################################################
#Reconcile 
########################################################################
#load("WD_ets_arima_theta.Rdata")
DoTheJob_ML_XGB <- function(x){
  # xgb_cv_bayes <-
  #   function(max_depth=4,
  #            min_child_weight=1,
  #            gamma=0,
  #            eta=0.01,
  #            subsample = 0.6,
  #            colsample_bytree =0.6,
  #            nrounds = 200,
  #            early_stopping_rounds = 50,
  #            nfold =  10){
  #     
  #     cv <- xgb.cv(params = list(
  #       booster = "gbtree",
  #       eta = eta,
  #       max_depth = max_depth,
  #       min_child_weight = min_child_weight,
  #       gamma = gamma,
  #       subsample = subsample,
  #       colsample_bytree = colsample_bytree,
  #       objective = 'reg:linear',
  #       eval_metric = "rmse"),
  #       data = dftest,
  #       nround = nrounds,
  #       nfold=  10,
  #       prediction = FALSE,
  #       showsd = TRUE,
  #       early_stopping_rounds = 50,
  #       verbose = 0)
  #     
  #     list(Score = cv$evaluation_log[, max(test_rmse_mean)],
  #          Pred = 0)
  #   }
  # 
  # 
  # bayesian_results <- rBayesianOptimization::BayesianOptimization(
  #   FUN = xgb_cv_bayes,
  #   bounds = list(max_depth = c(2L, 10L),
  #                 colsample_bytree = c(0.3, 1),
  #                 subsample = c(0.3,1),
  #                 min_child_weight = c(1L, 10L),
  #                 eta = c(0.01, 0.05),
  #                 gamma = c(0, 5)),
  #   init_grid_dt = NULL, init_points = 4, 
  #   n_iter = 7,
  #   acq = "ucb", kappa = 2.576, eps = 0.0,
  #   verbose = TRUE)
  
  
  # params <- list(objective = "reg:linear", eval.metric = "rmse", max.depth = bayesian_results$Best_Par[1],
  #                colsample_bytree = bayesian_results$Best_Par[2], subsample = bayesian_results$Best_Par[3],
  #                eta = bayesian_results$Best_Par[5], 
  #                min_child_weight = bayesian_results$Best_Par[4],
  #                gamma = bayesian_results$Best_Par[6])
  
  params <- list(objective = "reg:linear", eval.metric = "rmse", max.depth = hyper_ml[[1]][[3]],
                 colsample_bytree = hyper_ml[[1]][[4]], subsample = hyper_ml[[1]][[5]],
                 eta = hyper_ml[[1]][[6]], 
                 min_child_weight = hyper_ml[[1]][[7]],
                 gamma = hyper_ml[[1]][[8]])
  
  dtrain <- as.matrix(cbind(act_ml_array[,x], frc_ml_array)) ; colnames(dtrain)[1] <- "Y"
  #dtrain1 <- (as.numeric(unlist(dtrain)), ncol = 16, nrow = 26)
  dftrain <- xgb.DMatrix(dtrain, label= dtrain[,1])
  
  model_ML <-  xgb.train(params, dftrain, 300, verbose = -1)
  
  return(model_ML)
}

Summary_error<- c()
originid=169
for (originid in seq(169,max(Forecast_file$origin),1)){
  
  model_type <- "ARIMA" ; model_type_id <- 2 #Consider a base model of type ETS (1), ARIMA (2) or Theta (3)
  
  tempf <- Forecast_file[Forecast_file$origin == originid,]
  tempf <- tempf[tempf$model==model_type_id,]
  Outsample <- tempf[,(1:m_series)+m_series]
  Forecasts <- tempf[,1:m_series]
  Insample <- aggts(data)[(1:originid),]
  
  # Forecasts_st <- t(reconcile(Forecasts)) #WLS-Structural
  # Forecasts_bu <- t(bu(Forecasts)) #BU
  # Forecasts_td <- t(td(Forecasts)) #TD
  
  res = Residuals_file[(Residuals_file$model==model_type_id)&(Residuals_file$origin==originid),]
  res$model = res$origin = res$period = res$rep <- NULL
  row.names(res) <- NULL 
  res <- as.matrix(res) ; forc <- as.matrix(Forecasts)
  #Forecasts_mint <- MinT(forc, get_nodes(structure), residual = res, 
     #         covariance = "shr", keep = "all", algorithms = "lu")
  
  #Train ML
  data_ml <- Forecast_file[Forecast_file$period <= originid,]
  data_ml <- data_ml[data_ml$fh==1,] #Consider just one step ahead forecasts
  data_ml <- data_ml[data_ml$model==model_type_id,] 
  data_ml$origin = data_ml$period = data_ml$rep = data_ml$fh = data_ml$model <- NULL
  frc_ml_array <- data_ml[,c(1:m_series)] #Forecasts for all levels
  act_ml_array <- data_ml[,c(1:m_series)+m_series] #Actual values of bottom level
  act_ml_array <- act_ml_array[,c((m_series-b_series+1):m_series)]
  
  Models_ml <- NULL
  for (x in 1:ncol(act_ml_array)){
    Models_ml[length(Models_ml)+1] <- list(DoTheJob_ML(x))
  }
  
  Forecasts_ml_b <- NULL 
  for (iid in 1:b_series){
    Forecasts_ml_b <- cbind(Forecasts_ml_b, as.matrix(predict(Models_ml[[iid]] , newdata = Forecasts)))
  }
  Forecasts_ml <- Forecasts
  Forecasts_ml[,1:(ncol(Forecasts_ml)-b_series)] <- 0
  Forecasts_ml[,(ncol(Forecasts_ml)-b_series+1):ncol(Forecasts_ml)] <- Forecasts_ml_b
  Forecasts_ml_b <- NULL
  Forecasts_ml <- t(bu(Forecasts_ml))
  
  # just saving ML forecasts
  Forecasts_ml_s <- cbind(Forecasts_ml,originid )
  Forecasts_ml_saved <- (rbind(Forecasts_ml_s, Forecasts_ml_saved))
  
  ##########################################################################
  #End ML reconciliation
  ##########################################################################
  # for (mid in 1:5){
  #   if (mid==1){
  #     Forecasts_temp <- Forecasts_bu
  #   }else if (mid==2){
  #     Forecasts_temp <- Forecasts_td
  #   }else if (mid==3){
  #     Forecasts_temp <- Forecasts_st
  #   }else if (mid==4){
  #     Forecasts_temp <- Forecasts_mint
  #   }else{
      Forecasts_temp <- Forecasts_ml
   # }
    
    error_list <- NULL
    for (i in 1:ncol(Outsample)){
      error_list <- rbind(error_list, error_cal(Insample[,i],Forecasts_temp[,i],Outsample[,i],ppy=12))
    }
    Errors <- data.frame(error_list) ; colnames(Errors) <- c("MASE","AMSE","RMSSE")
    Errors$Tags <- as.character(unlist(colnames(Outsample)))
    for (i in 1:nrow(Errors)){
      Errors$Tags[i] <- substr(Errors$Tags[i],3, nchar(Errors$Tags[i]))
    }
    Errors$Level <- NA
    for (idd in 1:nrow(Errors)){
      for (ldd in 1:length(structure$labels)){
        if (Errors$Tags[idd] %in% structure$labels[[ldd]]){
          Errors$Level[idd] <- ldd
        }
      }
    }
    Errors$Tags <- NULL
    if (mid==1){
      Errors$mid <- "BU"
    }else if (mid==2){
      Errors$mid <- "TD"
    }else if (mid==3){
      Errors$mid <- "SS"
    }else if (mid==4){
      Errors$mid <- "MinT"
    }else if (mid==5){
      Errors$mid <- "ML"
    }
    
    Errors$origin <- originid
    Summary_error <- rbind(Summary_error, Errors)
  }
#}


Errors_plot <- ddply(Summary_error, .(mid, Level,origin), colwise(mean))
Errors_plot$mid <- as.factor(Errors_plot$mid)
Errors_plot$Level <- as.factor(Errors_plot$Level)
ggplot(data=Errors_plot, aes(x=origin, y=MASE, shape=mid, colour=Level)) + geom_line() + geom_point()


Errors_agg <- ddply(Summary_error, .(mid, Level), colwise(mean))
Errors_agg$origin <- NULL
ddply(Errors_agg, .(mid), colwise(mean))

write.csv(Summary_error, paste0("Tourism_Summary_error_ML_XGB",model_type,".csv"), row.names = F)
write.csv(Forecasts_ml_saved, paste0("Tourism_forecast_ML_XGB",model_type,".csv"), row.names = F)
