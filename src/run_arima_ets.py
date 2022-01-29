import rpy2.robjects as robjects

r_source = robjects.r['source']
r_source('../src/arima.R')
r_source('../src/ets.R')