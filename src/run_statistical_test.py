import rpy2.robjects as robjects

r_source = robjects.r['source']
r_source('statistical_testing.R')

# get_ts_features = robjects.globalenv['get_ts_features']
# get_ts_features(data, DATA[data]['freq'])