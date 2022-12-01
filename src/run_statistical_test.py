import rpy2.robjects as robjects

r_source = robjects.r['source']
r_source('statistical_testing.R')

get_stat_results = robjects.globalenv['method_errors']
methods = ['arima', 'ets', 'deepAR', 'waveNet']
datasets = ['prison', 'labour', 'tourism', 'wikipedia']


# for method in methods:
#     for data in datasets:
#         get_stat_results(data, method)

get_stat_results('prison', 'waveNet')