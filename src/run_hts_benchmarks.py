import rpy2.robjects as robjects

data = 'tourism'
base_model = 'arima'

r_source = robjects.r['source']
r_source('../src/hts-benchmarks.R')

hts_benchmarks = robjects.globalenv['hts_benchmarks']

# defining the args
result_r = hts_benchmarks(data, base_model)