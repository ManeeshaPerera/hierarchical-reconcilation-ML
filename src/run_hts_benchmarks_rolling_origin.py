import sys

import rpy2.robjects as robjects

r_source = robjects.r['source']
r_source('../src/hts-benchmarks.R')

hts_benchmarks = robjects.globalenv['hts_benchmarks']

ROLLING_WINDOWS = {'prison': 24,
                   'tourism': 120,
                   'labour': 60,
                   'wikipedia': 70}

# For ROLLING WINDOW EVALUATION
dataset_name = sys.argv[1]  # prison, tourism, labour, wikipedia
windows = ROLLING_WINDOWS[dataset_name]
method = sys.argv[2]  # arima, ets, deepAR or waveNet

for iter_window in range(1, windows + 1):
    result_r = hts_benchmarks(dataset_name, method, iter_window, iter_window, 'rolling_window')
