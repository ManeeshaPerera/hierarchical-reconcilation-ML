import rpy2.robjects as robjects

r_source = robjects.r['source']
r_source('../src/hts-benchmarks.R')

hts_benchmarks = robjects.globalenv['hts_benchmarks']


ROLLING_WINDOWS = {'prison': 24,
                   'tourism': 120,
                   'labour': 60,
                   'wikipedia': 70}

# For ROLLING WINDOW EVALUATION
dataset_name = 'wikipedia'
windows = ROLLING_WINDOWS[dataset_name]
n = 10  # refit after 10 samples
last_iter = 0
method = 'ets'

for iter_window in range(1, windows + 1):
    if iter_window % n == 1:
        print("Recalculating values", iter_window)
        result_r = hts_benchmarks(dataset_name, '', '', method, iter_window, iter_window, 'rolling_window')
        last_iter = iter_window
    else:
        result_r = hts_benchmarks(dataset_name, '', '', method, last_iter, iter_window, 'rolling_window')
