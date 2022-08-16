import rpy2.robjects as robjects

# data = 'wikipedia'
# base_model = 'deepAR_cluster'
#
# DATA = {'prison': 3,
#         'tourism': 10,
#         'labour': 5,
#         'wikipedia': 10}
#
# samples = DATA[data]

r_source = robjects.r['source']
r_source('../src/hts-benchmarks.R')

hts_benchmarks = robjects.globalenv['hts_benchmarks']

# full dataset
# result_r = hts_benchmarks(data, f'input_data/{data}', data, base_model)

# across samples - expanding window
# for sample in range(0, samples):
#     input_file_name = f'input_data/new_data_samples/{data}_{sample}'
#     fc_file_name = f'{data}_{sample}'
#     result_r = hts_benchmarks(data, input_file_name, fc_file_name, base_model)

ROLLING_WINDOWS = {'prison': 3,
                   'tourism': 0,
                   'labour': 0,
                   'wikipedia': 0}

# For ROLLING WINDOW EVALUATION
dataset_name = 'prison'
windows = ROLLING_WINDOWS[dataset_name]
n = 10  # refit after 10 samples
last_iter = 0

for iter_window in range(1, windows + 1):
    if iter_window % n == 1:
        print("Recalculating values", iter_window)
        result_r = hts_benchmarks(dataset_name, '', '', 'arima', iter_window, iter_window)
        last_iter = iter_window
    else:
        result_r = hts_benchmarks(dataset_name, '', '', 'arima', last_iter, iter_window)
