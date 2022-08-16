import rpy2.robjects as robjects

DATA = {'prison': {'freq': 4, 'horizon': 8, 'min_train_length': 24},
        'tourism': {'freq': 12, 'horizon': 12, 'min_train_length': 144},
        'labour': {'freq': 4, 'horizon': 12, 'min_train_length': 68},
        'wikipedia': {'freq': 7, 'horizon': 7, 'min_train_length': 324}}

dataset_name = 'prison'
min_train_length = DATA[dataset_name]['min_train_length']
freq = DATA[dataset_name]['freq']
horizon = DATA[dataset_name]['horizon']

r_source = robjects.r['source']
r_source('../src/rolling_origin.R')

run_rollin_origin = robjects.globalenv['run_rolling_origin']

run_rollin_origin(dataset_name, horizon, freq, min_train_length, 'arima', True)
run_rollin_origin(dataset_name, horizon, freq, min_train_length, 'ets', False)


