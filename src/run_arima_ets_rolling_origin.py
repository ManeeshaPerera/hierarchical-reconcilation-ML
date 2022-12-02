import rpy2.robjects as robjects
import sys

DATA = {'prison': {'freq': 4, 'min_train_length': 24},
        'tourism': {'freq': 12, 'min_train_length': 144},
        'labour': {'freq': 4, 'min_train_length': 68},
        'wikipedia': {'freq': 7, 'min_train_length': 324}}

DATASET = ['prison', 'tourism', 'wikipedia', 'labour']

dataset_name = DATASET[int(sys.argv[1])]
arima_ets = sys.argv[2]

min_train_length = DATA[dataset_name]['min_train_length']
freq = DATA[dataset_name]['freq']
horizon = 1

r_source = robjects.r['source']
r_source('../src/rolling_origin_transformed.R')

run_rollin_origin = robjects.globalenv['run_rolling_origin']

if arima_ets == 'arima':
    run_rollin_origin(dataset_name, horizon, freq, min_train_length, 'arima', True)
elif arima_ets == 'ets':
    run_rollin_origin(dataset_name, horizon, freq, min_train_length, 'ets',
                      False)  # passing false as we don't want to save the actual/ test rolling windows again for the same dataset
