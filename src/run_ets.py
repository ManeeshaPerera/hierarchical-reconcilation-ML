import rpy2.robjects as robjects

DATA = {'prison': {'freq': 4, 'horizon': 8, 'samples': 3},
        'tourism': {'freq': 12, 'horizon': 12, 'samples': 10},
        'labour': {'freq': 4, 'horizon': 12, 'sample': 5},
        'wikipedia': {'freq': 7, 'horizon': 7, 'sample': 10}}

dataset_name = 'prison'
samples = DATA[dataset_name]['samples']
freq = DATA[dataset_name]['freq']
horizon = DATA[dataset_name]['horizon']

r_source = robjects.r['source']
r_source('../src/ets.R')

run_ets = robjects.globalenv['run_ets']

# full dataset
result_r = run_ets(f'{dataset_name}_actual.csv', freq, horizon, dataset_name)

# samples
for sample in range(0, samples):
    result_r = run_ets(f'data_samples/{dataset_name}_{sample}_actual.csv', freq, horizon, f'{dataset_name}_{sample}')
