import rpy2.robjects as robjects

data = 'wikipedia'
base_model = 'ets'

DATA = {'prison': 3,
        'tourism': 10,
        'labour': 5,
        'wikipedia': 10}

samples = DATA[data]

r_source = robjects.r['source']
r_source('../src/hts-benchmarks.R')

hts_benchmarks = robjects.globalenv['hts_benchmarks']

# full dataset
# result_r = hts_benchmarks(data, f'input_data/{data}', data, base_model)

# across samples
for sample in range(0, samples):
    input_file_name = f'input_data/new_data_samples/{data}_{sample}'
    fc_file_name = f'{data}_{sample}'
    result_r = hts_benchmarks(data, input_file_name, fc_file_name, base_model)
