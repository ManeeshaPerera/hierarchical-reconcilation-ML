import rpy2.robjects as robjects

r_source = robjects.r['source']
r_source('../src/transform_fitted_values.R')

run_transform_fitted = robjects.globalenv['run_transform_fitted']


ROLLING_WINDOWS = {'prison': 24,
                   'tourism': 120,
                   'labour': 60,
                   'wikipedia': 70}

# For ROLLING WINDOW EVALUATION
dataset_name = 'prison'
windows = ROLLING_WINDOWS[dataset_name]
method = 'deepAR'

for window in range(1, windows + 1):
    run_transform_fitted(dataset_name, method, window)