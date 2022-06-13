import rpy2.robjects as robjects

r_source = robjects.r['source']
r_source('ts_features.R')

DATA = {'prison': {'freq': 4},
        'tourism': {'freq': 12},
        'labour': {'freq': 4},
        'wikipedia': {'freq': 7}}

data = 'wikipedia'

get_ts_features = robjects.globalenv['get_ts_features']
get_ts_features(data, DATA[data]['freq'])