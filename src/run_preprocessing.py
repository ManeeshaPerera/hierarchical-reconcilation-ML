import rpy2.robjects as robjects

r_source = robjects.r['source']
r_source('../src/prison/prison_preprocessing.R')