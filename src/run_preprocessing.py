import rpy2.robjects as robjects

r_source = robjects.r['source']
## Code to only clean the dataset and create the heirarchy
r_source('../src/prison/prison_preprocessing.R')
r_source('../src/tourism/tourism_preprocessing.R')
r_source('../src/wikipedia/wikipedia_preprocessing.R')
r_source('../src/labour/labour_preprocessing.R')


## Code to only clean the dataset
# r_source('../src/prison/data_cleaning.R')
# r_source('../src/tourism/data_cleaning.R')
# r_source('../src/labour/data_cleaning.R')
# r_source('../src/wikipedia/data_cleaning.R')