### Hierarchical Reconciliation with ML

Current Directory structure
```
|-example_code : initial example code to use as reference
|-src
    |- data : original datasets
    |- input_data : data files after pre-processing the datasets includes actual and test files
    |- forecasts : fitted and forecasts by base models 
    (naming convention <dataset_name>_<model_name>_<fitted/forecasts>)
    |- results : reconciled forecasts, best hyper-parameters , model history
    (naming convention <dataset_name>_<model_name>_<adjusted_forecasts/best_params/model_history>)
        |- benchmarks : benchmark heirarchcial forecasting method results
        |- errors : all errors
    |- <dataset_name> : R pre-processing files for each dataset (e.g. prison, tourism)
    |- arima.R - R code to run ARIMA
    |- ets.R - R code to run ETS
    |- run_arima_ets.py - ARIMA and ETS code execution point for all datasets
    |- run_preprocessing.py - run R preprocessing files for all datasets
    |- construct_heirarchy.py - class implementation to create a heirarchical structure for a given dataset
    |- ml_reconcilation.py - class implemenatation for the ML reconciliation model
    |- run_ml_reconciliation.py - code starting point for ML reconciliation
    |- calculate_error.py - error calculation code
    |- run_hts_benchmarks.py - run all benchmark in hts-benchmarks.R
    |- hts-benchmarks.R - benchmark heirarchcial forecasting method's implementation
```

How to run the code:

1. First run_preprocessing.py - you may need to provide the path for the preprocessing R file relevant to the dataset (e.g. for the prison dataset `'../src/prison/prison_preprocessing.R'`)
2. Run the base model python file (e.g. for arima/ets run `run_arima_ets.py`)
3. Run the ML reconcilation `run_ml_reconciliation.py` (provide the dataset name, base model name and the number of levels in the dataset hierarchy)
4. Run `hts-benchmarks.R` to run benchmark approaches
5. Finally Calculate errors - `run calculate_error.py` (this will create a file with the calculated errors under the results/errors directory) 

