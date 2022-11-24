### Hierarchical Reconciliation with ML

Current Directory structure
```
|-example_code : initial example code to use as reference
|-src
    |- data : original datasets
    |- input_data : data files after pre-processing the datasets includes actual and test files
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

How to run the code with rolling window evaluation:

1. First run_preprocessing.py - you may need to provide the path for the preprocessing R file relevant to the dataset (e.g. for the prison dataset `'../src/prison/prison_preprocessing.R'`)
2. Run the base model python files
   1. For ARIMA and ETS run `run_arima_ets_rolling_origin.py` -- during this stage the rolling windows will be created
   2. For DeepAR and WaveNet - 
      - `run_ts_features.py` to get time series features for each dataset
      - `cluster_ts.py` to cluster the time series 
      - `run_global_models.py` to get forecasts and fitted values
      - `transform_fitted_global_models.py` to transform fitted values and forecasts
3. Run `hts-benchmarks.R` to run benchmark approaches
4. Run the ML reconciliation `run_ml_method_rolling_origin_transform.py` (provide the dataset name, base model name and the number of levels in the dataset hierarchy)
5. Calculate errors - `calculate_rolling_origin_errors.py` (this will create a file with the calculated errors) 

