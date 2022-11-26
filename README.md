### Hierarchical Reconciliation with Machine Learning

This repository contains the code for the paper "NHR-TFNet:Forecasting Hierarchical Time Series using Non-linear Mappings".

#### Code execution with rolling window evaluation.

```
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

```

#### Current Directory structure
```
|-src
    |- data : original datasets
    |- input_data : data files after pre-processing the datasets includes actual and test files
    |- <dataset_name> : R pre-processing files for each dataset (e.g. prison, tourism)
    |- calculate_rolling_origin_errors.py - error calculation code
    |- cluster_ts.py - K-means clustering for datasets
    |- construct_heirarchy.py - class implementation to create a heirarchical structure for a given dataset
    |- hts-benchmarks.R - benchmark heirarchcial forecasting method's implementation
    |- ml_reconcilation_transform.py - class implemenatation for the ML reconciliation model
    |- run_arima_ets_rolling_origin.py - ARIMA and ETS code execution point for all datasets
    |- run_global_models.py - code to run global models for all datasets
    |- run_hts_benchmarks_rolling_origin.py - run all benchmark in hts-benchmarks.R
    |- run_ml_method_rolling_origin_transform.py - code starting point for ML reconciliation
    |- run_preprocessing.py - run R preprocessing files for all datasets
    |- run_ts_features.py - code to extract time series features from datasets
    |- transform_fitted_global_models.py - code to transform fitted and forecast values of global models
