### Hierarchical Reconciliation with Machine Learning

This repository contains the code for the paper "NHR-TFNet:Forecasting Hierarchical Time Series using Non-linear Mappings".

#### Code execution with rolling window evaluation.


1. First run_preprocessing.py - you may need to provide the path for the preprocessing R file relevant to the dataset (e.g. for the prison dataset `'../src/prison/prison_preprocessing.R'`). This will preprocess the data and create a hierarchical dataset under the `<input data>` directory with the naming format <dataset>.csv (e.g., prison.csv).
2. Create a directory `results/<dataset_name>` (e.g.,results/prison), 
3. Run the base model python files
   1. For ARIMA and ETS run `run_arima_ets_rolling_origin.py` with two parameters dataset id (0-prison, 1-tourism, 2-wikipedia, 3-labour) and method name as input (e.g., python run_arima_ets_rolling_origin.py 0 'arima') -- during this stage the rolling windows will be created 
      - Under rolling_window_experiments_transformed/<data_set> directory the actual, fitted, forecasts, and tranformed fitted and forecasts will be created using the below naming convension
        - `actual_<window_id>.csv` (e.g., `actual_1.csv`) indicate actual values to evaluate the fitted values for given rolling window id
        - `<model_name>_fitted_<window_id>.csv` (e.g., `arima_fitted_1.csv`) are the fitted values for the given rolling window id
        - `<model_name>_fitted_transformed_<window_id>.csv` (e.g., `arima_fitted_transformed_1.csv`) are the transformed fitted values for the given rolling window id
        - `<model_name>_forecasts_<window_id>.csv` (e.g., `arima_forecasts_1.csv`) are the forecasts for the given rolling window id
        - `<model_name>_forecasts_transformed_<window_id>.csv` (e.g., `arima_forecasts_transformed_1.csv`) are the transformed forecasts for the given rolling window id
        - `test_<window_id>.csv` (e.g., `test_1.csv`) are the actual values to evaluate the forecasts for given rolling window id
   2. For DeepAR and WaveNet - 
      - `run_ts_features.py` to get time series features for each dataset. This will create a csv with time series feature under `input_data/ts_features/<dataset_name>.csv`
      - `cluster_ts.py` to cluster the time series. This will create a csv with clustering information under `input_data/ts_features/<dataset_name>_clusters.csv'` 
      - `run_global_models.py` to get forecasts and fitted values as shown below:
            `python run_global_models.py <dataset index> <model index> <window_id>` (e.g.,`python run_global_models.py 0 0 1` - this will run the deepAR for rolling window 1 of the prison dataset)
      - `transform_fitted_global_models.py` to transform fitted values and forecasts. e.g., `python transform_fitted_global_models.py 'waveNet'` 
4. Run `run_hts_benchmarks_rolling_origin.py` to run benchmark approaches 
    - Create a directory `results/hts/<dataset_name>` - this directory will store reconciled forecasts
    - run python file -- `python run_hts_benchmarks_rolling_origin.py <dataset_name> <base_method_name>` (e.g., `python run_hts_benchmarks_rolling_origin.py 'prison' 'arima'`)
    - This will run all benchmarks BU, OLS, WLS, MinT(Sample), MinT(Shrink), ERM for all rolling windows of the given dataset using the given base methods fitted and forecast values
    - A csv file will be created under `results/hts/<dataset_name>` with the naming `<base_method_name><benchmark_method_name>_<window_id>.csv` (e.g., `arima_mintsample_1.csv`)
5. Run the ML reconciliation `run_ml_method_rolling_origin_transform.py` 
    -  `python run_ml_method_rolling_origin_transform.py <dataset_index (0-prison, 1-tourism, 2-wikipedia, 3-labour)> <base_method_name (arima, ets, deepAR, waveNet)> <Tune_lambda value or not (True or '')> <validation loss for the whole hierachy or not (True or '')> 'lambda range index (0 - [0.01, 0.09], 1- [0.1, 0.9], 2 - [1, 4], 3-[0.01, 5])' 'rolling window id (1, 2, ..)`
       (e.g., `python run_ml_method_rolling_origin_transform.py 0 arima True True 0 1` - this will run the ML reconciliation method for the Prison dataset, arima as base method, with lambda tuned between [0.01-0.09], considering the validation loss for the whole hierarchy, and running for rolling window 1)
    - This will store the reconciled forecasts in `results/hts/<dataset_name>` with the naming `<base_method_name>_case<1,2>_lambda_<lambda_range>_<window_id>.csv`
      (e.g., `arima_case2_lambda_[0.01-0.09]_1.csv` will be created by running `python run_ml_method_rolling_origin_transform.py 0 arima True True 0 1`)
6. Calculate errors - `calculate_rolling_origin_errors.py` (this will create a csv with MSE and MAE improvements under `results/errors/<dataset_name>/error_percentages`) 


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
