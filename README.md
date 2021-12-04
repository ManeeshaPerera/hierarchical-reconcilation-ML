### Hierarchical Reconciliation with ML

Current folder structure
```
|-example_code : initial example code to use as reference
|-src
    |- data : original datasets
    |- input_data : data files after pre-processing the datasets includes actual and test files
    |- forecasts : fitted and forecasts by base models (naming convention <dataset_name>_<model_name>_<fitted/forecasts>)
    |- <dataset_name> : R pre-processing files for each dataset (e.g. prison, tourism)
    |- reconcilation : python code for reconcilation (yet to be added)
    |- arima.R - R code to run ARIMA
    |- run_arima.py - ARIMA code execution point for all datasets
    |- run_preprocessing.py - run R preprocessing files for all datasets
```
