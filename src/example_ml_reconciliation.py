# This code shows an example of how to run the ML reconciliation approach using the MLReconcile class in ml_reconcilation_transform.py file

# example_data_files directory shows an example of the data files to run the code. This include:
# 1. fitted values of the base forecasting model for the entire hierarchy
# 2. transformed fitted values of the base forecasting model for the non-bottom level series in the hierarchy
# 3. actual values corresponding to the fitted values
# 4. forecasts from the base forecasting model for the entire hierarchy
# 5. transformed forecasts from the base forecasting model for the non-bottom level series in the hierarchy
# 6. actual values corresponding to the forecasts -- This is only required for evaluation

# import required packages
import pandas as pd
from ml_reconcilation_transform import MLReconcile

dataset = 'prison'  # dataset name
rolling_iter = '1'  # rolling window iteration number
model = 'arima'  # base forecasting model
hyper_params = {'number_of_layers': 5, 'epochs': [10, 200], 'dropout_rate': [0, 0.5], 'max_norm_value': [0, 10],
                'reconciliation_loss_lambda': [0.01, 0.09], 'learning_rate': [0.0001, 0.1]}  # hyper-parameter range

# read the files
df_actual = pd.read_csv(f"example_data_files/{dataset}/actual_{rolling_iter}.csv")
df_fitted = pd.read_csv(f"example_data_files/{dataset}/{model}_fitted_{rolling_iter}.csv")
df_forecasts = pd.read_csv(f"example_data_files/{dataset}/{model}_forecasts_{rolling_iter}.csv")
df_fitted_transform = pd.read_csv(f"example_data_files/{dataset}/{model}_fitted_transformed_{rolling_iter}.csv")
df_fc_transform = pd.read_csv(f"example_data_files/{dataset}/{model}_forecasts_transformed_{rolling_iter}.csv")

# initialise the model
ml_model_case = MLReconcile(seed_value=1234, actual_data=df_actual, fitted_data=df_fitted, forecasts=df_forecasts,
                            fitted_transform_matrix=df_fitted_transform, fc_transform_matrix=df_fc_transform,
                            number_of_levels=5, seed_runs=[1234, 3456, 2311, 8311, 5677],
                            hyper_params_tune=hyper_params, validate_hf_loss=True)

# run reconciliation -- this will return
forecasts_median, forecast_mean, model_history, best_hyper_params, trained_models = ml_model_case.run_ml_reconciliation()

# pandas dataframe with two columns. First column indicate the time series name and second column shows the reconciled forecasts
print('Median of seed runs', forecasts_median)

# pandas dataframe with two columns. First column indicate the time series name and second column shows the reconciled forecasts
print('Mean of seed runs', forecast_mean)

# pandas dataframe showing the change in loss across epochs for each run across the five different seeds
print('Model training history', model_history)

# pandas dataframe showing the best hyper-parameter values found
print('Best hyper-parameters found', best_hyper_params)

# a python list with keras models as values corresponding to the trained MLP model that can be saved for latter use
print('Trained models', trained_models)
