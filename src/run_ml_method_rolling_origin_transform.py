import pandas as pd
from ml_reconcilation_transform import MLReconcile
import sys
import time
import os


def run_ml_reconciliation(dataset, rolling_iter, ml_method, lambda_range_run, seed_to_run, seed_array, hf_levels,
                          lambada_tune, run_saved_model, validate_hf_loss, saved_models=None):
    def check_actual_and_fitted_len(actual, fitted):
        # this is for DeepAR and WaveNet implementations
        if len(actual.columns) > len(fitted.columns):
            meta_data = actual.iloc[:, 0:2]
            ts_values = actual.iloc[:, -len(fitted.columns[2:]):]
            actual_df = pd.concat([meta_data, ts_values], axis=1)
            return actual_df
        else:
            return actual

    hyper_params = {'number_of_layers': 5, 'epochs': [10, 200], 'dropout_rate': [0, 0.5], 'max_norm_value': [0, 10],
                    'reconciliation_loss_lambda': lambda_range_run, 'learning_rate': [0.0001, 0.1]}

    # if hyperparameter tuning is not required tune_hyper_params = False and
    # best_hyper_params parameter should be passed

    if not run_saved_model:
        df_actual = pd.read_csv(f"results/{dataset}/actual_{rolling_iter}.csv")
        df_fitted = pd.read_csv(f"results/{dataset}/{model}_fitted_{rolling_iter}.csv")
        df_forecasts = pd.read_csv(
            f"results/{dataset}/{model}_forecasts_{rolling_iter}.csv")
        df_fitted_transform = pd.read_csv(
            f"results/{dataset}/{model}_fitted_transformed_{rolling_iter}.csv")
        df_fc_transform = pd.read_csv(
            f"results/{dataset}/{model}_forecasts_transformed_{rolling_iter}.csv")

        df_actual = check_actual_and_fitted_len(df_actual, df_fitted)

        ml_model_case = MLReconcile(seed_to_run, df_actual, df_fitted, df_forecasts, df_fitted_transform,
                                    df_fc_transform, hf_levels, seed_array,
                                    hyper_params_tune=hyper_params,
                                    tune_hyper_params=True, tune_lambda=lambada_tune, validate_hf_loss=validate_hf_loss)
    else:
        df_actual = pd.read_csv(f"results/{dataset}/actual_{prev_window}.csv")
        df_fitted = pd.read_csv(f"results/{dataset}/{model}_fitted_{prev_window}.csv")
        df_forecasts = pd.read_csv(
            f"results/{dataset}/{model}_forecasts_{rolling_iter}.csv")
        df_fitted_transform = pd.read_csv(
            f"results/{dataset}/{model}_fitted_transformed_{prev_window}.csv")
        df_fc_transform = pd.read_csv(
            f"results/{dataset}/{model}_forecasts_transformed_{rolling_iter}.csv")

        df_actual = check_actual_and_fitted_len(df_actual, df_fitted)

        ml_model_case = MLReconcile(seed_to_run, df_actual, df_fitted, df_forecasts, df_fitted_transform,
                                    df_fc_transform, hf_levels, seed_array,
                                    hyper_params_tune=hyper_params,
                                    tune_hyper_params=False, tune_lambda=lambada_tune,
                                    validate_hf_loss=validate_hf_loss,
                                    saved_model=run_saved_model, saved_models=saved_models)

    _, forecasts_adjusted_mean, _, best_hyper_params, saved_models = ml_model_case.run_ml_reconciliation()

    forecast_path = f'results/hts/{dataset}'
    hyper_param_path = f'results/hyper_params/{dataset}'

    if not os.path.exists(forecast_path):
        os.makedirs(forecast_path)

    # only saving the mean across the seeds
    forecasts_adjusted_mean.to_csv(f'{forecast_path}/{model}_{ml_method}_{rolling_iter}.csv')

    if not run_saved_model:
        if not os.path.exists(hyper_param_path):
            os.makedirs(hyper_param_path)
        best_hyper_params.to_csv(f'{hyper_param_path}/{model}_{ml_method}_{rolling_iter}.csv')
        return saved_models


def retrain_network(data, window_num, method_name, lambada_val, seed, seed_runs, levels, tune_lambda,
                    times_array, validate_hf_loss, run_saved_model=False):
    st = time.time()
    if not run_saved_model:
        models = run_ml_reconciliation(data, window_num, method_name, lambada_val, seed,
                                       seed_runs,
                                       levels,
                                       tune_lambda, run_saved_model, validate_hf_loss)
    else:
        models = run_ml_reconciliation(data, rolling_window, ml_method_name, lambda_case, seed_value,
                                       seed_runs,
                                       number_of_levels,
                                       tune_lambda, run_saved_model, validate_hf_loss, saved_models=saved_models)
    et = time.time()
    times_array.append(et - st)

    if not run_saved_model:
        return models


if __name__ == '__main__':
    DATASET = ['prison', 'tourism', 'wikipedia', 'labour']
    levels_in_hierarchy = {'prison': 5, 'tourism': 3, 'wikipedia': 6, 'labour': 4}

    data = DATASET[int(sys.argv[1])]
    model = sys.argv[2]

    number_of_levels = levels_in_hierarchy[data]
    seed_value = 1234
    seed_runs = [1234, 3456, 2311, 8311, 5677]

    tune_lambda = bool(sys.argv[3])
    validate_hf_loss = bool(sys.argv[4])
    lambda_range = [[0.01, 0.09], [0.1, 0.9], [1, 4], [0.01, 5]]

    # CASE 1 - validation loss for bottom level
    # CASE 2 - validation loss for complete hierarchy

    if validate_hf_loss:
        case = 'case2'
    else:
        case = 'case1'

    if tune_lambda == False:
        lambda_case = 1
    else:
        lambda_range_idx = int(sys.argv[5])
        lambda_case = lambda_range[lambda_range_idx]
    ml_method_name = f'{case}_lambda_{lambda_case}'
    print(ml_method_name)

    n = 10  # refit after 10 samples
    saved_models = None
    times = []
    rolling_window = sys.argv[6]
    prev_window = None
    rolling_window = int(rolling_window)
    retrain_network(data, rolling_window, ml_method_name, lambda_case, seed_value,
                    seed_runs,
                    number_of_levels,
                    tune_lambda, times, validate_hf_loss, run_saved_model=False)

    exe_time = pd.DataFrame(times, columns=['program time'])
    exe_time.to_csv(f'results/hyper_params/{data}/EXE_{model}_{ml_method_name}.csv')
