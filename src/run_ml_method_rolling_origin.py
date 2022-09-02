import pandas as pd
from ml_reconcilation import MLReconcile
import sys
import time


def run_ml_reconciliation(dataset, rolling_iter, ml_method, lambda_range_run, seed_to_run, seed_array, hf_levels,
                          lambada_tune, run_saved_model, saved_models=None):
    df_actual = pd.read_csv(f"rolling_window_experiments/{dataset}/actual_{rolling_iter}.csv")
    df_fitted = pd.read_csv(f"rolling_window_experiments/{dataset}/{model}_fitted_{rolling_iter}.csv")
    df_forecasts = pd.read_csv(f"rolling_window_experiments/{dataset}/{model}_forecasts_{rolling_iter}.csv")

    # this is for DeepAR and WaveNet implementations
    if len(df_actual.columns) > len(df_fitted.columns):
        meta_data = df_actual.iloc[:, 0:2]
        ts_values = df_actual.iloc[:, -len(df_fitted.columns[2:]):]
        df_actual = pd.concat([meta_data, ts_values], axis=1)

    hyper_params = {'number_of_layers': 5, 'epochs': [10, 200], 'dropout_rate': [0, 0.5], 'max_norm_value': [0, 10],
                    'reconciliation_loss_lambda': lambda_range_run, 'learning_rate': [0.0001, 0.1]}

    # if hyper-parameter tuning is not required tune_hyper_params = False and
    # best_hyper_params parameter should be passed

    if not run_saved_model:
        ml_model_case = MLReconcile(seed_to_run, df_actual, df_fitted, df_forecasts, hf_levels, seed_array,
                                    hyper_params_tune=hyper_params,
                                    tune_hyper_params=True, tune_lambda=lambada_tune)
    else:
        ml_model_case = MLReconcile(seed_to_run, df_actual, df_fitted, df_forecasts, hf_levels, seed_array,
                                    hyper_params_tune=hyper_params,
                                    tune_hyper_params=False, tune_lambda=lambada_tune,
                                    saved_model=run_saved_model, saved_models=saved_models)
    _, forecasts_adjusted_mean, _, best_hyper_params, saved_models = ml_model_case.run_ml_reconciliation()

    # only saving the mean across the seeds
    forecasts_adjusted_mean.to_csv(
        f'rolling_window_experiments/hts/{dataset}/{dataset}_new/{model}_{ml_method}_{rolling_iter}.csv')

    if not run_saved_model:
        best_hyper_params.to_csv(
            f'rolling_window_experiments/hyper_params/{dataset}/{dataset}_new/{model}_{ml_method}_{rolling_iter}.csv')
        return saved_models


if __name__ == '__main__':
    times = []
    DATASET = ['prison', 'tourism', 'wikipedia', 'labour']
    rolling_windows = {'prison': 24,
                       'tourism': 120,
                       'labour': 60,
                       'wikipedia': 70}
    levels_in_hierarchy = {'prison': 5, 'tourism': 3, 'wikipedia': 6, 'labour': 4}

    data = DATASET[int(sys.argv[1])]
    model = sys.argv[2]
    num_rolling_windows = rolling_windows[data]

    number_of_levels = levels_in_hierarchy[data]
    seed_value = 1234
    seed_runs = [1234, 3456, 2311, 8311, 5677]

    tune_lambda = bool(sys.argv[3])
    validate_hf_loss = bool(sys.argv[4])
    lambda_range_idx = int(sys.argv[5])
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
        lambda_case = lambda_range[lambda_range_idx]
    ml_method_name = f'{case}_lambda_{lambda_case}'
    print(ml_method_name)

    n = 10  # refit after 10 samples
    saved_models = None
    times = []
    rolling_window = int(sys.argv[6])

    st = time.time()
    run_ml_reconciliation(data, rolling_window, ml_method_name, lambda_case, seed_value,
                          seed_runs,
                          number_of_levels,
                          tune_lambda, run_saved_model=False)
    et = time.time()
    times.append(et - st)

    # we need to only calculate hyper-params after like 10th window
    # for rolling_window in range(1, num_rolling_windows + 1):
    # get the start time

    # if rolling_window % n == 1:
    #     # get the start time
    #     st = time.time()
    #     saved_models = run_ml_reconciliation(data, rolling_window, ml_method_name, lambda_case, seed_value,
    #                                          seed_runs,
    #                                          number_of_levels,
    #                                          tune_lambda, remove_skip, run_saved_model=False)
    #     et = time.time()
    #     times.append(et-st)
    #
    # else:
    #     st = time.time()
    #     run_ml_reconciliation(data, rolling_window, ml_method_name, lambda_case, seed_value, seed_runs,
    #                           number_of_levels,
    #                           tune_lambda, remove_skip, run_saved_model=True, saved_models=saved_models)
    #     et = time.time()
    #     times.append(et - st)

    exe_time = pd.DataFrame(times, columns=['program time'])
    exe_time.to_csv(f'rolling_window_experiments/hyper_params/{data}/{data}_new/EXE_{model}_{ml_method_name}.csv')
