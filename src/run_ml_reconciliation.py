import pandas as pd
from ml_reconcilation import MLReconcile
import sys


def run_ml_reconciliation(data_actual, model_filename, save_file, lambda_range_run, seed_to_run, seed_array, hf_levels,
                          lambada_tune, remove_skip):
    df_actual = pd.read_csv(f"input_data/{data_actual}_actual.csv")
    df_fitted = pd.read_csv(f"forecasts/new_data_samples/{model_filename}_fitted.csv")
    df_forecasts = pd.read_csv(f"forecasts/new_data_samples/{model_filename}_forecasts.csv")

    # this is for DeepAR and WaveNet implementations
    if len(df_actual.columns) > len(df_fitted.columns):
        meta_data = df_actual.iloc[:, 0:2]
        ts_values = df_actual.iloc[:, -len(df_fitted.columns[2:]):]
        df_actual = pd.concat([meta_data, ts_values], axis=1)

    hyper_params = {'number_of_layers': 5, 'epochs': [10, 200], 'dropout_rate': [0, 0.5], 'max_norm_value': [0, 10],
                    'reconciliation_loss_lambda': lambda_range_run, 'learning_rate': [0.0001, 0.1]}

    # if hyper parameter tuning is not required tune_hyper_params = False and
    # best_hyper_params parameter should be passed

    ml_model_case = MLReconcile(seed_to_run, df_actual, df_fitted, df_forecasts, hf_levels, seed_array,
                                hyper_params_tune=hyper_params,
                                tune_hyper_params=True, tune_lambda=lambada_tune, remove_skip=remove_skip)
    forecasts_adjusted_median, forecasts_adjusted_mean, model_history, best_hyper_params = ml_model_case.run_ml_reconciliation()

    # new change - only saving the mean across the seeds
    forecasts_adjusted_mean.to_csv(
        f'results/expanding_window_results/ml_fc/{model_filename}_{save_file}.csv')
    model_history.to_csv(f'results/expanding_window_results/model_history/{model_filename}_{save_file}.csv')
    best_hyper_params.to_csv(
        f'results/expanding_window_results/best_params/{model_filename}_{save_file}.csv')


if __name__ == '__main__':
    DATASET = ['prison', 'tourism', 'wikipedia', 'labour']
    dataset_samples = {'prison': 3, 'tourism': 10, 'wikipedia': 10, 'labour': 5}
    levels_in_hierarchy = {'prison': 5, 'tourism': 3, 'wikipedia': 6, 'labour': 4}
    data = DATASET[int(sys.argv[1])]
    model = sys.argv[2]
    file_name = f'{data}_{model}'

    number_of_levels = levels_in_hierarchy[data]
    seed_value = 1234
    seed_runs = [1234, 3456, 2311, 8311, 5677]

    tune_lambda = bool(sys.argv[3])
    validate_hf_loss = bool(sys.argv[4])
    lambda_range_idx = int(sys.argv[5])
    remove_skip = bool(sys.argv[6])
    lambda_range = [[0.01, 0.09], [0.1, 0.9], [1, 4], [0.01, 5]]
    l1_regularizer = False

    # CASE 1 - validation loss for bottom level
    # CASE 2 - validation loss for complete hierarchy
    # CASE 3 - regularizer + validation loss for bottom level
    # CASE 4 - regularizer + validation loss for whole hierarchy

    if l1_regularizer:
        if validate_hf_loss:
            case = 'case4'
        else:
            case = 'case3'
    else:
        if validate_hf_loss:
            case = 'case2'
        else:
            case = 'case1'

    if len(seed_runs) == 1:
        name_file = f'{case}_one_seed'
    else:
        name_file = case

    if tune_lambda == False:
        lambda_case = 1
    else:
        lambda_case = lambda_range[lambda_range_idx]
    name_file = f'{name_file}_lambda_{lambda_case}'
    print(lambda_case)
    # print(name_file)

    if remove_skip:
        name_file = f'{name_file}_lambda_{lambda_case}_no_skip'
    print(name_file)

    # # full dataset
    # # run_ml_reconciliation(data, file_name, name_file, lambda_case, seed_value, seed_runs, number_of_levels, tune_lambda)
    # # samples = dataset_samples[data]
    #
    # # samples
    # # for sample in range(0, samples):
    # #     data_path = f'new_data_samples/{data}_{sample}'
    # #     model_file_path = f'{data}_{sample}_{model}'
    # #     run_ml_reconciliation(data_path, model_file_path, name_file, lambda_case, seed_value, seed_runs,
    # #                           number_of_levels, tune_lambda)
    sample = int(sys.argv[7])
    print(sample)

    data_path = f'new_data_samples/{data}_{sample}'
    model_file_path = f'{data}_{sample}_{model}'
    run_ml_reconciliation(data_path, model_file_path, name_file, lambda_case, seed_value, seed_runs,
                          number_of_levels, tune_lambda, remove_skip)
