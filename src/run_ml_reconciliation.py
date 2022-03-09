import pandas as pd
from ml_reconcilation import MLReconcile
import sys

if __name__ == '__main__':
    DATASET = ['prison', 'tourism', 'wikipedia', 'labour']
    levels_in_hierarchy = {'prison': 5, 'tourism': 3, 'wikipedia': 5, 'labour': 4}
    data = DATASET[int(sys.argv[1])]
    model = sys.argv[2]
    number_of_levels = levels_in_hierarchy[data]
    seed_value = 1234
    # seed_runs = [1234]
    seed_runs = [1234, 3456, 2311, 8311, 5677]
    file_name = f'{data}_{model}'
    tune_lambda = bool(sys.argv[3])
    validate_hf_loss = bool(sys.argv[4])
    lambda_range_idx = int(sys.argv[5])
    lambda_range = [[0.0001, 0.1], [1.1, 5]]
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
        name_file = f'{name_file}_lambda1'
        lambda_case = 1
    else:
        lambda_case = lambda_range[lambda_range_idx]
        if lambda_range_idx >0:
            name_file = f'{name_file}_lambda2'
    print(lambda_case)
    print(name_file)

    df_actual = pd.read_csv(f"input_data/{data}_actual.csv")
    df_fitted = pd.read_csv(f"forecasts/{file_name}_fitted.csv")
    df_forecasts = pd.read_csv(f"forecasts/{file_name}_forecasts.csv")

    hyper_params = {'number_of_layers': 5, 'epochs': [10, 200], 'dropout_rate': [0, 0.5], 'max_norm_value': [0, 10],
                    'reconciliation_loss_lambda': lambda_case, 'learning_rate': [0.0001, 0.1]}

    # if hyper parameter tuning is not required tune_hyper_params = False and
    # best_hyper_params parameter should be passed

    ml_model_case1 = MLReconcile(seed_value, df_actual, df_fitted, df_forecasts, number_of_levels, seed_runs,
                                 hyper_params_tune=hyper_params,
                                 tune_hyper_params=True, tune_lambda=tune_lambda)
    forecasts_adjusted_case1, forecasts_adjusted_case1_mean, model_history_case1, best_hyper_params_case1 = ml_model_case1.run_ml_reconciliation()

    forecasts_adjusted_case1.to_csv(f'results/{file_name}_adjusted_forecasts_{name_file}.csv')
    if len(seed_runs) > 1:
        forecasts_adjusted_case1_mean.to_csv(f'results/{file_name}_adjusted_forecasts_{name_file}_mean.csv')
        model_history_case1.to_csv(f'results/model_history/{file_name}_model_history_{name_file}.csv')
        best_hyper_params_case1.to_csv(f'results/model_history/{file_name}_best_params_{name_file}.csv')
