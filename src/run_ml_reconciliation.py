import pandas as pd
from src.ml_reconcilation import MLReconcile

if __name__ == '__main__':
    levels_in_hierarchy = {'prison': 5, 'tourism': 3}
    data = 'prison'
    number_of_levels = levels_in_hierarchy[data]
    model = 'ets'
    seed_value = 1234
    seed_runs = [1234]
    # seed_runs = [1234, 3456, 2311, 8311, 5677]
    file_name = f'{data}_{model}'
    tune_hyper_params = True
    validate_hf_loss = False
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

    df_actual = pd.read_csv(f"input_data/{data}_actual.csv")
    df_fitted = pd.read_csv(f"forecasts/{file_name}_fitted.csv")
    df_forecasts = pd.read_csv(f"forecasts/{file_name}_forecasts.csv")

    hyper_params = {'number_of_layers': 5, 'epochs': [10, 200], 'dropout_rate': [0, 0.5], 'max_norm_value': [0, 10],
                    'reconciliation_loss_lambda': [0.1, 0.9], 'learning_rate': [0.0001, 0.1]}

    # if hyper parameter tuning is not required tune_hyper_params = False and
    # best_hyper_params parameter should be passed

    # CASE 1 - validation loss for bottom level
    ml_model_case1 = MLReconcile(seed_value, df_actual, df_fitted, df_forecasts, number_of_levels, seed_runs,
                                 hyper_params_tune=hyper_params,
                                 tune_hyper_params=True)
    forecasts_adjusted_case1, forecasts_adjusted_case1_mean, model_history_case1, best_hyper_params_case1 = ml_model_case1.run_ml_reconciliation()

    forecasts_adjusted_case1.to_csv(f'results/{file_name}_adjusted_forecasts_{name_file}.csv')
    # forecasts_adjusted_case1_mean.to_csv(f'results/{file_name}_adjusted_forecasts_{name_file}_mean.csv')
    # model_history_case1.to_csv(f'results/{file_name}_model_history_{name_file}.csv')
    # best_hyper_params_case1.to_csv(f'results/{file_name}_best_params_{name_file}.csv')
