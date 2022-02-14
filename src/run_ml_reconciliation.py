import pandas as pd
from src.ml_reconcilation import MLReconcile

if __name__ == '__main__':
    levels_in_hierarchy = {'prison': 5, 'tourism': 3}
    data = 'tourism'
    number_of_levels = levels_in_hierarchy[data]
    model = 'ets'
    seed_value = 1234
    seed_runs = [1234, 3456, 2311, 8311, 5677]
    file_name = f'{data}_{model}'

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
    ml_model_case1.run_ml_reconciliation()
    forecasts_adjusted_case1, model_history_case1, best_hyper_params_case1 = ml_model_case1.run_ml_reconciliation()

    forecasts_adjusted_case1.to_csv(f'results/{file_name}_adjusted_forecasts_case1.csv')
    model_history_case1.to_csv(f'results/{file_name}_model_history_case1.csv')
    best_hyper_params_case1.to_csv(f'results/{file_name}_best_params_case1.csv')

    # CASE 2 - validation loss for complete hierarchy

    ml_model_case2 = MLReconcile(seed_value, df_actual, df_fitted, df_forecasts, number_of_levels,
                                              seed_runs,
                                              hyper_params_tune=hyper_params,
                                              tune_hyper_params=True, validate_hf_loss=True)
    ml_model_case2.run_ml_reconciliation()
    forecasts_adjusted_case2, model_history_case2, best_hyper_params_case2 = ml_model_case2.run_ml_reconciliation()

    forecasts_adjusted_case2.to_csv(f'results/{file_name}_adjusted_forecasts_case2.csv')
    model_history_case2.to_csv(f'results/{file_name}_model_history_case2.csv')
    best_hyper_params_case2.to_csv(f'results/{file_name}_best_params_case2.csv')

    # CASE 3 - regularizer + validation loss for bottom level

    ml_model_case3 = MLReconcile(seed_value, df_actual, df_fitted, df_forecasts, number_of_levels, seed_runs,
                           hyper_params_tune=hyper_params,
                           tune_hyper_params=True, l1_regularizer=True)
    ml_model_case3.run_ml_reconciliation()
    forecasts_adjusted_case3, model_history_case3, best_hyper_params_case3 = ml_model_case3.run_ml_reconciliation()

    forecasts_adjusted_case3.to_csv(f'results/{file_name}_adjusted_forecasts_case3.csv')
    model_history_case3.to_csv(f'results/{file_name}_model_history_case3.csv')
    best_hyper_params_case3.to_csv(f'results/{file_name}_best_params_case3.csv')

    # CASE 4 - regularizer + validation loss for whole hierarchy
    ml_model_case4 = MLReconcile(seed_value, df_actual, df_fitted, df_forecasts, number_of_levels, seed_runs,
                                 hyper_params_tune=hyper_params,
                                 tune_hyper_params=True, l1_regularizer=True, validate_hf_loss=True)
    ml_model_case4.run_ml_reconciliation()
    forecasts_adjusted_case4, model_history_case4, best_hyper_params_case4 = ml_model_case4.run_ml_reconciliation()

    forecasts_adjusted_case4.to_csv(f'results/{file_name}_adjusted_forecasts_case4.csv')
    model_history_case4.to_csv(f'results/{file_name}_model_history_case4.csv')
    best_hyper_params_case4.to_csv(f'results/{file_name}_best_params_case4.csv')
