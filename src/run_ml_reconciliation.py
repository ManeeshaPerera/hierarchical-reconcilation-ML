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

    # ml_model = MLReconcile(seed_value, df_actual, df_fitted, df_forecasts, number_of_levels, seed_runs,
    #                        hyper_params_tune=hyper_params,
    #                        tune_hyper_params=True)
    # ml_model.run_ml_reconciliation()
    # forecasts_adjusted, model_history, best_hyper_params = ml_model.run_ml_reconciliation()
    #
    # forecasts_adjusted.to_csv(f'results/{file_name}_adjusted_forecasts.csv')
    # model_history.to_csv(f'results/{file_name}_model_history.csv')
    # best_hyper_params.to_csv(f'results/{file_name}_best_params.csv')

    ml_model_validation_loss_hf = MLReconcile(seed_value, df_actual, df_fitted, df_forecasts, number_of_levels,
                                              seed_runs,
                                              hyper_params_tune=hyper_params,
                                              tune_hyper_params=True, validate_hf_loss=True)
    ml_model_validation_loss_hf.run_ml_reconciliation()
    forecasts_adjusted_case1, model_history_case1, best_hyper_params_case1 = ml_model_validation_loss_hf.run_ml_reconciliation()

    forecasts_adjusted_case1.to_csv(f'results/{file_name}_adjusted_forecasts_hf_val_loss.csv')
    model_history_case1.to_csv(f'results/{file_name}_model_history_hf_val_loss.csv')
    best_hyper_params_case1.to_csv(f'results/{file_name}_best_params_hf_val_loss.csv')
