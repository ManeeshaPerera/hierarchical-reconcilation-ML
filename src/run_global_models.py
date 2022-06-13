import pandas as pd
from deepAR import DeepAR
from waveNet import WaveNet
import sys

dataset = ['prison', 'tourism', 'labour', 'wikipedia']
DATA = {'prison': {'freq': 'Q', 'horizon': 8, 'samples': 3, 'start_date': '2005'},
        'tourism': {'freq': 'M', 'horizon': 12, 'samples': 10, 'start_date': '1998 Jan'},
        'labour': {'freq': 'Q', 'horizon': 12, 'samples': 5, 'start_date': '1987 Feb'},
        'wikipedia': {'freq': 'W', 'horizon': 7, 'samples': 10, 'start_date': '2016-06-01'}}
levels_in_hierarchy = {'prison': 5, 'tourism': 3, 'wikipedia': 6, 'labour': 4}
models = [DeepAR, WaveNet]
model_names = ['deepAR', 'waveNet']

data = dataset[int(sys.argv[1])]
model = models[int(sys.argv[2])]
model_name = model_names[int(sys.argv[2])]
clusters = bool(sys.argv[3])

freq = DATA[data]['freq']
horizon = DATA[data]['horizon']
start_date = DATA[data]['start_date']
samples = DATA[data]['samples']
levels = levels_in_hierarchy[data]

for sample in range(0, samples):
    sample_train = pd.read_csv(f'input_data/new_data_samples/{data}_{sample}_actual.csv')
    sample_test = pd.read_csv(f'input_data/new_data_samples/{data}_{sample}_test.csv')

    if clusters:
        cluster_df = pd.read_csv(f'input_data/ts_features/{data}_clusters.csv')
        level_wise_fitted = []
        level_wise_fc = []

        for cluster_val in range(0, 20):
            deepAR_model = model(None, sample_train, sample_test, start_date, freq, horizon, cluster=True,
                                 cluster_num=cluster_val,
                                 cluster_df=cluster_df)
            fitted_values_level, forecast_level = deepAR_model.run_model()
            level_wise_fitted.append(fitted_values_level)
            level_wise_fc.append(forecast_level)

        level_wise_fitted_df = pd.concat(level_wise_fitted)
        level_wise_fc_df = pd.concat(level_wise_fc)

        level_wise_fc_df['Description'] = pd.Categorical(
            level_wise_fc_df['Description'],
            categories=sample_train['Description'].values.tolist(),
            ordered=True
        )

        level_wise_fitted_df['Description'] = pd.Categorical(
            level_wise_fitted_df['Description'],
            categories=sample_train['Description'].values.tolist(),
            ordered=True
        )

        level_wise_fitted_df.sort_values('Description', inplace=True)
        level_wise_fc_df.sort_values('Description', inplace=True)

        level_wise_fitted_df.to_csv(f'forecasts/new_data_samples/{data}_{sample}_{model_name}_cluster_fitted.csv',
                                    index=False)
        level_wise_fc_df.to_csv(f'forecasts/new_data_samples/{data}_{sample}_{model_name}_cluster_forecasts.csv',
                                index=False)

    else:
        level_wise_fitted = []
        level_wise_fc = []

        for level in range(1, levels + 1):
            deepAR_model = model(level, sample_train, sample_test, start_date, freq, horizon)
            fitted_values_level, forecast_level = deepAR_model.run_model()
            level_wise_fitted.append(fitted_values_level)
            level_wise_fc.append(forecast_level)

        level_wise_fitted_df = pd.concat(level_wise_fitted)
        level_wise_fc_df = pd.concat(level_wise_fc)

        level_wise_fitted_df.to_csv(f'forecasts/new_data_samples/{data}_{sample}_{model_name}_fitted.csv', index=False)
        level_wise_fc_df.to_csv(f'forecasts/new_data_samples/{data}_{sample}_{model_name}_forecasts.csv', index=False)
