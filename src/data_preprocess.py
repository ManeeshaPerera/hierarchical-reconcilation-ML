import pandas as pd


def create_window_samples(idx=0, expanding=False):
    datasets = ['prison', 'tourism', 'wikipedia', 'labour']
    dataset_samples = [3, 10, 10, 5]
    horizon = [8, 12, 7, 12]

    data_sample_idx = idx
    dataset_name = datasets[data_sample_idx]
    ts_data = pd.read_csv(f'input_data/{dataset_name}.csv')
    data = ts_data.iloc[:, 2:]
    meta_data = ts_data.iloc[:, 0:2]
    print(meta_data)

    print(len(data.columns))
    actual_one_sample = pd.read_csv(f'input_data/{dataset_name}_actual.csv').iloc[:, 2:]
    print(len(actual_one_sample.columns))

    number_of_samples = dataset_samples[data_sample_idx]
    horizon_data = horizon[data_sample_idx]

    size = number_of_samples * horizon_data

    data_start = data.iloc[:, :-size]
    test_samples = data.iloc[:, -size:]
    print("Sample Training size:", len(data_start.columns))

    for i in range(0, number_of_samples):
        train = data_start
        test = test_samples.iloc[:, 0:horizon_data]
        print("train ", train)
        print("test", test)

        train_save = pd.concat([meta_data, train], axis=1)
        test_save = pd.concat([meta_data, test], axis=1)

        if expanding:
            dir_name = 'new_data_samples'
        else:
            dir_name = 'data_samples'

        train_save.to_csv(f'input_data/{dir_name}/{dataset_name}_{i}_actual.csv', index=False)
        test_save.to_csv(f'input_data/{dir_name}/{dataset_name}_{i}_test.csv', index=False)

        if not expanding:
            data_start = data.iloc[:, (i + 1) * horizon_data:(-size + horizon_data * (i + 1))]
            test_samples = test_samples.iloc[:, horizon_data:]
        else:
            data_start = data.iloc[:, :(-size + horizon_data * (i + 1))]
            test_samples = test_samples.iloc[:, horizon_data:]


if __name__ == '__main__':
    create_window_samples(expanding=True)
    create_window_samples(idx=1, expanding=True)
    create_window_samples(idx=2, expanding=True)
    create_window_samples(idx=3, expanding=True)
