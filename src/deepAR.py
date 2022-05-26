import pandas as pd
from gluonts.dataset.common import ListDataset
from gluonts.model.deepar import DeepAREstimator
from gluonts.mx.trainer import Trainer
from gluonts.evaluation import make_evaluation_predictions


class DeepAR:
    def __init__(self, level, data_train, data_test, start_date, freq, horizon):
        self.level = level
        self.data_train = data_train.loc[data_train['Level'] == level]
        original_test = data_test.loc[data_test['Level'] == level]
        self.data_test = pd.concat([self.data_train, original_test.iloc[:, 2:]], axis=1)
        self.train_dataset = None
        self.test_dataset = None
        self.train_list = []
        self.test_list = []
        self.fitted_samples = []
        self.start_date = start_date
        self.freq = freq
        self.horizon = horizon
        self.predictor = None
        self.meta_data = self.data_train['Description'].values.tolist()
        self.process_data_model(self.data_train, self.train_list)
        self.process_data_model(self.data_test, self.test_list)
        self.create_listDataset()

    def process_data_model(self, dataframe, list_):
        for idx, ts_ in dataframe.iterrows():
            df_ = pd.DataFrame(ts_).iloc[2:, ].reset_index()
            df_.columns = ['timestamp', 'value']
            list_.append(df_)

    def create_listDataset(self):
        train_data_ls = []
        test_data_ls = []
        for time_series in self.train_list:
            train_data_ls.append({'target': time_series['value'].values, 'start': self.start_date})

        for time_series in self.test_list:
            test_data_ls.append({'target': time_series['value'].values, 'start': self.start_date})

        self.train_dataset = ListDataset(train_data_ls, freq=self.freq)
        self.test_dataset = ListDataset(test_data_ls, freq=self.freq)

    def train_model(self):
        estimator = DeepAREstimator(freq=self.freq, prediction_length=self.horizon, trainer=Trainer(epochs=50))
        self.predictor = estimator.train(training_data=self.train_dataset)

    def add_meta_data(self, list_convert):
        df_fc = pd.DataFrame(list_convert)
        columns = ["Level", "Description"]
        last_col_value = df_fc.columns[-1]
        for h in range(1, last_col_value):
            columns.append(str(h))
        df_fc.columns = columns
        return df_fc

    def create_dataframe_fc(self, forecast_list):
        fc_data = []
        for i in range(0, len(forecast_list)):
            fc_values = forecast_list[i].median.tolist()
            fc_values.insert(0, self.meta_data[i])
            fc_values.insert(0, self.level)
            fc_data.append(fc_values)
        return self.add_meta_data(fc_data)

    def get_forecasts(self):
        forecast_it, ts_it = make_evaluation_predictions(
            dataset=self.test_dataset,
            predictor=self.predictor)
        forecasts = list(forecast_it)
        return self.create_dataframe_fc(forecasts)

    def create_samples_fitted(self):
        all_samples = []
        sample_count = 0
        for sample_len in range(len(self.train_list[0]), 0, -self.horizon):
            all_samples.append([])
            for ts in self.train_list:
                all_samples[sample_count].append(ts[0: sample_len])
            sample_count += 1
        self.fitted_samples = all_samples

    def predict_sample(self, ts_array):
        input_samples = []
        for time_series in ts_array:
            input_samples.append({'target': time_series['value'].values, 'start': self.start_date})
        input_dataset = ListDataset(input_samples, freq=self.freq)
        fitted, ts_it = make_evaluation_predictions(
            dataset=input_dataset,
            predictor=self.predictor,
        )
        return list(fitted)

    def get_fitted_values(self):
        # self.train_list length will have the length of all time series
        fitted_data = []

        for ts in range(len(self.train_list)):
            fitted_data.append([])

        for sample in range(len(self.fitted_samples) - 1, -1, -1):
            if len(self.fitted_samples[sample][0]) > self.horizon:
                all_sample_fc = self.predict_sample(self.fitted_samples[sample])
                for i in range(len(all_sample_fc)):
                    fc_values = all_sample_fc[i].median.tolist()
                    fitted_data[i].extend(fc_values)
        for ts in range(len(fitted_data)):
            fitted_data[ts].insert(0, self.meta_data[ts])
            fitted_data[ts].insert(0, self.level)
        return self.add_meta_data(fitted_data)

    def run_model(self):
        self.train_model()
        forecasts_test = self.get_forecasts()
        self.create_samples_fitted()
        fitted_values = self.get_fitted_values()
        return fitted_values, forecasts_test
