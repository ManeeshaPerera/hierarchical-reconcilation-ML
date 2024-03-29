from gluonts.model.deepar import DeepAREstimator
from gluonts.mx.trainer import Trainer
from dnn_class import DNN


class DeepAR(DNN):
    def __init__(self, level, data_train, data_test, start_date, freq, horizon, cluster=False, cluster_num=None,
                 cluster_df=None):
        super().__init__(level, data_train, data_test, start_date, freq, horizon, cluster, cluster_num,
                         cluster_df)

    def train_model(self):
        estimator = DeepAREstimator(freq=self.freq, prediction_length=self.horizon, trainer=Trainer(epochs=50))
        self.predictor = estimator.train(training_data=self.train_dataset)

    def run_model(self):
        self.train_model()
        forecasts_test = self.get_forecasts()
        self.create_samples_fitted()
        fitted_values = self.get_fitted_values()
        return fitted_values, forecasts_test
