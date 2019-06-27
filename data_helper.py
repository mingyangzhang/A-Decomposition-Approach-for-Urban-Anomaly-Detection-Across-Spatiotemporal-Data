import numpy as np
import scipy.io as sio

class DataHelper(object):
    """ Load data for model train and test. """

    def __init__(self, data, train_len, test_len, ano_num):
        """
        """

        feature = data["x"]
        flow = data["y"]
        label = data["label"]

        self._num_region = flow.shape[1]
        self._num_dynamics = flow.shape[2]

        train_feature = feature[:train_len, :, :]

        self._train_feature = train_feature.reshape(-1, train_feature.shape[-1])

        train_flow = flow[:train_len, :, :]
        self._train_flow = train_flow.reshape(-1, train_flow.shape[-1])

        test_label = label[-test_len:, :]
        self._label = test_label.reshape(-1, test_label.shape[-1])

        test_feature = feature[-test_len:, :, :]
        self._test_feature = test_feature.reshape(-1, test_feature.shape[-1])

        test_flow = flow[-test_len:, :, :]
        self._test_flow = test_flow.reshape(-1, test_flow.shape[-1])

    def gen_train_batch(self, batch_size):

        train_feature_batches = []
        train_flow_batches = []

        batch_num = self._train_feature.shape[0] // batch_size
        for i in range(batch_num):
            train_feature_batches.append(self._train_feature[i*batch_size:(i+1)*batch_size, :])
            train_flow_batches.append(self._train_flow[i*batch_size:(i+1)*batch_size, :])

        return train_feature_batches, train_flow_batches

    def gen_test_samples(self):

        return self._test_feature, self._test_flow, self._label.flatten()

    def gen_baseline_data(self):

        return self._train_feature, self._train_flow, self._test_feature, self._test_flow, self._label.flatten()

    def get_size(self):

        return self._num_dynamics, self._num_region