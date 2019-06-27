import os
import tensorflow as tf
import pandas as pd
import numpy as np
import scipy.io as sio

class AnoModel(object):

    def __init__(self, tf_units, sf_units, st_units, out_dim, sf_dim, tf_dim, batch_size):
        self._tf_units = tf_units
        self._sf_units = sf_units
        self._st_units = st_units
        self._out_dim = out_dim

        self._sf_dim = sf_dim
        self._tf_dim = tf_dim
        self._batch_size = batch_size
        self._varibles = []

    def forward(self, tfeature, sfeature):

        tfeature = tf.cast(tfeature, tf.float32)
        sfeature = tf.cast(sfeature, tf.float32)

        self._varibles.append(tfeature)
        self._varibles.append(sfeature)

        with tf.variable_scope("dense", reuse=tf.AUTO_REUSE) as scp:
            for i,lsize in enumerate(self._tf_units):
                tfeature = tf.layers.dense(inputs=tfeature, units=lsize, activation=tf.nn.relu, name="tf_dense_{}".format(i))

            for i,lsize in enumerate(self._sf_units):
                sfeature = tf.layers.dense(inputs=sfeature, units=lsize, activation=tf.nn.relu, name="sf_dense_{}".format(i))

            x = tf.concat([tfeature, sfeature], axis=1)
            st_feature = x
            self._varibles.append(st_feature)

            for i,lsize in enumerate(self._st_units):
                x = tf.layers.dense(inputs=x, units=lsize, activation=tf.nn.relu, name="st_dense_{}".format(i))

        st_dim = self._st_units[-1]
        with tf.variable_scope("output", reuse=tf.AUTO_REUSE) as scp:
            W = tf.get_variable(name="output_weight",
                                initializer=tf.truncated_normal((st_dim, self._out_dim), dtype=tf.float32, stddev=1e-1))

            b = tf.get_variable(name="output_bias",
                                initializer=tf.truncated_normal((1,), dtype=tf.float32, stddev=1e-1))

        y = tf.add(tf.matmul(x, W), b)
        return y, st_feature

    def construct_loss(self):

        st_dim = self._tf_dim + self._sf_dim
        self.x0 = tf.placeholder(tf.float32, shape=(self._batch_size, st_dim))
        tfeature0 = self.x0[:, :self._tf_dim]
        sfeature0 = self.x0[:, -self._sf_dim:]
        y_pred0, stfeature0 = self.forward(tfeature0, sfeature0)

        self.x1 = tf.placeholder(tf.float32, shape=(self._batch_size, st_dim))
        tfeature1 = self.x1[:, :self._tf_dim]
        sfeature1 = self.x1[:, -self._sf_dim:]
        y_pred1, stfeature1 = self.forward(tfeature1, sfeature1)

        self.y0 = tf.placeholder(tf.float32, shape=(self._batch_size, self._out_dim))
        self.y1 = tf.placeholder(tf.float32, shape=(self._batch_size, self._out_dim))

        self.pred_loss = tf.losses.mean_squared_error(self.y0, y_pred0) + tf.losses.mean_squared_error(self.y1, y_pred1)

        output_dis = tf.losses.mean_squared_error(y_pred0, y_pred1)
        input_dis = tf.losses.mean_squared_error(stfeature0, stfeature1)

        _lambda = 0.001
        self.simi_loss = _lambda*output_dis/(input_dis+1)
        self.loss = self.pred_loss + self.simi_loss

    def decompose(self):

        self._sfeature = tf.placeholder(tf.float32, shape=(None, self._sf_dim))
        self._tfeature = tf.placeholder(tf.float32, shape=(None, self._tf_dim))
        self._flow = tf.placeholder(tf.float32, shape=(None, self._out_dim))

        y_pred, self.stfeature  = self.forward(self._tfeature, self._sfeature)
        err = y_pred - self._flow
        return err


if __name__ == "__main__":
    pass
