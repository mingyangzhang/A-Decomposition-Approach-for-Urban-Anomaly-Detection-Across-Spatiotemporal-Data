import scipy.io as sio
import tensorflow as tf
import os
import numpy as np

from data_helper import DataHelper
from model import AnoModel
from sklearn.neighbors import LocalOutlierFactor
from utils import *

BATCH_SIZE = 128
EPOCH = 100

def model_train(model):
    """ """

    data = sio.loadmat("./data/fake_data.mat")
    dh = DataHelper(data, 19*7*24, 1*7*24, 50)

    model.construct_loss()
    optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(model.loss)
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(init)
        ckpt = tf.train.get_checkpoint_state(os.path.dirname("checkpoints/checkpoint"))
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)

        train_feature_batches, train_flow_batches = dh.gen_train_batch(BATCH_SIZE)
        for i in range(EPOCH):

            for x0, y0 in zip(train_feature_batches, train_flow_batches):
                shuffled_index = np.arange(BATCH_SIZE)
                np.random.shuffle(shuffled_index)
                x1 = x0[shuffled_index]
                y1 = y0[shuffled_index]

                feed_dict = {
                    model.x0: x0,
                    model.y0: y0,
                    model.x1: x1,
                    model.y1: y1,
                }
                _, loss = sess.run([optimizer, model.loss], feed_dict=feed_dict)

            print("Epoch {}: loss {}".format(i, loss))
            saver.save(sess, "checkpoints/model")

def model_test(model):
    """ """
    data = sio.loadmat("./data/fake_data.mat")
    dh = DataHelper(data, 19*7*24, 1*7*24, 50)

    feature, flow, label = dh.gen_test_samples()
    tfeature = feature[:, :model._tf_dim]
    sfeature = feature[:, -model._sf_dim:]

    err = model.decompose()
    saver = tf.train.Saver()
    with tf.Session() as sess:
        ckpt = tf.train.get_checkpoint_state(os.path.dirname("checkpoints/checkpoint"))
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            return
        feed_dict = {
            model._sfeature: sfeature,
            model._tfeature: tfeature,
            model._flow: flow
        }
        err, stfeature = sess.run([err, model.stfeature], feed_dict=feed_dict)

    ano_detect(flow, err, stfeature, label)

def ano_detect(flow, err, stfeature, label):
    """ """

    # FAL
    points =  np.concatenate([stfeature, err], axis=1)
    detector = LocalOutlierFactor(n_neighbors=100)
    detector.fit(points)
    ano_scores = - detector._decision_function(points)
    compute_metrics(ano_scores, label, "FAL")

    # LOF
    points = flow
    detector = LocalOutlierFactor(n_neighbors=100)
    detector.fit(points)
    ano_scores = - detector._decision_function(points)
    compute_metrics(ano_scores, label, "LOF")