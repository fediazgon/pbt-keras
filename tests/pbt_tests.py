import unittest

import keras
import numpy as np
import tensorflow as tf
from keras import backend as K

from pbt import Member

TRAIN_DATASET_SIZE = 64
config = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)


class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))


class TestsPbtMethods(unittest.TestCase):

    def reset_session(self):
        K.clear_session()
        self.session = tf.Session(config=config, graph=tf.get_default_graph())
        K.set_session(self.session)
        np.random.seed(42)
        tf.set_random_seed(42)

    @classmethod
    def setUpClass(cls):
        mnist = tf.keras.datasets.mnist
        (x_train, y_train), (_, _) = mnist.load_data()
        x_train = x_train[:TRAIN_DATASET_SIZE, :, :]
        cls.y_train = y_train[:TRAIN_DATASET_SIZE]
        cls.x_train = x_train / 255.0

    def setUp(self):
        self.reset_session()

    def tearDown(self):
        self.session.close()

    def test_same_member_configuration(self):
        member_a, history_a = Member(), LossHistory()
        member_a.model.fit(self.x_train, self.y_train, verbose=0, callbacks=[history_a])
        member_a_reg_values = member_a.regularizer.get_l1_l2()
        self.reset_session()
        member_b, history_b = Member(), LossHistory()
        member_b.model.fit(self.x_train, self.y_train, verbose=0, callbacks=[history_b])
        member_b_reg_values = member_b.regularizer.get_l1_l2()
        self.assertSequenceEqual(member_a_reg_values, member_b_reg_values)
        self.assertSequenceEqual(history_a.losses, history_b.losses)

    def test_explore_regularization_one_member(self):
        # Train member A without changing the regularization
        member_a, history_a = Member(), LossHistory()
        member_a.model.fit(self.x_train, self.y_train, verbose=0)
        member_a.model.fit(self.x_train, self.y_train, verbose=0, callbacks=[history_a])
        member_a_reg_values = member_a.regularizer.get_l1_l2()
        self.reset_session()
        # Train member B changing the regularization after the first epoch
        member_b, history_b = Member(), LossHistory()
        member_b.model.fit(self.x_train, self.y_train, verbose=0)
        member_b.regularizer.explore()
        member_b.model.fit(self.x_train, self.y_train, verbose=0, callbacks=[history_b])
        member_b_reg_values = member_b.regularizer.get_l1_l2()
        self.assertRaises(AssertionError, self.assertSequenceEqual, member_a_reg_values, member_b_reg_values)
        self.assertRaises(AssertionError, self.assertSequenceEqual, history_a.losses, history_b.losses)
