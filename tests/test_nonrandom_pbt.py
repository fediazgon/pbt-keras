import unittest

import numpy as np
import tensorflow as tf
from keras import backend as K

from pbt.population import Member, BatchGenerator

TRAIN_DATASET_SIZE = 64
TEST_DATASET_SIZE = 16
BATCH_SIZE = 8

config = tf.ConfigProto(intra_op_parallelism_threads=1,
                        inter_op_parallelism_threads=1)


class TestsPbtMethods(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        mnist = tf.keras.datasets.mnist
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        cls.x_train = x_train[:TRAIN_DATASET_SIZE, :, :]
        cls.y_train = y_train[:TRAIN_DATASET_SIZE]
        cls.x_test = x_test[:TEST_DATASET_SIZE, :, :]
        cls.y_test = y_test[:TEST_DATASET_SIZE]

    def start_nonrandom_session(self):
        np.random.seed(42)
        tf.set_random_seed(42)
        self.session = tf.Session(config=config, graph=tf.get_default_graph())
        K.set_session(self.session)

    def close_current_session(self):
        K.clear_session()
        self.session.close()

    def setUp(self):
        self.start_nonrandom_session()

    def tearDown(self):
        self.close_current_session()

    def create_batch_generator(self):
        return BatchGenerator(self.x_train, self.y_train,
                              self.x_test, self.y_test, batch_size=BATCH_SIZE)

    def test_step(self):
        """Training two members of the population with the same parameters.
         Losses should be equal."""
        member_a = Member(self.create_batch_generator())
        loss_a, accuracy_a = member_a.step()
        member_a_reg_values = member_a.regularizer.get_config()
        self.close_current_session()
        self.start_nonrandom_session()
        member_b = Member(self.create_batch_generator())
        loss_b, accuracy_b = member_b.step()
        member_b_reg_values = member_b.regularizer.get_config()
        self.assertDictEqual(member_a_reg_values, member_b_reg_values)
        self.assertSequenceEqual([loss_a, accuracy_a], [loss_b, accuracy_b])

    def test_explore(self):
        """Training two members of the population. Calling 'explore' in one of
         them. Losses should be different."""
        member_a = Member(self.create_batch_generator())
        member_a.step()
        loss_a, accuracy_a = member_a.step()
        member_a_reg_values = member_a.regularizer.get_config()
        self.close_current_session()
        self.start_nonrandom_session()
        member_b = Member(self.create_batch_generator())
        member_b.explore()
        loss_b, accuracy_b = member_b.step()
        member_b_reg_values = member_b.regularizer.get_config()
        self.assertRaises(AssertionError, self.assertDictEqual,
                          member_a_reg_values, member_b_reg_values)
        self.assertRaises(AssertionError, self.assertSequenceEqual,
                          [loss_a, accuracy_a], [loss_b, accuracy_b])

    def test_replace(self):
        """Replace the hyperparameters and weights of one model with
        hyperparameters and weights of other model."""
        member_a = Member(self.create_batch_generator())
        member_b = Member(self.create_batch_generator())
        loss_a, accuracy_a = member_a.step()
        loss_b, accuracy_b = member_b.step()
        self.assertRaises(AssertionError, np.testing.assert_array_equal,
                          member_a.model.get_weights(),
                          member_b.model.get_weights())
        self.assertRaises(AssertionError, self.assertSequenceEqual,
                          [loss_a, accuracy_a],
                          [loss_b, accuracy_b])
        member_a.replace_with(member_b)
        np.testing.assert_array_equal(member_a.model.get_weights(),
                                      member_b.model.get_weights())
        self.assertDictEqual(member_a.regularizer.get_config(),
                             member_b.regularizer.get_config())
