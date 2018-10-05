import unittest

import numpy as np
import tensorflow as tf
from keras import backend as K

from pbt.population import Member, BatchGenerator

DATASET = tf.keras.datasets.boston_housing
TRAIN_DATASET_SIZE = 404  # Use all
TEST_DATASET_SIZE = 102  # Use all
BATCH_SIZE = 64

config = tf.ConfigProto(intra_op_parallelism_threads=1,
                        inter_op_parallelism_threads=1)


class TestsPbtMethods(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        (x_train, y_train), (x_test, y_test) = DATASET.load_data()
        cls.x_train = x_train[:TRAIN_DATASET_SIZE]
        cls.y_train = y_train[:TRAIN_DATASET_SIZE]
        cls.x_test = x_test[:TEST_DATASET_SIZE]
        cls.y_test = y_test[:TEST_DATASET_SIZE]

    def start_nonrandom_session(self):
        np.random.seed(42)
        tf.set_random_seed(42)
        self.session = tf.Session(config=config, graph=tf.get_default_graph())
        K.set_session(self.session)

    def close_current_session(self):
        K.clear_session()
        self.session.close()

    @staticmethod
    def assert_list_arrays_equal(list_a, list_b):
        for i in range(len(list_a)):
            np.testing.assert_array_equal(list_a[i], list_b[i])

    def setUp(self):
        self.start_nonrandom_session()

    def tearDown(self):
        self.close_current_session()

    def create_batch_generator(self):
        return BatchGenerator(self.x_train, self.y_train,
                              self.x_test, self.y_test, batch_size=BATCH_SIZE)

    def test_step(self):
        """Training two members of the population with the same parameters."""
        ma = Member(self.create_batch_generator())
        loss_a = ma.step()
        ma_reg_config = ma.regularizer.get_config()
        self.close_current_session()
        self.start_nonrandom_session()
        mb = Member(self.create_batch_generator())
        loss_b = mb.step()
        mb_reg_config = mb.regularizer.get_config()
        self.assertDictEqual(ma_reg_config, mb_reg_config)
        self.assertEqual(loss_a, loss_b)

    def test_explore(self):
        """Training two members of the population. Calling 'explore' in one."""
        ma = Member(self.create_batch_generator())
        ma.step()
        loss_a = ma.step()
        ma_reg_config = ma.regularizer.get_config()
        self.close_current_session()
        self.start_nonrandom_session()
        mb = Member(self.create_batch_generator())
        mb.explore()
        loss_b = mb.step()
        mb_reg_config = mb.regularizer.get_config()
        self.assertRaises(AssertionError, self.assertDictEqual,
                          ma_reg_config, mb_reg_config)
        self.assertNotEqual(loss_a, loss_b)

    def test_replace(self):
        """Replacing the hyperparameters and weights of one model."""
        ma = Member(self.create_batch_generator())
        mb = Member(self.create_batch_generator())
        loss_a = ma.step()
        loss_b = mb.step()
        self.assertRaises(AssertionError, np.testing.assert_array_equal,
                          ma.model.get_weights(),
                          mb.model.get_weights())
        self.assertNotEqual(loss_a, loss_b)
        ma.replace_with(mb)
        self.assert_list_arrays_equal(ma.model.get_weights(),
                                      mb.model.get_weights())
        self.assertDictEqual(ma.regularizer.get_config(),
                             mb.regularizer.get_config())

    def test_eval(self):
        """Training one model for a few steps to evaluate the performance."""
        ma = Member(self.create_batch_generator())
        loss_1 = ma.eval()
        for i in range(5):
            ma.step()
        loss_n = ma.eval()
        self.assertLess(loss_n, loss_1)
