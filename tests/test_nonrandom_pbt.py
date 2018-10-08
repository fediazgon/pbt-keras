import unittest

import keras
import numpy as np
import tensorflow as tf
from keras import backend as K

from pbt.population import Member, exploit, BatchGenerator

DATASET = tf.keras.datasets.boston_housing

BATCH_SIZE = 64
STEPS_TO_READY = 5

TEST_MODEL = keras.models.Sequential([
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(1)
])

config = tf.ConfigProto(intra_op_parallelism_threads=1,
                        inter_op_parallelism_threads=1)


class TestsPbtMethods(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        (x_train, y_train), (x_test, y_test) = DATASET.load_data()
        cls.x_train, cls.y_train = x_train, y_train
        cls.x_test, cls.y_test = x_test, y_test

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

    def create_test_member(self):
        return Member(TEST_MODEL,
                      BatchGenerator(self.x_train, self.y_train,
                                     self.x_test, self.y_test,
                                     batch_size=BATCH_SIZE),
                      STEPS_TO_READY)

    def test_step(self):
        """Training two members of the population with the same parameters."""
        ma = self.create_test_member()
        loss_a = ma.step()
        ma_reg_config = ma.regularizer.get_config()
        self.close_current_session()
        self.start_nonrandom_session()
        mb = self.create_test_member()
        loss_b = mb.step()
        mb_reg_config = mb.regularizer.get_config()
        self.assertDictEqual(ma_reg_config, mb_reg_config)
        self.assertEqual(loss_a, loss_b)

    def test_eval(self):
        """Training one model for a few steps to evaluate the performance."""
        ma = self.create_test_member()
        loss_1 = ma.eval()
        for i in range(5):
            ma.step()
        loss_n = ma.eval()
        self.assertLess(loss_n, loss_1)

    def test_ready(self):
        """Train one model for the required number of steps to be ready."""
        ma = self.create_test_member()
        self.assertFalse(ma.ready())
        for i in range(STEPS_TO_READY):
            ma.step()
        self.assertTrue(ma.ready())
        ma.step()
        self.assertFalse(ma.ready())

    def test_exploit(self):
        """Train a population and check that worst members are replaced."""
        population_size = 2
        # if we increase the size, we cannot predict which one is going to
        # replaced or not
        population = []
        for i in range(population_size):
            population.append(self.create_test_member())

        # Make the loss of one the members significantly lower than the rest
        member_best = population[0]
        for i in range(5):
            member_best.step()

        # Call eval to update 'last_loss' in members
        for member in population:
            member.eval()

        exploit(population)
        for member in population:
            self.assert_list_arrays_equal(member.model.get_weights(),
                                          member_best.model.get_weights())
            self.assertDictEqual(member.regularizer.get_config(),
                                 member_best.regularizer.get_config())

    def test_explore(self):
        """Training two members of the population. Calling 'explore' in one."""
        ma = self.create_test_member()
        ma.step()
        loss_a = ma.step()
        ma_reg_config = ma.regularizer.get_config()
        self.close_current_session()
        self.start_nonrandom_session()
        mb = self.create_test_member()
        mb.explore()
        loss_b = mb.step()
        mb_reg_config = mb.regularizer.get_config()
        self.assertRaises(AssertionError, self.assertDictEqual,
                          ma_reg_config, mb_reg_config)
        self.assertNotEqual(loss_a, loss_b)

    def test_replace(self):
        """Replacing the hyperparameters and weights of one model."""
        ma = self.create_test_member()
        mb = self.create_test_member()
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
