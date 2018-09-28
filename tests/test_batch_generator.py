import unittest

import numpy as np

from pbt import BatchGenerator

BATCH_AXIS_SIZE = 128


class TestsBatchGenerator(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.data_x = np.random.randn(BATCH_AXIS_SIZE, 8, 8, 3)
        cls.data_y = np.random.rand(BATCH_AXIS_SIZE, )

    def test_all_batches_equal_size(self):
        """Total number of examples is a multiple of batch size."""
        batch_size = 64
        generator = BatchGenerator(self.data_x, self.data_y, self.data_x, self.data_y, batch_size=batch_size)
        batch1_x, batch1_y = generator.next()
        batch2_x, batch2_y = generator.next()
        batch3_x, batch3_y = generator.next()
        self.assertEqual(batch1_x.shape[0], batch2_x.shape[0])
        self.assertEqual(batch2_x.shape[0], batch3_x.shape[0])
        np.testing.assert_array_equal(batch1_x, batch3_x)
        np.testing.assert_array_equal(batch1_y, batch3_y)

    def test_last_batch_different_size(self):
        """Total number of examples is not a multiple of batch size."""
        batch_size = 100
        generator = BatchGenerator(self.data_x, self.data_y, self.data_x, self.data_y, batch_size=batch_size)
        batch1_x, batch1_y = generator.next()
        batch2_x, batch2_y = generator.next()
        batch3_x, batch3_y = generator.next()
        self.assertEqual(batch_size, batch1_x.shape[0])
        self.assertEqual(BATCH_AXIS_SIZE - batch_size, batch2_x.shape[0])
        np.testing.assert_array_equal(batch1_x, batch3_x)
        np.testing.assert_array_equal(batch1_y, batch3_y)

    def test_total_example_less_batch_size(self):
        """The total number of examples is less than the batch size."""
        batch_size = BATCH_AXIS_SIZE * 2
        generator = BatchGenerator(self.data_x, self.data_y, self.data_x, self.data_y, batch_size=batch_size)
        batch1_x, batch1_y = generator.next()
        batch2_x, batch2_y = generator.next()
        self.assertEqual(BATCH_AXIS_SIZE, batch1_x.shape[0])
        np.testing.assert_array_equal(batch1_x, batch2_x)
        np.testing.assert_array_equal(batch1_y, batch2_y)
