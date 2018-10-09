import numpy as np
import pytest

from pbt.utils import BatchGenerator

num_examples = 128


def batch_generator(batch_size):
    return BatchGenerator(
        np.random.randn(num_examples, 8, 8, 3),
        np.random.rand(num_examples, ),
        batch_size=batch_size)


def test_all_batches_equal_size():
    """Total number of examples is a multiple of batch size."""
    batch_size = int(num_examples / 2)
    generator = batch_generator(batch_size)
    batch1_x, batch1_y = generator.next()
    batch2_x, batch2_y = generator.next()
    batch3_x, batch3_y = generator.next()
    assert batch1_x.shape[0] == batch2_x.shape[0]
    assert batch2_x.shape[0] == batch3_x.shape[0]
    np.testing.assert_array_equal(batch1_x, batch3_x)
    np.testing.assert_array_equal(batch1_y, batch3_y)


def test_last_batch_different_size():
    """Total number of examples is not a multiple of batch size."""
    batch_size = num_examples - 28
    generator = batch_generator(batch_size)
    batch1_x, batch1_y = generator.next()
    batch2_x, batch2_y = generator.next()
    batch3_x, batch3_y = generator.next()
    assert batch_size == batch1_x.shape[0]
    assert num_examples - batch_size == batch2_x.shape[0]
    np.testing.assert_array_equal(batch1_x, batch3_x)
    np.testing.assert_array_equal(batch1_y, batch3_y)


def test_total_example_less_batch_size():
    """Total number of examples is less than the batch size."""
    batch_size = num_examples * 2
    generator = batch_generator(batch_size)
    batch1_x, batch1_y = generator.next()
    batch2_x, batch2_y = generator.next()
    assert num_examples == batch1_x.shape[0]
    np.testing.assert_array_equal(batch1_x, batch2_x)
    np.testing.assert_array_equal(batch1_y, batch2_y)


if __name__ == '__main__':
    pytest.main([__file__])
