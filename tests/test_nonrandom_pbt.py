import keras
import numpy as np
import pytest
from keras import backend as K
from keras.utils import test_utils

from pbt.members import Member

data_dim = 10
batch_size = 64
steps_to_ready = 5


def get_data():
    (x_train, y_train), _ = test_utils.get_test_data(
        num_train=batch_size,
        num_test=batch_size,
        input_shape=(data_dim,),
        output_shape=(1,),
        classification=False)
    return x_train, y_train


def get_test_model():
    np.random.seed(42)
    model = keras.models.Sequential([
        keras.layers.Dense(64, activation='relu', input_shape=(data_dim,)),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model


def get_test_member():
    return Member(get_test_model, steps_to_ready)


def assert_list_arrays_equal(list_a, list_b):
    for i in range(len(list_a)):
        np.testing.assert_array_equal(list_a[i], list_b[i])


def test_step():
    """Training two members of the population with the same parameters."""
    x, y = get_data()
    ma = get_test_member()
    loss_a = ma.step_on_batch(x, y)
    ma_reg_config = ma.regularizer.get_config()
    K.clear_session()
    mb = get_test_member()
    loss_b = mb.step_on_batch(x, y)
    mb_reg_config = mb.regularizer.get_config()
    assert loss_a == loss_b
    assert ma_reg_config == mb_reg_config


def test_eval():
    """Training one model for a few steps to evaluate the performance."""
    ma = get_test_member()
    loss_1 = ma.eval_on_batch(*get_data())
    for i in range(5):
        ma.step_on_batch(*get_data())
    loss_n = ma.eval_on_batch(*get_data())
    assert loss_n < loss_1


def test_ready():
    """Train one model for the required number of steps to be ready."""
    ma = get_test_member()
    assert not ma.ready()
    for i in range(steps_to_ready):
        ma.step_on_batch(*get_data())
    assert ma.ready()
    ma.step_on_batch(*get_data())
    assert not ma.ready()


def test_exploit():
    """Train a population and check that worst members are replaced."""
    member_best = get_test_member()
    member_worst = get_test_member()
    population = [member_best, member_worst]
    # Make the loss of one the members significantly lower than the rest
    for i in range(20):
        member_best.step_on_batch(*get_data())
    # Call eval to update 'last_loss' in members
    for member in population:
        member.eval_on_batch(*get_data())
    # Consider 'member_worst' is ready
    member_worst.exploit(population)
    member_best_weights = member_best.model.get_weights()
    member_best_reg_config = member_best.regularizer.get_config()
    for member in population:
        member_weights = member.model.get_weights()
        member_reg_config = member.regularizer.get_config()
        assert_list_arrays_equal(member_best_weights, member_weights)
        assert member_best_reg_config == member_reg_config


def test_explore():
    """Training two members of the population. Calling 'explore' in one."""
    ma = get_test_member()
    loss_a = ma.step_on_batch(*get_data())
    ma_reg_config = ma.regularizer.get_config()
    K.clear_session()
    mb = get_test_member()
    mb.explore()
    loss_b = mb.step_on_batch(*get_data())
    mb_reg_config = mb.regularizer.get_config()
    assert loss_a != loss_b
    assert ma_reg_config != mb_reg_config


def test_replace():
    """Replacing the hyperparameters and weights of one model."""
    ma = get_test_member()
    mb = get_test_member()
    loss_a = ma.step_on_batch(*get_data())
    loss_b = mb.step_on_batch(*get_data())
    assert loss_a != loss_b
    with pytest.raises(AssertionError):
        assert_list_arrays_equal(ma.model.get_weights(), mb.model.get_weights())
    ma.replace_with(mb)
    assert_list_arrays_equal(ma.model.get_weights(), mb.model.get_weights())
    assert ma.regularizer.get_config() == mb.regularizer.get_config()


if __name__ == '__main__':
    pytest.main([__file__])
