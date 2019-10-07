import numpy as np
import pytest

from pbt import TF

if TF:
    import tensorflow as tf
    from tensorflow.keras import backend as K
    from tensorflow.keras.layers import Dense
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.utils import test_utils

    session_conf = tf.ConfigProto(intra_op_parallelism_threads=1,
                                  inter_op_parallelism_threads=1)
    sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
else:
    from keras import backend as K
    from keras.layers import Dense
    from keras.models import Sequential
    from keras.optimizers import Adam
    from keras.utils import test_utils

    sess = K.get_session()

from pbt.hyperparameters import L1L2Mutable
from pbt.members import Member

data_dim = 10
batch_size = 64
steps_to_ready = 5


K.set_session(sess)


# *****************************
# ***** UTILITY FUNCTIONS *****
# *****************************

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
    hyperparameter1 = L1L2Mutable(l1=0.1, l2=1e-5)
    hyperparameter2 = L1L2Mutable(l1=0.2, l2=1e-6)
    model = Sequential([
        Dense(64, kernel_regularizer=hyperparameter1, input_shape=(data_dim,)),
        Dense(1, kernel_regularizer=hyperparameter2)
    ])
    adam = Adam(lr=0.1)
    model.compile(optimizer=adam, loss='mean_squared_error',
                  metrics=['mse', 'mae'])
    return model


def get_test_member():
    return Member(get_test_model, steps_to_ready, tune_lr=True)


def assert_list_arrays_equal(list_a, list_b):
    for i in range(len(list_a)):
        np.testing.assert_array_equal(list_a[i], list_b[i])


def assert_hyperparameter_config_equal(config_a, config_b):
    # Since key is formed by hyperparameter:layerName_layerNumber. Sorting
    # values by key should give the same order in two different models
    config_a_values = [value for (key, value) in sorted(config_a.items())]
    config_b_values = [value for (key, value) in sorted(config_b.items())]
    np.testing.assert_allclose(config_a_values, config_b_values, atol=1e-6)


# *****************************

@pytest.yield_fixture(autouse=True)
def run_around_tests():
    K.clear_session()


def test_step():
    """Trains two members of the population with the same parameters."""
    x, y = get_data()
    ma = get_test_member()
    loss_a = ma.step_on_batch(x, y)
    for i in range(100):
        loss_a = ma.step_on_batch(x, y)
    ma_hyperparameter_config = ma.get_hyperparameter_config()
    # Start a new session to get the same results
    # Model initialization does not change because we set the seed
    K.clear_session()
    mb = get_test_member()
    loss_b = mb.step_on_batch(x, y)
    for i in range(100):
        loss_b = mb.step_on_batch(x, y)
    mb_hyperparameter_config = mb.get_hyperparameter_config()
    assert loss_a == loss_b
    assert_hyperparameter_config_equal(ma_hyperparameter_config,
                                       mb_hyperparameter_config)


def test_eval():
    """Trains one model for a few steps to evaluate the performance."""
    ma = get_test_member()
    loss_1 = ma.eval_on_batch(*get_data())
    for i in range(20):
        ma.step_on_batch(*get_data())
    loss_n = ma.eval_on_batch(*get_data())
    assert loss_n < loss_1


def test_ready():
    """Trains one model for the required number of steps to be ready."""
    ma = get_test_member()
    assert not ma.ready()
    for i in range(steps_to_ready):
        ma.step_on_batch(*get_data())
    assert ma.ready()
    ma.step_on_batch(*get_data())
    assert not ma.ready()


def test_explore():
    """Trains two members of the population. Calls 'explore' in one."""
    ma = get_test_member()
    loss_a = ma.step_on_batch(*get_data())
    ma_hyperparameter_config = ma.get_hyperparameter_config()
    # Clear session. But this time we are going to change the hyperparameters
    # of the second member (as opposed to what we did in `test_step`
    K.clear_session()
    mb = get_test_member()
    mb.explore()
    loss_b = mb.step_on_batch(*get_data())
    mb_hyperparameter_config = mb.get_hyperparameter_config()
    assert loss_a != loss_b
    with pytest.raises(AssertionError):
        assert_hyperparameter_config_equal(ma_hyperparameter_config,
                                           mb_hyperparameter_config)


def test_replace():
    """Replaces the hyperparameters and weights of one model."""
    ma = get_test_member()
    mb = get_test_member()
    loss_a = ma.step_on_batch(*get_data())
    loss_b = mb.step_on_batch(*get_data())
    mb.explore()
    assert loss_a != loss_b
    with pytest.raises(AssertionError):
        assert_list_arrays_equal(ma.model.get_weights(), mb.model.get_weights())
    with pytest.raises(AssertionError):
        assert_hyperparameter_config_equal(ma.get_hyperparameter_config(),
                                           mb.get_hyperparameter_config())
    ma.replace_with(mb)
    assert_list_arrays_equal(ma.model.get_weights(), mb.model.get_weights())
    assert_hyperparameter_config_equal(ma.get_hyperparameter_config(),
                                       mb.get_hyperparameter_config())


def test_exploit():
    """Trains a population and checks that worst members are replaced."""
    member_best = get_test_member()
    member_worst = get_test_member()
    population = [member_best, member_worst]
    # Make the loss of one the members significantly lower than the rest
    for i in range(20):
        member_best.step_on_batch(*get_data())
    # Call eval to update 'last_loss' in members
    for member in population:
        member.eval_on_batch(*get_data())
    # Consider 'member_worst' is ready. Exploit.
    member_worst.exploit(population)
    member_best_weights = member_best.model.get_weights()
    member_best_hyperparameter_config = member_best.get_hyperparameter_config()
    for member in population:
        member_weights = member.model.get_weights()
        member_hyperparameter_config = member.get_hyperparameter_config()
        assert_list_arrays_equal(member_best_weights, member_weights)
        assert_hyperparameter_config_equal(member_best_hyperparameter_config,
                                           member_hyperparameter_config)


if __name__ == '__main__':
    pytest.main([__file__])
