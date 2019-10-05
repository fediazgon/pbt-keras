from unittest.mock import call, patch

import numpy as np
import pytest

from pbt import TF

if TF:
    from tensorflow.keras import backend as K
    from tensorflow.keras.layers import Dense
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.utils import test_utils as k_test_utils
else:
    from keras import backend as K
    from keras.layers import Dense
    from keras.models import Sequential
    from keras.optimizers import Adam
    from keras.utils import test_utils as k_test_utils

from pbt.hyperparameters import L1L2Mutable, DropoutMutable
from pbt.members import Member

data_dim = 10
batch_size = 64
steps_to_ready = 5


# *****************************
# ***** UTILITY FUNCTIONS *****
# *****************************
def get_data():
    (x_train, y_train), _ = k_test_utils.get_test_data(
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
    hyperparameter3 = DropoutMutable(rate=0.1)
    model = Sequential([
        Dense(64, kernel_regularizer=hyperparameter1, input_shape=(data_dim,)),
        hyperparameter3,
        Dense(1, kernel_regularizer=hyperparameter2)
    ])
    adam = Adam(lr=0.1)
    model.compile(optimizer=adam, loss='mean_squared_error', metrics=['mse'])
    return model


def get_test_member():
    return Member(get_test_model, steps_to_ready, tune_lr=True)


# *****************************

def test_regularizers_found():
    """Checks that the member could find the two regularizers added."""
    m = Member(get_test_model, steps_to_ready)
    assert len(m.hyperparameters) == 3


def test_learning_rate_found():
    """Checks that the member knows that it should tune the learning rate."""
    m = Member(get_test_model, steps_to_ready, tune_lr=True)
    assert len(m.hyperparameters) == 4


def test_hyperparameters_config():
    """Checks that the model has the right hyperparameter configuration."""
    # Clear the session to restart the naming of the Keras layers
    K.clear_session()
    m = Member(get_test_model, steps_to_ready, tune_lr=True)
    config = m.get_hyperparameter_config()
    expected_config = {'l1:0': 0.1, 'l2:0': 1e-5,
                       'dr:1': 0.1,
                       'l1:2': 0.2, 'l2:2': 1e-6,
                       'lr': 0.1}
    for k, v in expected_config.items():
        assert v == pytest.approx(config[k], abs=1e-6)


@patch('logging.Logger.debug')
def test_hyperparameters_called(mock):
    """Checks that __call__ function of regularizers is executed."""
    x, y = get_data()
    ma = get_test_member()
    ma.step_on_batch(x, y)
    calls = [call('Called {}'.format(h)) for h in ma.hyperparameters
             if isinstance(h, L1L2Mutable)]
    mock.assert_has_calls(calls, any_order=True)


if __name__ == '__main__':
    pytest.main([__file__])
