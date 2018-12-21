import logging
from abc import ABC, abstractmethod

import numpy as np
from keras import backend as K
from keras.layers import Dropout
from keras.regularizers import Regularizer

log = logging.getLogger(__name__)


class Hyperparameter(ABC):
    """Base class of all hyperparameters that can be modified while training a
    model.

    """

    @abstractmethod
    def perturb(self, factors=None):
        """Perturb the hyperparameter with a random chosen factor.

        Args:
            factors (List[double]): factors to choose from.

        """
        pass

    @abstractmethod
    def replace_with(self, hyperparameter):
        """Replace the configuration of this hyperparameter with the
        configuration of the given one.

        Args:
            hyperparameter (Hyperparameter): hyperparameter to copy.

        """

        pass

    @abstractmethod
    def get_config(self):
        """Return the configuration (value(s)) for this hyperparameter.

        Returns:
             dict: dictionary where the key is the name of the hyperparameter.

        """
        pass


class L1L2Mutable(Hyperparameter, Regularizer):
    """To be used with 'kernel_regularizer' in a Keras layer.

    """

    def __init__(self, l1=0., l2=0.):
        super(L1L2Mutable, self).__init__()
        self.l1 = K.variable(K.cast_to_floatx(l1), name='l1')
        self.l2 = K.variable(K.cast_to_floatx(l2), name='l2')

    def perturb(self, factors=None):
        if not factors:
            factors = [0.8, 1.2]
        K.set_value(self.l1,
                    K.get_value(self.l1) * np.random.choice(factors))
        K.set_value(self.l2,
                    K.get_value(self.l2) * np.random.choice(factors))

    def replace_with(self, regularizer):
        K.set_value(self.l1,
                    K.cast_to_floatx(regularizer.get_config().get('l1')))
        K.set_value(self.l2,
                    K.cast_to_floatx(regularizer.get_config().get('l2')))

    def get_config(self):
        return {'l1': float(K.get_value(self.l1)),
                'l2': float(K.get_value(self.l2))}

    def __call__(self, x):
        # Useful for testing. Since it is impossible to patch __call__
        logging.debug('Called {}'.format(self))
        regularization = 0.
        if K.get_value(self.l1):
            regularization += K.sum(self.l1 * K.abs(x))
        if K.get_value(self.l2):
            regularization += K.sum(self.l2 * K.square(x))
        return regularization


# Aliases
def l1_l2(l1=0., l2=0.):
    return L1L2Mutable(l1, l2)


class DropoutMutable(Hyperparameter, Dropout):

    def __init__(self, rate, noise_shape=None, seed=None, **kwargs):
        super().__init__(rate, noise_shape=noise_shape, seed=seed, **kwargs)

    def perturb(self, factors=None):
        if not factors:
            factors = [0.8, 1.2]
        self.rate = self.rate * np.random.choice(factors)

    def replace_with(self, hyperparameter):
        self.rate = hyperparameter.get_config().get('dr')

    def get_config(self):
        return {'dr': self.rate}


class FloatHyperparameter(Hyperparameter):

    def __init__(self, name, variable):
        self.name = name
        self.variable = variable

    def perturb(self, factors=None):
        if not factors:
            factors = [0.8, 1.2]
        K.set_value(self.variable,
                    K.get_value(self.variable) * np.random.choice(factors))

    def replace_with(self, hyperparameter):
        K.set_value(self.variable, K.cast_to_floatx(
            hyperparameter.get_config().get(self.name)))

    def get_config(self):
        return {self.name: float(K.get_value(self.variable))}


def find_hyperparameters_model(keras_model):
    """Finds instances of class Hyperparameter that are used in the given model.

    For example, in the following model::

        keras.layers.Dense(
            42,
            kernel_regularizer=pbt.hyperparameters.L1L2Mutable(l1=0, l2=1e-5),
            bias_initializer=keras.initializers.Zeros(),
            input_shape=(13,)
        )

    L1L2Mutable is an instance of pbt.hyperparameters.Hyperparameter, but
    l1_l2 is not. As a result, the method will only return the former.

    Args:
        keras_model (keras.models.Sequential): a compiled Keras model.

    Returns:
        A list of hyperparameters.

    """
    hyperparameters = []
    for layer in keras_model.layers:
        if isinstance(layer, Hyperparameter):
            hyperparameters.append(layer)
        else:
            hyperparameters.extend(find_hyperparameters_layer(layer))
    return hyperparameters


def find_hyperparameters_layer(keras_layer):
    """Finds instances of class Hyperparameter that are used in the given layer.

    For example, in the following model::

        keras.layers.Dense(
            42,
            kernel_regularizer=pbt.hyperparameters.L1L2Mutable(l1=0, l2=1e-5),
            bias_initializer=keras.initializers.Zeros(),
            input_shape=(13,)
        )

    L1L2Mutable is an instance of pbt.hyperparameters.Hyperparameter, but
    Zeros is not. As a result, the method will only return the former.

    Args:
        keras_layer (keras.layers.Layer): a Keras layer object.

    Returns:
        A list of hyperparameters.

    """
    hyperparameters_names = ['kernel_regularizer']
    hyperparameters = []
    for h_name in hyperparameters_names:
        if hasattr(keras_layer, h_name):
            h = getattr(keras_layer, h_name)
            if isinstance(h, Hyperparameter):
                hyperparameters.append(h)
    return hyperparameters
