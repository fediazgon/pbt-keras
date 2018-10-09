from abc import ABC, abstractmethod

import numpy as np
from keras import backend as K
from keras.regularizers import Regularizer


class Hyperparameter(ABC):
    """Base class of all hyperparameters that can be modified while training a
    model.

    """

    @abstractmethod
    def __init__(self):
        self.history = []

    @abstractmethod
    def perturb(self, factors):
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

    def save_config_record(self):
        """Save an history record with the current configuration.

        """
        self.history.append(self.get_config())


class L1L2Mutable(Hyperparameter, Regularizer):
    """To be used with 'kernel_regularizer' in a Keras layer.

    """

    def __init__(self, l1=0., l2=0.):
        super(L1L2Mutable, self).__init__()
        self.l1 = K.variable(K.cast_to_floatx(l1), name='l1')
        self.l2 = K.variable(K.cast_to_floatx(l2), name='l2')

    def perturb(self, factors):
        self.save_config_record()
        K.set_value(self.l1,
                    K.get_value(self.l1) * np.random.choice(factors))
        K.set_value(self.l2,
                    K.get_value(self.l2) * np.random.choice(factors))

    def replace_with(self, regularizer):
        self.save_config_record()
        K.set_value(self.l1,
                    K.cast_to_floatx(regularizer.get_config().get('l1', 0)))
        K.set_value(self.l1,
                    K.cast_to_floatx(regularizer.get_config().get('l2', 0)))

    def get_config(self):
        return {'l1': float(K.get_value(self.l1)),
                'l2': float(K.get_value(self.l2))}

    def __call__(self, x):
        regularization = 0.
        if K.get_value(self.l1):
            regularization += K.sum(self.l1 * K.abs(x))
        if K.get_value(self.l2):
            regularization += K.sum(self.l2 * K.square(x))
        return regularization


# Aliases
def l1_l2(l1=0., l2=0.):
    return L1L2Mutable(l1, l2)
