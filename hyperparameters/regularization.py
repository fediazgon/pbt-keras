import numpy as np
from keras import backend as K
from keras.regularizers import Regularizer


class L1L2Explorer(Regularizer):

    def __init__(self, l1=0., l2=0., perturb_factors=None):
        if perturb_factors is None:
            perturb_factors = [0.8, 1.2]
        self.l1 = K.variable(K.cast_to_floatx(l1), name='l1')
        self.l2 = K.variable(K.cast_to_floatx(l2), name='l2')
        self.perturb_factors = perturb_factors

    def explore(self):
        K.set_value(self.l1, K.get_value(self.l1) * np.random.choice(self.perturb_factors))
        K.set_value(self.l2, K.get_value(self.l2) * np.random.choice(self.perturb_factors))

    def get_l1_l2(self):
        return K.get_value(self.l1), K.get_value(self.l2)

    def set_l1_l2(self, l1, l2):
        K.set_value(self.l1, K.cast_to_floatx(l1))
        K.set_value(self.l2, K.cast_to_floatx(l2))

    def __call__(self, x):
        regularization = 0.
        if K.get_value(self.l1):
            regularization += K.sum(self.l1 * K.abs(x))
        if K.get_value(self.l2):
            regularization += K.sum(self.l2 * K.square(x))
        return regularization

    def get_config(self):
        return {'l1': float(K.get_value(self.l1)),
                'l2': float(K.get_value(self.l2))}
