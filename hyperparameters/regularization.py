import numpy as np
from keras import backend as K
from keras.regularizers import Regularizer


class L1L2Mutable(Regularizer):

    def __init__(self, l1=0., l2=0.):
        self.l1 = K.variable(K.cast_to_floatx(l1), name='l1')
        self.l2 = K.variable(K.cast_to_floatx(l2), name='l2')

    def perturb(self, perturb_factors):
        K.set_value(self.l1, K.get_value(self.l1) * np.random.choice(perturb_factors))
        K.set_value(self.l2, K.get_value(self.l2) * np.random.choice(perturb_factors))

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
