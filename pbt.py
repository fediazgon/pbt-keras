import keras

from hyperparameters.regularization import L1L2Explorer


class Member:

    def __init__(self):
        self.regularizer = L1L2Explorer(l1=1e-5, l2=1e-5, perturb_factors=[0.8, 1.2])

        self.model = self._create_model()
        self.model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy'])

    def _create_model(self):
        model = keras.models.Sequential([
            keras.layers.Flatten(),
            keras.layers.Dense(512, activation='relu', kernel_regularizer=self.regularizer),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(10, activation='softmax', kernel_regularizer=self.regularizer)
        ])
        return model
