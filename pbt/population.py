import keras

from pbt.hyperparameters import L1L2Mutable


class Member:

    def __init__(self, batch_generator):
        self.batch_generator = batch_generator
        self.total_steps = 0

        self.regularizer = L1L2Mutable(l1=1e-5, l2=1e-5)

        self.model = self._create_model()
        self.model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy'])

    def _create_model(self):
        model = keras.models.Sequential([
            keras.layers.Flatten(),
            keras.layers.Dense(512,
                               activation='relu',
                               kernel_regularizer=self.regularizer),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(10,
                               activation='softmax',
                               kernel_regularizer=self.regularizer)
        ])
        return model

    def step(self):
        """Step of gradient descent with Adam on model weights."""
        x, y = self.batch_generator.next()
        train_loss, train_accuracy = self.model.train_on_batch(x, y)
        self.total_steps += 1
        return train_loss, train_accuracy

    def explore(self):
        """Randomly perturb regularization by a factor of 0.8 or 1.2"""
        factors = [0.8, 1.2]
        self.regularizer.perturb(factors)

    def replace_with(self, member):
        """Replace the hyperparameters and weights of this member of with the
        hyperparameters and the wights of the given member."""
        self.model.set_weights(member.model.get_weights())
        self.regularizer.replace_with(member.regularizer)


class BatchGenerator:
    def __init__(self, x_train, y_train, x_test, y_test, batch_size=64):
        self.x_train, self.y_train = x_train, y_train
        self.x_test, self.y_test = x_test, y_test
        self.batch_size = batch_size
        self.num_examples = self.x_train.shape[0]
        self.k = 0  # current batch index

    def next(self):
        first_index = self.k * self.batch_size
        last_index = (self.k + 1) * self.batch_size
        if last_index <= self.num_examples:
            batch_x = self.x_train[first_index:last_index]
            batch_y = self.y_train[first_index:last_index]
            if last_index == self.num_examples:
                self.k = 0
            else:
                self.k += 1
        else:
            batch_x = self.x_train[first_index:]
            batch_y = self.y_train[first_index:]
            self.k = 0
        return batch_x, batch_y
