import keras
import numpy as np
from keras.layers import Dense, Conv1D, Conv2D, Conv3D

from pbt.hyperparameters import L1L2Mutable


class Member:

    def __init__(self, model, batch_generator, steps_to_ready):
        self.batch_generator = batch_generator

        self.steps_remaining_ready = self.steps_to_ready = steps_to_ready
        self.total_steps = 0

        self.last_loss = 0

        self.regularizer = L1L2Mutable(l1=1e-5, l2=1e-5)
        self.model = keras.models.clone_model(model)
        self._set_kernel_regularizer()

        self.model.compile(
            optimizer='adam',
            loss='mean_squared_error')

    def _set_kernel_regularizer(self):
        for layer in self.model.layers:
            if isinstance(layer, (Dense, Conv1D, Conv2D, Conv3D)):
                layer.kernel_regularizer = self.regularizer

    def step(self):
        """Step of gradient descent with Adam on model weights."""
        x, y = self.batch_generator.next()
        train_loss = self.model.train_on_batch(x, y)
        self.total_steps += 1
        self.steps_remaining_ready -= 1
        return train_loss

    def eval(self):
        """Evaluate the current model by computing the loss on the validation
        set."""
        x, y = self.batch_generator.val()
        eval_loss = self.model.evaluate(x, y, verbose=0)
        self.last_loss = eval_loss
        return eval_loss

    def ready(self):
        """Returns if the member of the population is considered ready to
        exploit and explore"""
        # In case the user call step twice just when the model is ready
        if self.steps_remaining_ready <= 0:
            self.steps_remaining_ready = self.steps_to_ready
            return True
        else:
            return False

    def explore(self):
        """Randomly perturb regularization by a factor of 0.8 or 1.2."""
        factors = [0.8, 1.2]
        self.regularizer.perturb(factors)

    def replace_with(self, member):
        """Replace the hyperparameters and weights of this member of with the
        hyperparameters and the weights of the given member."""
        self.model.set_weights(member.model.get_weights())
        self.regularizer.replace_with(member.regularizer)


def exploit(population):
    """Truncation selection: rank all the agents in the population by loss.
    If the current agent is in the bottom 20% of the population, we sample
    another agent uniformly from the top 20% of the population, and copy its
    weights and hyperparameters."""
    losses = np.array([member.last_loss for member in population])
    # Lower is better. Top 20% means percentile 20 in losses
    threshold_best, threshold_worst = np.percentile(losses, [20, 80])
    top_performers = [member for member in population
                      if member.last_loss < threshold_best]
    for member in population:
        if member.last_loss > threshold_worst:
            top_member = np.random.choice(top_performers)
            member.replace_with(top_member)


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

    def val(self):
        return self.x_test, self.y_test
