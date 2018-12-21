from collections import deque

import numpy as np

from pbt.hyperparameters import find_hyperparameters_layer, \
    find_hyperparameters_model, Hyperparameter, FloatHyperparameter


class Member:
    """Population member.

    Each member wraps an instance of a Keras model to tune. The member holds
    references to the hyperparameters of this model, allowing it to change them.

    Members of the same population are characterized by the behaviour of the
    following methods: step, eval, ready, exploit and explore. If you think
    the current implementation of these methods does not work for your problem,
    just create a subclass and override them.

    """

    def __init__(self, build_fn, steps_ready=None, tune_lr=False):
        """Creates a new population member.

        Args:
            build_fn (callable): a function that should construct, compile and
                return a Keras model. At least one layer of the model should
                hold a reference to a pbt.hyperparameters.Hyperparameter.
            steps_ready: number of steps before the member is considered ready
                to go through the exploit-and-explore process. Or 'None' if the
                member should not explore new hyperparameters.

        Raises:
            ValueError: if the given model does not have at least one layer
            holding a reference to a pbt.hyperparameters.Hyperparameter.

        """

        self.model = build_fn()

        self.steps_cycle = 0
        self.step_ready = steps_ready

        self.current_loss = np.Inf
        self.recent_losses = deque(maxlen=10)

        self.hyperparameters = find_hyperparameters_model(self.model)

        if tune_lr:
            lr = FloatHyperparameter('lr', self.model.optimizer.lr)
            self.hyperparameters.append(lr)

        if not self.hyperparameters:
            raise ValueError('The model has no hyperparameters to tune')

    def loss_smoothed(self):
        return sum(self.recent_losses) / len(self.recent_losses)

    def step_on_batch(self, x, y):
        """Gradient descent update on a single batch of data.

        Args:
            x (numpy.ndarray): numpy array of training data.
            y (numpy.ndarray): numpy array of target data.

        Returns:
            double: scalar train loss.

        """
        train_loss = self.model.train_on_batch(x, y)
        self.steps_cycle += 1
        return train_loss

    def eval_on_batch(self, x, y):
        """Evaluates the model on a single batch of samples.

        Args:
            x (numpy.ndarray): numpy array of evaluation data.
            y (numpy.ndarray): numpy array of target data.

        Returns:
            double: scalar evaluation loss.

        """
        self.current_loss = self.model.test_on_batch(x, y)
        self.recent_losses.append(self.current_loss)
        return self.current_loss

    def test_on_batch(self, x, y):
        return self.model.test_on_batch(x, y)

    def ready(self):
        """Returns if the member of the population is considered ready to
        go through the exploit-and-explore process.

        Returns:
            bool: True if this member is ready, False otherwise.

        """
        if not self.step_ready or self.steps_cycle < self.step_ready:
            return False
        else:
            self.steps_cycle = 0
            return True
            pass

    def explore(self):
        """Randomly perturbs hyperparameters by a factor of 0.8 or 1.2.

        """
        for h in self.hyperparameters:
            h.perturb(None)

    def exploit(self, population):
        """Truncation selection.

        Ranks all the agents in the population by loss. If the current agent is
        in the bottom 20% of the population, it samples another agent uniformly
        from the top 20% of the population, and copies its weights and
        hyperparameters.

        Args:
            population (List[Member]): entire population.

        Returns:
            True if the member was altered, False otherwise.

        """
        losses = np.array([m.current_loss for m in population])
        # Lower is better. Top 20% means percentile 20 in losses
        threshold_best, threshold_worst = np.percentile(losses, (20, 80))
        if self.current_loss > threshold_worst:
            top_performers = [m for m in population
                              if m.current_loss < threshold_best]
            if top_performers:
                self.replace_with(np.random.choice(top_performers))
            return True
        else:
            return False

    def replace_with(self, member):
        """Replaces the hyperparameters and weights of this member with the
        hyperparameters and the weights of the given member.

        Args:
            member (Member): member to copy.

        """
        assert len(self.hyperparameters) == len(member.hyperparameters), \
            'Members do not belong to the same population!'
        self.model.set_weights(member.model.get_weights())
        for i, hyperparameter in enumerate(self.hyperparameters):
            hyperparameter.replace_with(member.hyperparameters[i])

    def get_hyperparameter_config(self):
        config = {}
        for idx, layer in enumerate(self.model.layers):
            # layer_name = layer.get_config().get('name')
            if isinstance(layer, Hyperparameter):
                hyperparameters = [layer]
            else:
                hyperparameters = find_hyperparameters_layer(layer)
            for h in hyperparameters:
                for k, v in h.get_config().items():
                    config['{}:{}'.format(k, idx)] = v
        for h in self.hyperparameters:
            if isinstance(h, FloatHyperparameter):
                config.update(h.get_config())
        return config

    def __str__(self):
        return str(id(self))
