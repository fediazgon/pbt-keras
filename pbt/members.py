import numpy as np

from pbt.hyperparameters import find_hyperparameters_layer, \
    find_hyperparameters_model


class Member:
    """Population member.

    Each member wraps an instance of a Keras model to tune. The member holds
    references to the hyperparameters of this model, allowing it to change them.

    Members of the same population are characterized by the behaviour of the
    following methods: step, eval, ready, exploit and explore. If you think
    the current implementation of these methods does not work for your problem,
    just create a subclass and override them.

    """

    def __init__(self, build_fn, steps_ready=None):
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

        self.hyperparameters = find_hyperparameters_model(self.model)
        if not self.hyperparameters:
            raise ValueError('The model has no hyperparameters to tune')

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
        self.current_loss = self.model.evaluate(x, y, verbose=0)
        return self.current_loss

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
        self.model.set_weights(member.model.get_weights())
        for i, layer1 in enumerate(self.model.layers):
            layer2 = member.model.layers[i]
            h_layer1 = find_hyperparameters_layer(layer1)
            h_layer2 = find_hyperparameters_layer(layer2)
            for (h1, h2) in zip(h_layer1, h_layer2):
                h1.replace_with(h2)

    def get_hyperparameter_config(self):
        config = {}
        for idx, layer in enumerate(self.model.layers):
            # layer_name = layer.get_config().get('name')
            h_layer = find_hyperparameters_layer(layer)
            for h in h_layer:
                for k, v in h.get_config().items():
                    config['{}:{}'.format(k, idx)] = v
        return config

    def __str__(self):
        return str(id(self))
