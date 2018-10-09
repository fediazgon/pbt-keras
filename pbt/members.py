import logging
from collections import namedtuple

import numpy as np
from keras.layers import Dense, Conv1D, Conv2D, Conv3D

from pbt import hyperparameters

log = logging.getLogger(__name__)

LossRecord = namedtuple('LossRecord', 'step loss')


class Member:

    def __init__(self, build_fn, batch_generator, steps_to_ready):
        self.model = build_fn()
        self.batch_generator = batch_generator
        self.steps_remaining_ready = self.steps_to_ready = steps_to_ready

        self.total_steps = 0
        self.loss_history = []

        self.regularizer = hyperparameters.l1l2(l1=1e-5, l2=1e-5)
        self._set_kernel_regularizer()

    def step(self):
        """Step of gradient descent with Adam on model weights.

        """
        x, y = self.batch_generator.next()
        train_loss = self.model.train_on_batch(x, y)
        self.total_steps += 1
        self.steps_remaining_ready -= 1
        return train_loss

    def eval(self):
        """Evaluate the current model by computing the loss on the validation
        set.

        """
        x, y = self.batch_generator.val()
        eval_loss = self.model.evaluate(x, y, verbose=0)
        self.loss_history.append(
            LossRecord(step=self.total_steps, loss=eval_loss))
        return eval_loss

    def ready(self):
        """Returns if the member of the population is considered ready to
        exploit and explore.

        """
        # In case the user call step twice just when the model is ready
        if self.steps_remaining_ready <= 0:
            self.steps_remaining_ready = self.steps_to_ready
            return True
        else:
            return False

    def explore(self):
        """Randomly perturb regularization by a factor of 0.8 or 1.2.

        """
        factors = [0.8, 1.2]
        self.regularizer.perturb(factors)

    def exploit(self, population):
        """Truncation selection.

        Rank all the agents in the population by loss. If the current agent is
        in the bottom 20% of the population, we sample another agent uniformly
        from the top 20% of the population, and copy its weights and
        hyperparameters.

        """
        log.debug('Exploit. Deciding fate of member {}'.format(self))
        losses = np.array([m.loss_history[-1].loss for m in population])
        member_loss = self.loss_history[-1].loss
        # Lower is better. Top 20% means percentile 20 in losses
        threshold_best, threshold_worst = np.percentile(losses, (20, 80))
        log.debug('Top 20 loss is {:f}, bottom 20 is {:f}, member loss is {:f}'
                  .format(threshold_best, threshold_worst, member_loss))
        if member_loss > threshold_worst:
            log.debug('Underperforming! Replacing weights and hyperparameters')
            top_performers = [m for m in population
                              if m.loss_history[-1].loss < threshold_best]
            self.replace_with(np.random.choice(top_performers))
            return True
        else:
            log.debug('Member is doing great')
            return False

    def replace_with(self, member):
        """Replace the hyperparameters and weights of this member with the
        hyperparameters and the weights of the given member.

        """
        self.model.set_weights(member.model.get_weights())
        self.regularizer.replace_with(member.regularizer)

    def _set_kernel_regularizer(self):
        for layer in self.model.layers:
            if isinstance(layer, (Dense, Conv1D, Conv2D, Conv3D)):
                layer.kernel_regularizer = self.regularizer

    def __str__(self):
        return str(id(self))
