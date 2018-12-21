from collections import defaultdict

import pandas as pd
from keras.utils.generic_utils import Progbar
from sklearn.model_selection import train_test_split


class BatchGenerator:
    """An utility class to access data by batch."""

    def __init__(self, x, y, batch_size=64):
        self.x_train, self.y_train = x, y
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


def train_population(population, x, y, batch_size, steps,
                     steps_save=100, validation_split=0.3):
    # Split data in train and validation. Set seed to get same splits in
    # consequent calls
    x_train, x_val, y_train, y_val = train_test_split(
        x, y, test_size=validation_split, random_state=42)

    population_size = len(population)
    batch_generator = BatchGenerator(x_train, y_train, batch_size)

    results = defaultdict(lambda: [])
    progbar = Progbar(steps, stateful_metrics=['min', 'max', 'mean'])

    for step in range(1, steps + 1):
        x, y = batch_generator.next()
        for idx, member in enumerate(population):
            # One step of optimisation using hyperparameters of 'member'
            member.step_on_batch(x, y)
            # Model evaluation
            loss = member.eval_on_batch(x_val, y_val)
            # If optimised for 'STEPS_READY' steps
            if member.ready():
                # Use the rest of population to find better solutions
                exploited = member.exploit(population)
                # If new weights != old weights
                if exploited:
                    # Produce new hyperparameters for 'member'
                    member.explore()
                    loss = member.eval_on_batch(x_val, y_val)

            if step % steps_save == 0 or step == steps:
                results['model_id'].append(str(member))
                results['step'].append(step)
                results['loss'].append(loss)
                results['loss_smoothed'].append(member.loss_smoothed())
                for h, v in member.get_hyperparameter_config().items():
                    results[h].append(v)

        # Get recently added losses to show in the progress bar
        all_losses = results['loss']
        recent_losses = all_losses[-population_size:]
        if recent_losses:
            loss_stats = _statistics(recent_losses)
            progbar.update(step, loss_stats)

    return pd.DataFrame(results)


def _statistics(values):
    """Returns a List of tuples to use with Keras Progbar.

    Args:
        values (List[int]): list of values.

    Return:
        Minimum, maximum and mean value for the given list.

    """
    min_value = ('min', min(values))
    max_value = ('max', max(values))
    mean_value = ('mean', sum(values) / len(values))
    return [min_value, max_value, mean_value]
