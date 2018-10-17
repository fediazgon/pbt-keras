from collections import defaultdict

import keras
import keras.backend as K
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.utils.generic_utils import Progbar
from sklearn import preprocessing
from sklearn.model_selection import ParameterGrid, train_test_split

import pbt

fmt = '%(asctime)s - %(name)10.10s - %(levelname)5.5s - %(message)s'

test_split = 0.3
# population_size = 20
batch_size = 32  # one step in a member is equivalent to train on a batch
total_steps = 1000  # total number of steps before ending training
steps_to_ready = 50  # number of steps before updating a member


def create_model_fn(data_dim, l1=1e-5, l2=1e-5):
    """Returns a function which can be called with no parameters to create a new
     model.

    This function is required by Member and CrossValidation. Both classes invoke
     this function to construct a new instance of the model.

    Returns:
        function: a function that can be called to get a compiled Keras model.

    """

    def _create_model_fn():
        np.random.seed(42)
        model = keras.models.Sequential([
            keras.layers.Dense(64,
                               activation='relu',
                               input_shape=(data_dim,),
                               kernel_regularizer=
                               pbt.hyperparameters.l1_l2(l1, l2)),
            keras.layers.Dense(1,
                               kernel_regularizer=
                               pbt.hyperparameters.l1_l2(l1, l2)),
        ])
        model.compile(optimizer='adam', loss='mean_squared_error')
        return model

    return _create_model_fn


def create_member(data_dim, **h):
    """Create a new population member.

    Each member wraps a Keras model with the same architecture. However, each
    one of them has a different starting value for the model's hyperparameters.

    Args:
        data_dim (int): number of features in the training set.
        **h: hyperparameters.

    Returns:
        pbt.members.Member: A new population Member.

    """
    return pbt.members.Member(create_model_fn(data_dim, **h), steps_to_ready)


def statistics(values, suffix=''):
    """Returns a List of tuples to use with Keras Progbar.

    Args:
        values (List[int]): list of values.
        suffix (str): suffix to add to each metric (i.e., min_<suffix>)

    Return:
        Minimum, maximum and mean value for the given list.

    """
    min_value = ('min_' + suffix, min(values))
    max_value = ('max_' + suffix, max(values))
    mean_value = ('mean_' + suffix, sum(values) / len(values))
    return [min_value, max_value, mean_value]


def main():
    # Load dataset. Split training dataset into train and validation sets
    dataset = tf.keras.datasets.boston_housing
    (x_train, y_train), (x_test, y_test) = dataset.load_data()
    x_train, x_val, y_train, y_val = train_test_split(
        x_train, y_train, test_size=test_split)
    print('Train examples: {}, val examples: {}, test examples {}:'.format(
        x_train.shape[0], x_val.shape[0], x_test.shape[0]
    ))

    num_examples, num_features = x_train.shape

    # ------------------------------------------
    # SEARCH SPACE
    # ------------------------------------------
    l1 = np.linspace(1e-5, 0.01, num=3).tolist()
    l2 = np.linspace(1e-5, 0.01, num=3).tolist()
    param_grid = ParameterGrid(dict(l1=l1, l2=l2))

    # ------------------------------------------
    # GRID SEARCH
    # ------------------------------------------
    K.clear_session()

    # Train using both approaches during the same period of time (steps)
    steps_per_epoch = int(num_examples / batch_size)
    epochs = int(total_steps / steps_per_epoch)

    gs_results = defaultdict(lambda: [])

    progbar = Progbar(len(param_grid),
                      stateful_metrics=['min_loss', 'max_loss', 'mean_loss'])

    for idx, h in enumerate(param_grid):
        model = create_model_fn(num_features, **h)()
        model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs,
                  verbose=0, validation_data=(x_val, y_val), shuffle=False)
        model_id = 'Model_{}'.format(idx)
        final_loss = model.test_on_batch(x_val, y_val)
        gs_results['model_id'].append(model_id)
        gs_results['final_loss'].append(final_loss)
        for h, v in h.items():
            gs_results[h].append(v)

        loss_stats = statistics(gs_results['final_loss'], suffix='loss')
        progbar.update(idx + 1, loss_stats)

    gs_results_df = pd.DataFrame(gs_results)

    print('***** GRID SEARCH RESULTS *****')
    print('** Ranking **')
    print(gs_results_df.sort_values('final_loss'))
    print('** Statistics **')
    print(gs_results_df['final_loss'].describe())
    print('** Best hyperparameters **')
    best = gs_results_df.iloc[[gs_results_df['final_loss'].idxmin()]]
    print(best.filter(regex="l.+"))

    # ------------------------------------------
    # PBT
    # ------------------------------------------
    K.clear_session()

    batch_generator = pbt.utils.BatchGenerator(x_train, y_train, batch_size)
    population = []
    for h in param_grid:
        # member = create_member(x_train.shape[1], **np.random.choice(param_grid))
        member = create_member(x_train.shape[1], **h)
        population.append(member)
    
    pbt_results = defaultdict(lambda: [])

    progbar = Progbar(total_steps,
                      stateful_metrics=['min_loss', 'max_loss', 'mean_loss'])

    step = 0
    population_size = len(population)
    while step < total_steps:
        x, y = batch_generator.next()
        for idx, member in enumerate(population):
            # One step of optimisation using hyperparameters of 'member'
            member.step_on_batch(x, y)
            # Model evaluation
            loss = member.eval_on_batch(x_val, y_val)
            # If optimised for 'STEPS_TO_READY' steps
            if member.ready():
                # Use the rest of population to find better solutions
                exploited = member.exploit(population)
                # If new weights != old weights
                if exploited:
                    # Produce new hyperparameters for 'member'
                    member.explore()
                    loss = member.eval_on_batch(x_val, y_val)
                pbt_results['model_id'].append(str(member))
                pbt_results['step'].append(step)
                pbt_results['loss'].append(loss)
                for h, v in member.get_hyperparameter_config().items():
                    pbt_results[h].append(v)

                # Update progress bar after updating last member
                if idx == population_size - 1:
                    # Get recently added losses to show in the progress bar
                    all_losses = pbt_results['loss']
                    recent_losses = all_losses[-population_size:]
                    if recent_losses:
                        loss_stats = statistics(recent_losses, suffix='loss')
                        progbar.update(step, loss_stats)

        step += 1
        progbar.update(step)

    pbt_results_df = pd.DataFrame(pbt_results)
    pbt_results_df['model_id'] = pbt_results_df['model_id'].astype('category')
    le = preprocessing.LabelEncoder()
    le.fit(pbt_results_df['model_id'])
    pbt_results_df['model_id_idx'] = le.transform(pbt_results_df['model_id'])

    print('***** PBT RESULTS *****')
    pbt_final_df = pbt_results_df.tail(population_size)
    print('** Ranking **')
    print(pbt_final_df[['model_id', 'loss']].sort_values('loss'))
    print('** Statistics **')
    print(pbt_final_df['loss'].describe())
    print('** Best hyperparameters **')
    best = pbt_results_df.iloc[[pbt_final_df['loss'].idxmin()]]
    print(best.filter(regex=".+:.+"))


if __name__ == "__main__":
    main()
