import keras
import keras.backend as K
import numpy as np
import tensorflow as tf
from sklearn.model_selection import ParameterGrid

import pbt
from pbt.utils import train_population

VALIDATION_SPLIT = 0.3
# population_size = 20
BATCH_SIZE = 32  # one step in a member is equivalent to train on a batch
TOTAL_STEPS = 1000  # total number of steps before ending training
STEPS_READY = 50  # number of steps before updating a member
STEPS_SAVE = 100  # number of steps before saving the status of the population

# ------------------------------------------
# SEARCH SPACE
# ------------------------------------------
l1_values = np.linspace(1e-5, 0.01, num=4).tolist()
l2_values = np.linspace(1e-5, 0.01, num=4).tolist()
param_grid = ParameterGrid(dict(l1=l1_values, l2=l2_values))


def build_fn(data_dim, l1=1e-5, l2=1e-5):
    """Returns a function which can be called with no parameters to create a new
     model.

    The returned function is required by Member, which invokes it to construct a
    new instance of the model.

    Returns:
        function: a function that can be called to get a compiled Keras model.

    """

    def _build_fn():
        # Set seed to get same weight initialization
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

    return _build_fn


def show_results(df):
    print('***** RESULTS *****')
    population_size = df['model_id'].nunique()
    df_final = df.tail(population_size)
    print('** Ranking **')
    print(df_final[['model_id', 'loss']].sort_values('loss'))
    print('** Statistics **')
    print(df_final['loss'].describe())
    print('** Best hyperparameters **')
    best = df.iloc[[df_final['loss'].idxmin()]]
    print(best.filter(regex=".+:.+"))


def main():
    # The following configuration is to make sure the final comparison is not
    # biased by randomness (e.g., initialization)
    session_conf = tf.ConfigProto(intra_op_parallelism_threads=1,
                                  inter_op_parallelism_threads=1)
    sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
    K.set_session(sess)

    # Load dataset. Split training dataset into train and validation sets
    dataset = tf.keras.datasets.boston_housing
    (x_train, y_train), (x_test, y_test) = dataset.load_data()
    print('Train examples: {}, test examples: {}'.format(
        x_train.shape[0], x_test.shape[0]
    ))
    data_dim = x_train.shape[1]

    # ------------------------------------------
    # GRID SEARCH
    # ------------------------------------------

    # We create a population where we never explore new hyperparameters
    population = []
    for h in param_grid:
        member = pbt.members.Member(build_fn(data_dim, **h),
                                    steps_ready=None)
        population.append(member)

    res_gd = train_population(population, x_train, y_train, BATCH_SIZE,
                              TOTAL_STEPS, STEPS_SAVE, VALIDATION_SPLIT)
    show_results(res_gd)

    # ------------------------------------------
    # PBT
    # ------------------------------------------
    K.clear_session()

    # This time, the members will explore new hyperparameters
    population = []
    for h in param_grid:
        member = pbt.members.Member(build_fn(data_dim, **h),
                                    steps_ready=STEPS_READY)
        population.append(member)

    res_pbt = train_population(population, x_train, y_train, BATCH_SIZE,
                               TOTAL_STEPS, STEPS_SAVE, VALIDATION_SPLIT)
    show_results(res_pbt)


if __name__ == "__main__":
    main()
