import logging
from collections import namedtuple

import keras
import numpy as np
import tensorflow as tf
from sklearn.model_selection import ParameterGrid, train_test_split

import pbt.members
import pbt.utils

FORMAT = '%(asctime)s - %(name)10.10s - %(levelname)5.5s - %(message)s'

TEST_SPLIT = 0.3
POPULATION_SIZE = 10
BATCH_SIZE = 32  # one step in a member is equivalent to train on a batch
TOTAL_STEPS = 1000  # total number of steps before ending training
STEPS_TO_READY = 50  # number of steps before updating a member

sh = logging.StreamHandler()
sh.setFormatter(logging.Formatter(fmt=FORMAT, datefmt='%I:%M:%S %p'))
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
logger.addHandler(sh)


def create_model_fn(l1=0.0, l2=0.0):
    """Create the model to tune.

    This function is required by Member and KerasClassifier. Both classes
    invoke this function to construct a new instance of the model.

    Returns:
        A compiled Keras model.

    """
    model = keras.models.Sequential([
        keras.layers.Dense(64,
                           activation='relu',
                           kernel_regularizer=keras.regularizers.l1_l2(l1, l2)),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(1,
                           kernel_regularizer=keras.regularizers.l1_l2(l1, l2))
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model


def create_member():
    """Create a new population member.

    Each member wraps a Keras model with the same architecture. However, each
    one of them has a different starting value for the model's hyperparameters.

    Returns:
        A new population Member.

    """
    return pbt.members.Member(create_model_fn, STEPS_TO_READY)


def log(step, population):
    """Log the status of the population.

    Log the loss and hyperparameters of each member at a specific step.

    Args:
        step (int): current training step.
        population (List[Member]): population members.

    """
    logger.info('***** Step {:d} *****'.format(step))
    population.sort(key=lambda m: m.loss_history[-1].loss)
    for member in population:
        current_loss = member.loss_history[-1].loss
        logger.info('Member {!s} loss is {:f}'.format(member, current_loss))
        logger.debug('Hyperparameters: {}'.format(
            member.regularizer.get_config()
        ))
    logger.info('*****')


def main():
    # Load dataset. Split training dataset into train and validation sets
    dataset = tf.keras.datasets.boston_housing
    (x_train, y_train), (x_test, y_test) = dataset.load_data()
    x_train, x_val, y_train, y_val = train_test_split(
        x_train, y_train, test_size=TEST_SPLIT)
    print('Train examples: {}, val examples: {}, test examples {}:'.format(
        x_train.shape[0], x_val.shape[0], x_test.shape[0]
    ))

    # ------------------------------------------
    #   GRID SEARCH
    # ------------------------------------------

    print('***** GRID SEARCH *****')

    # Train using both approaches during the same period of time (steps)
    steps_per_epoch = int(x_train.shape[0] / BATCH_SIZE)
    epochs = int(TOTAL_STEPS / steps_per_epoch)
    l1 = np.linspace(1e-5, 1e-2, num=3).tolist()
    l2 = np.linspace(1e-5, 1e-2, num=3).tolist()
    param_grid = dict(l1=l1, l2=l2)
    Result = namedtuple('Result', ['id', 'loss', 'hyperparameters'])
    results = []
    for idx, h in enumerate(ParameterGrid(param_grid)):
        model = create_model_fn(**h)
        model_id = 'Model_{}'.format(idx)
        logger.info('Training {} with hyperparameters {}'.format(model_id, h))
        model.fit(x_train, y_train, batch_size=BATCH_SIZE, epochs=epochs,
                  verbose=0, validation_data=(x_val, y_val), shuffle=False)
        final_loss = model.test_on_batch(x_val, y_val)
        results.append(Result(id=model_id, loss=final_loss, hyperparameters=h))

    print('***** GRID SEARCH RESULTS *****')
    all_losses = []
    print('** Ranking **')
    results.sort(key=lambda result: result.loss)
    for r in results:
        all_losses.append(r.loss)
        print('{} -> loss = {:f}'.format(r.id, r.loss))
    print('** Statistics **')
    print('Mean loss = {:f}, median loss = {:f}, std loss = {:f}'.format(
        np.mean(all_losses), np.median(all_losses), np.std(all_losses)
    ))
    print('** Best hyperparameters **')
    print(results[0].hyperparameters)

    # ------------------------------------------
    #   PBT
    # ------------------------------------------

    # Initial population
    batch_generator = pbt.utils.BatchGenerator(x_train, y_train, BATCH_SIZE)
    population = [create_member() for _ in range(POPULATION_SIZE)]

    print('***** POPULATION BASED TRAINING *****')
    step = 1
    while step < TOTAL_STEPS:
        x, y = batch_generator.next()
        if step % STEPS_TO_READY == 0:
            log(step, population)
        for member in population:
            # One step of optimisation using hyperparameters of 'member'
            member.step_on_batch(x, y)
            # Model evaluation
            loss_before = member.eval_on_batch(x_val, y_val)
            # If optimised for 'STEPS_TO_READY' steps
            if member.ready() and member != population[0]:
                # Use the rest of population to find better solutions
                exploited = member.exploit(population)
                # If new weights != old weights
                if exploited:
                    logger.info(
                        'Finding new hyperparameters for {!s}'.format(member))
                    # Produce new hyperparameters for 'member'
                    member.explore()
                    loss_after = member.eval_on_batch(x_val, y_val)
                    logger.info('Loss before {:f}, loss after {:f}'.format(
                        loss_before, loss_after
                    ))
        step += 1

    print('***** PBT RESULTS *****')
    all_losses = []
    population.sort(key=lambda m: m.loss_history[-1].loss)
    print('** Ranking **')
    for member in population:
        final_loss = member.eval_on_batch(x_val, y_val)
        all_losses.append(final_loss)
        print('Member {!s} -> loss = {:f}'.format(member, final_loss))
    print('** Statistics **')
    print('Mean loss = {:f}, median loss = {:f}, std loss = {:f}'.format(
        np.mean(all_losses), np.median(all_losses), np.std(all_losses)
    ))
    print('** Best hyperparameters **')
    print(population[0].regularizer.get_config())


if __name__ == "__main__":
    main()
