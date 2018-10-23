<h1 align="center">
  <div>
    <img src="https://github.com/fdiazgon/pbt-keras/blob/assets/logo.png?raw=true" alt="project-logo">
  </div>
  pbt-keras
</h1>

<h4 align="center">
Population Based Training of Neural Network implemented in Keras.
</h4>

<p align="center">
  <a href="#getting-started">Getting started</a> •
  <a href="#using-your-own-model">Using your own model</a> •
  <a href="#contributors-wanted">Contributors wanted</a> •
  <a href="#credits">Credits</a> •
  <a href="#license">License</a>
</p>

## Getting started

To run the Boston Housing Dataset example, you need Keras with Tensorflow backend. Check any of the example_boston_ds files.

In this example,  we use two methods to tune the hyperparameters (L1 and L2 regularization for now): a grid search with 6 x 6 = 36 different configurations and a population with the same number of members.

In the following table, you can see a ranking of the models/members according to their final loss (you can take a look at them in `example_boston_ds.ipynb`):

| Rank | Grid Search loss | PBT loss  |
|------|------------------|-----------|
| 1    | 22.026169        | 21.782267 |
| 2    | 22.858138        | 21.782267 |
| ...  | ...              | ...       |
| 18   | 25.088423        | 21.782282 |
| 19   | 25.110988        | 21.782285 |
| ...  | ...              | ...       |
| 35   | 35.687804        | 21.918402 |
| 36   | 35.862466        | 21.955107 |

It is worth mentioning that the grid search was also performed using the PBT algorithm, the only difference is that, for grid search, we avoid exploring new hyperparameters.

Also, we have tried to avoid any randomness to have a comparison as unbiased as possible (e.g., the starting weights are always the same, train/validation splits always use the same seed, etc.). For example, if in the previous example you train the same population twice without the exploring phase (restarting the Keras session before training again), you should see the same result for all the members.

## Using your own model

When creating a new member, you need to supply a function which accepts no parameters and returns a compiled Keras model. However, to allow PBT to change the hyperparameters, you should use special classes when creating the model. For example:

```python
def build_fn(data_dim, l1=1e-5, l2=1e-5):

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
```

You can see how the `kernel_regularizer` is an instance of `pbt.hyperparameters.L1L2Mutable`. For now, this is the only available hyperparameter.

Then, you can create your Member like:

```python
pbt.members.Member(build_fn(data_dim, **h), steps_ready=1e2)
```

Where `**h` is just a dictionary with the hyperparameters (e.g., `{'l1': 1e-5, 'l2': 0}`). A Member exposes an interface to train the underlying model and change its hyperparameters. In the examples you can see how to use this interface to implement the PBT algorithm.

## Contributors wanted

* If you think that something could be done in a more clear and concise way, please tell while the project is small. I will gladly hear.

* Add more hyperparameters, but check how L1L2Mutable is implemented before (or propose a new method to include modifiable hyperparameters once the model is compiled).

* Find a new way to check that the `__call__` method of some hyperparameters is actually called. Right now we are capturing a debug logging call (shame).

## Credits

This work was based on the following paper:

> Jaderberg, M., Dalibard, V., Osindero, S., Czarnecki, W. M., Donahue, J., Razavi, A., ... & Fernando, C. (2017). Population based training of neural networks. arXiv preprint arXiv:1711.09846.

For a brief introduction, check also [DeepMind - Population based training of neural networks](https://deepmind.com/blog/population-based-training-neural-networks/).

Also, thanks to my colleague Giorgio Ruffa for useful discussions.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details
