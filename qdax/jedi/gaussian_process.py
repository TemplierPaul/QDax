# Summarize this section into a function that takes a repertoire and returns a GP
# Add a function for 2d plotting
import jax 
import jax.numpy as jnp
from jax import jit
import optax as ox
import gpjax as gpx

import matplotlib.pyplot as plt

from qdax.jedi.plotting import add_maze

# import partial
from functools import partial
from mpl_toolkits.axes_grid1 import make_axes_locatable

def fit_GP(repertoire, num_iters=1000):
    """Fit a GP to a repertoire."""
    random_key = jax.random.PRNGKey(0)
    # Remove empty points
    is_empty = repertoire.fitnesses == -jnp.inf
    bd = repertoire.centroids[~is_empty]
    fitness = repertoire.fitnesses[~is_empty]

    D = gpx.Dataset(X=bd, y=fitness.reshape(-1, 1))

    # Prior
    # prior_constant = jnp.max(fitness)
    prior_constant = jnp.mean(fitness)

    kernel = gpx.kernels.RBF()
    meanf = gpx.mean_functions.Constant(prior_constant)
    prior = gpx.Prior(mean_function=meanf, kernel=kernel)

    # Posterior
    likelihood = gpx.Gaussian(num_datapoints=D.n)

    posterior = prior * likelihood

    # Parameter state
    negative_mll = gpx.objectives.ConjugateMLL(negative=True)
    negative_mll(posterior, train_data=D)

    negative_mll = jit(negative_mll)

    # Optimization

    opt_posterior, history = gpx.fit(
        model=posterior,
        objective=negative_mll,
        train_data=D,
        optim=ox.adam(learning_rate=0.01),
        num_iters=num_iters,
        safe=True,
        key=random_key,
    )

    return opt_posterior, D


def plot_GP(repertoire, opt_posterior, grid_size=30, plot_std=False, larger=True, maze=False, min_bd=-1, max_bd=1):
    """Plot a GP on a grid."""
    # Remove empty points
    is_empty = repertoire.fitnesses == -jnp.inf
    bd = repertoire.centroids[~is_empty]
    fitness = repertoire.fitnesses[~is_empty]

    D = gpx.Dataset(X=bd, y=fitness.reshape(-1, 1))

    # plot contour on larger grid
    min_bd = min_bd - 0.5 if larger else min_bd
    max_bd = max_bd + 0.5 if larger else max_bd
    x1 = jnp.linspace(min_bd, max_bd, grid_size)
    x2 = jnp.linspace(min_bd, max_bd, grid_size)
    X1, X2 = jnp.meshgrid(x1, x2)
    X_grid = jnp.vstack([X1.ravel(), X2.ravel()]).T

    latent_dist = opt_posterior.predict(X_grid, train_data=D)
    predictive_dist = opt_posterior.likelihood(latent_dist)

    predictive_mean = predictive_dist.mean()
    predictive_std = predictive_dist.stddev()

    # plot contour
    fig, ax = plt.subplots(figsize=(5, 5))
    im = plt.contourf(X1, X2, predictive_mean.reshape(grid_size, grid_size), cmap="coolwarm")
    plt.scatter(bd[:, 0], bd[:, 1], c=fitness, cmap="viridis", s=1)
    if maze:
        ax = add_maze(ax)
    
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label('Fitness')

    # plt.title("Mean")

    if plot_std:
        # plot contour
        fig, std_ax = plt.subplots(figsize=(5, 5))
        plt.contourf(X1, X2, predictive_std.reshape(grid_size, grid_size), cmap="coolwarm")
        plt.colorbar()
        plt.scatter(bd[:, 0], bd[:, 1], c=fitness, cmap="viridis", s=1)
        if maze:
            std_ax = add_maze(std_ax)
        plt.title("Std")
        ax = [ax, std_ax]
    
    return ax

