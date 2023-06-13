from typing import Any, Dict, Iterable, List, Optional, Tuple

import jax.numpy as jnp
import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
from matplotlib.figure import Figure
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.spatial import Voronoi

from qdax.core.containers.age_repertoire import AgeMapElitesRepertoire

from qdax.utils.plotting import get_voronoi_finite_polygons_2d, plot_2d_map_elites_repertoire
from dataclasses import dataclass, asdict
from flax.struct import dataclass as fdataclass
from qdax.types import (
    Action,
    Mask,
    Observation,
    Params,
    Reward,
    Centroid,
    Descriptor,
    ExtraScores,
    Fitness,
    Genotype,
    RNGKey,
    Metrics,
)

### Logging

@fdataclass
class AgeMetrics:
    # Performance metrics
    evaluations: int = 0
    max_fitness: float = -np.inf
    coverage: float = 0.0
    qd_score: float = 0.0
    actor_fitness: Fitness = -jnp.inf

    # Age metrics
    from_actor_ratio: float = 0.0
    average_age: float = 0.0

def age_metrics(
    repertoire: AgeMapElitesRepertoire, 
    qd_offset: float
) -> Metrics:
    """Compute the usual QD metrics that one can retrieve
    from a MAP Elites repertoire.

    Args:
        repertoire: a MAP-Elites repertoire
        qd_offset: an offset used to ensure that the QD score
            will be positive and increasing with the number
            of individuals.

    Returns:
        a dictionary containing the QD score (sum of fitnesses
            modified to be all positive), the max fitness of the
            repertoire, the coverage (number of niche filled in
            the repertoire).
    """

    # QD metrics from archive
    archive_metrics = {}
    repertoire_empty = repertoire.fitnesses == -jnp.inf
    archive_metrics["qd_score"] = jnp.sum(repertoire.fitnesses, where=~repertoire_empty)
    archive_metrics["qd_score"] += qd_offset * jnp.sum(1.0 - repertoire_empty)
    archive_metrics["coverage"] = 100 * jnp.mean(1.0 - repertoire_empty)
    archive_metrics["max_fitness"] = jnp.max(repertoire.fitnesses)

    # ES metrics
    metrics = emitter_state.metrics
    # Turn into a dict
    metrics = metrics.__dict__.copy()

    # Merge
    metrics.update(archive_metrics)
    return metrics

### Plotting functions

def plot_2d_map_elites_origin(
    repertoire: AgeMapElitesRepertoire,
    minval: jnp.ndarray,
    maxval: jnp.ndarray,
    ax: Optional[plt.Axes] = None,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
) -> Tuple[Optional[Figure], Axes]:
    """Plot a visual representation of a 2d map elites repertoire.

    TODO: Use repertoire as input directly. Because this
    function is very specific to repertoires.

    Args:
        centroids: the centroids of the repertoire
        repertoire_fitnesses: the fitness of the repertoire
        minval: minimum values for the descritors
        maxval: maximum values for the descriptors
        repertoire_descriptors: the descriptors. Defaults to None.
        ax: a matplotlib axe for the figure to plot. Defaults to None.
        vmin: minimum value for the fitness. Defaults to None. If not given,
            the value will be set to the minimum fitness in the repertoire.
        vmax: maximum value for the fitness. Defaults to None. If not given,
            the value will be set to the maximum fitness in the repertoire.

    Raises:
        NotImplementedError: does not work for descriptors dimension different
        from 2.

    Returns:
        A figure and axes object, corresponding to the visualisation of the
        repertoire.
    """

    repertoire_fitnesses = repertoire.fitnesses
    repertoire_descriptors = repertoire.descriptors
    repertoire_ages = repertoire.ages
    centroids = repertoire.centroids

    grid_empty = repertoire_fitnesses == -jnp.inf
    num_descriptors = centroids.shape[1]
    if num_descriptors != 2:
        raise NotImplementedError("Grid plot supports 2 descriptors only for now.")

    my_cmap = cm.viridis

    fitnesses = repertoire_fitnesses
    ages = repertoire_ages
    if vmin is None:
        vmin = float(jnp.min(fitnesses[~grid_empty]))
    if vmax is None:
        vmax = float(jnp.max(fitnesses[~grid_empty]))

    # set the parameters
    font_size = 12
    params = {
        "axes.labelsize": font_size,
        "legend.fontsize": font_size,
        "xtick.labelsize": font_size,
        "ytick.labelsize": font_size,
        "text.usetex": False,
        "figure.figsize": [10, 10],
    }

    mpl.rcParams.update(params)

    # create the plot object
    fig = None
    if ax is None:
        fig, ax = plt.subplots(facecolor="white", edgecolor="white")

    assert (
        len(np.array(minval).shape) < 2
    ), f"minval : {minval} should be float or couple of floats"
    assert (
        len(np.array(maxval).shape) < 2
    ), f"maxval : {maxval} should be float or couple of floats"

    if len(np.array(minval).shape) == 0 and len(np.array(maxval).shape) == 0:
        ax.set_xlim(minval, maxval)
        ax.set_ylim(minval, maxval)
    else:
        ax.set_xlim(minval[0], maxval[0])
        ax.set_ylim(minval[1], maxval[1])

    ax.set(adjustable="box", aspect="equal")

    # create the regions and vertices from centroids
    regions, vertices = get_voronoi_finite_polygons_2d(centroids)

    norm = Normalize(vmin=vmin, vmax=vmax)

    # fill the plot with contours
    for region in regions:
        polygon = vertices[region]
        ax.fill(*zip(*polygon), alpha=0.05, edgecolor="black", facecolor="white", lw=1)

    from_actor_color = "red"
    from_init_color = "blue"
    # Add legend
    ax.scatter([], [], c=from_actor_color, label="From Actor")
    ax.scatter([], [], c=from_init_color, label="From Init")
    ax.legend(loc="upper left")

    # fill the plot with the colors
    for idx, fitness in enumerate(fitnesses):
        if fitness > -jnp.inf:
            age = ages[idx]
            if age >= 0:
                color = from_actor_color
            else:
                color = from_init_color
            
            region = regions[idx]
            polygon = vertices[region]

            ax.fill(*zip(*polygon), alpha=0.8, color=color)

    # if descriptors are specified, add points location
    if repertoire_descriptors is not None:
        descriptors = repertoire_descriptors[~grid_empty]
        ax.scatter(
            descriptors[:, 0],
            descriptors[:, 1],
            c=fitnesses[~grid_empty],
            cmap=my_cmap,
            s=10,
            zorder=0,
        )

    # aesthetic
    ax.set_xlabel("Behavior Dimension 1")
    ax.set_ylabel("Behavior Dimension 2")

    ax.set_title("Origin")
    ax.set_aspect("equal")
    return fig, ax

def plot_map_elites_age(
    env_steps: jnp.ndarray,
    metrics: Dict,
    repertoire: AgeMapElitesRepertoire,
    min_bd: jnp.ndarray,
    max_bd: jnp.ndarray,
    title: str = "Injection in Map-Elites",
) -> Tuple[Optional[Figure], Axes]:
    """Plots three usual QD metrics, namely the coverage, the maximum fitness
    and the QD-score, along the number of environment steps. This function also
    plots a visualisation of the final map elites grid obtained. It ensures that
    those plots are aligned together to give a simple and efficient visualisation
    of an optimization process.

    Args:
        env_steps: the array containing the number of steps done in the environment.
        metrics: a dictionary containing metrics from the optimizatoin process.
        repertoire: the final repertoire obtained.
        min_bd: the mimimal possible values for the bd.
        max_bd: the maximal possible values for the bd.

    Returns:
        A figure and axes with the plots of the metrics and visualisation of the grid.
    """
    # Customize matplotlib params
    font_size = 16
    params = {
        "axes.labelsize": font_size,
        "axes.titlesize": font_size,
        "legend.fontsize": font_size,
        "xtick.labelsize": font_size,
        "ytick.labelsize": font_size,
        "text.usetex": False,
        "axes.titlepad": 10,
    }

    mpl.rcParams.update(params)

    # Visualize the training evolution and final repertoire
    fig, axes = plt.subplots(nrows=2, ncols=5, figsize=(50, 20))

    axes[0, 0].plot(env_steps, metrics["coverage"])
    axes[0, 0].set_xlabel("Environment steps")
    axes[0, 0].set_ylabel("Coverage in %")
    axes[0, 0].set_title("Coverage evolution during training")
    axes[0, 0].set_aspect(0.95 / axes[0, 0].get_data_ratio(), adjustable="box")

    axes[0, 1].plot(env_steps, metrics["max_fitness"])
    axes[0, 1].set_xlabel("Environment steps")
    axes[0, 1].set_ylabel("Maximum fitness")
    axes[0, 1].set_title("Maximum fitness evolution during training")
    axes[0, 1].set_aspect(0.95 / axes[0, 1].get_data_ratio(), adjustable="box")

    axes[0, 2].plot(env_steps, metrics["qd_score"])
    axes[0, 2].set_xlabel("Environment steps")
    axes[0, 2].set_ylabel("QD Score")
    axes[0, 2].set_title("QD Score evolution during training")
    axes[0, 2].set_aspect(0.95 / axes[0, 2].get_data_ratio(), adjustable="box")

    _, axes[0, 3] = plot_2d_map_elites_repertoire(
        centroids=repertoire.centroids,
        repertoire_fitnesses=repertoire.fitnesses,
        minval=min_bd,
        maxval=max_bd,
        repertoire_descriptors=repertoire.descriptors,
        ax=axes[0, 3],
    )

    axes[0, 4].set_aspect(0.95 / axes[0, 4].get_data_ratio(), adjustable="box")

    # env_steps = jnp.arange(num_iterations) * episode_length * batch_size

    axes[1, 0].plot(env_steps, metrics["from_actor_ratio"])
    dy = 0.01
    axes[1, 0].set_ylim(-dy, 1 + dy)
    axes[1, 0].set_xlabel("Environment steps")
    axes[1, 0].set_ylabel("Actor ratio")
    axes[1, 0].set_title("Ratio of actor descendants in the archive")
    axes[1, 0].set_aspect(0.95 / axes[1, 0].get_data_ratio(), adjustable="box")

    axes[1, 1].plot(env_steps, metrics["average_age"])
    axes[1, 1].set_xlabel("Environment steps")
    axes[1, 1].set_ylabel("Average age")
    axes[1, 1].set_title("Average age of actor descendants in the archive")
    axes[1, 1].set_aspect(0.95 / axes[1, 1].get_data_ratio(), adjustable="box")

    _, axes[1, 2] = plot_2d_map_elites_origin(
        repertoire=repertoire,
        minval=min_bd,
        maxval=max_bd,
        ax=axes[1, 2],
    )

    has_actors = metrics["from_actor_ratio"][-1] > 0

    if has_actors:
        _, axes[1, 3] = plot_2d_map_elites_repertoire(
            centroids=repertoire.centroids,
            repertoire_fitnesses=repertoire.ages,
            minval=min_bd,
            maxval=max_bd,
            repertoire_descriptors=repertoire.descriptors,
            ax=axes[1, 3],
            title="Age of the archive",
        )

        # Scatter age vs fitness
        age = repertoire.ages
        fitness = repertoire.fitnesses
        # Filter: keep where age is not -inf
        f = age != -jnp.inf
        age = age[f]
        fitness = fitness[f]
        axes[1, 4].scatter(age, fitness)
        axes[1, 4].set_xlabel("Age")
        axes[1, 4].set_ylabel("Fitness")
        axes[1, 4].set_title("Fitness vs age")
        axes[1, 4].set_aspect(0.95 / axes[1, 4].get_data_ratio(), adjustable="box")

    # global title
    fig.suptitle(title, fontsize=font_size + 10)

    return fig, axes