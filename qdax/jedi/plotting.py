from qdax.utils.plotting import plot_2d_map_elites_repertoire
import matplotlib.pyplot as plt
import jax.numpy as jnp

def net_shape(net):
    return jax.tree_map(lambda x: x.shape, net)

def plot_2d_count(repertoire, min_bd, max_bd, log_scale=True, ax=None):
    # Replace 0 by -inf in count
    count = repertoire.count
    count = jnp.where(count == 0, -jnp.inf, count)
    title = "Number of solutions tried per cell"
    if log_scale:
        # make log 10 scale where not -inf
        count = jnp.where(count != -jnp.inf, jnp.log10(count), count)
        title += " (log10)"
        
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    plot_2d_map_elites_repertoire(
        centroids=repertoire.centroids,
        repertoire_fitnesses=count,
        minval=min_bd,
        maxval=max_bd,
        repertoire_descriptors=repertoire.descriptors,
        ax=ax,
        title=title,
    )
    return ax


def scatter_count(repertoire, log_scale=True, ax=None):
    is_empty = repertoire.fitnesses == -jnp.inf
    # counts
    counts = repertoire.count[~is_empty]
    # fitnesses
    fitnesses = repertoire.fitnesses[~is_empty]

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    ax.scatter(counts, fitnesses, s=1)

    title = "Budget allocation"
    # x log scale
    if log_scale:
        ax.set_xscale("log")
        title += " (log10)"
    ax.set_title(title)
    ax.set_xlabel("Number of solutions tried")
    ax.set_ylabel("Fitness")
    return ax


def add_maze(ax):
    # Outside walls
    ax.plot([-1, -1], [-1, 1], color="k")
    ax.plot([1, 1], [-1, 1], color="k")
    ax.plot([-1, 1], [-1, -1], color="k")
    ax.plot([-1, 1], [1, 1], color="k")

    circle_x = -1 + 0.5
    circle_y = 1 - 0.2
    circle_width = 0.1
    wall_width_ratio = 0.75
    upper_wall_height_offset = 0.2
    lower_wall_height_offset = -0.5

    wall_width = 2 * wall_width_ratio
    # lower wall
    ax.plot(
        [-1 + wall_width/2, 1],
        [lower_wall_height_offset, lower_wall_height_offset],
        color="k",
    )

    # upper wall
    ax.plot(
        [-1, 1 - wall_width/2],
        [upper_wall_height_offset, upper_wall_height_offset],
        color="k",
    )

    circle = plt.Circle((circle_x, circle_y), circle_width/2, color='g', fill=True, label="Target")
    ax.add_artist(circle)
    return ax

def plot_jedi_results(repertoire, logs, title, min_bd, max_bd, log_scale=True):
    n_plots = 4
    fig, ax = plt.subplots(1, 3, figsize=(n_plots*5, 4))
    x_axis = 'frames'
    x_label = 'Environment steps'
    plt.suptitle(title)
    ax[0].plot(logs[x_axis], logs["coverage"])
    ax[0].set_xlabel(x_label)
    ax[0].set_ylabel("Coverage (%)")
    ax[1].plot(logs[x_axis], logs["max_fitness"])
    ax[1].set_xlabel(x_label)
    ax[1].set_ylabel("Max fitness")
    ax[2].plot(logs[x_axis], logs["qd_score"])
    ax[2].set_xlabel(x_label)
    ax[2].set_ylabel("QD score")

    ax[3] = plot_2d_map_elites_repertoire(
            centroids=repertoire.centroids,
            repertoire_fitnesses=repertoire.fitnesses,
            minval=min_bd,
            maxval=max_bd,
            repertoire_descriptors=repertoire.descriptors,
            ax=ax[3],
            title="Final archive",
        )
    
    return fig, ax