import jax 
import jax.numpy as jnp

import matplotlib.pyplot as plt

# Pareto front
def pareto_front(points, f1, f2):
    def is_dominated(x):
        """Check if a point is dominated."""
        return jnp.any((f1 > x[0]) & (f2 > x[1]))
    # zip f1 and f2
    f = jnp.vstack([f1, f2]).T
    dominated = jax.vmap(is_dominated)(f)
    return points[~dominated]

def get_pareto(points, opt_posterior, train_data, bounds, n_points=100, plot=False, return_front=False):
    points = points.astype(jnp.float64)
    rng = jax.random.PRNGKey(0)
    # minval = jnp.array([bounds[0][0], bounds[1][0]])
    # maxval = jnp.array([bounds[0][1], bounds[1][1]])

    # Evaluate the points with the GP
    latent_dist = opt_posterior.predict(points, train_data=train_data)
    predictive_dist = opt_posterior.likelihood(latent_dist)

    predictive_mean = predictive_dist.mean()
    predictive_std = predictive_dist.stddev()

    if plot:
        fig, ax = plt.subplots(figsize=(5, 5))
        # scatter mean vs std
        plt.scatter(predictive_mean, predictive_std, color="b", s=5, label="Points")
        plt.xlabel("Mean")
        plt.ylabel("Std")
        plt.title("Mean vs Std")

    # Pareto front
    indices = jnp.arange(len(points))
    pf_indices = pareto_front(indices, predictive_mean, predictive_std)

    # Sample from the Pareto front
    if len(pf_indices) > n_points:
        selected_indices = jax.random.choice(rng, pf_indices, shape=(n_points,), replace=False)
    elif len(pf_indices) == n_points:
        selected_indices = pf_indices
    else:
        # Complete with random other points
        n_random = n_points - len(pf_indices)
        # randomly select indices
        random_indices = jax.random.choice(rng, indices, shape=(n_random,), replace=False)
        print(pf_indices.shape, random_indices.shape)
        # concatenate 
        selected_indices = jnp.vstack([pf_indices.reshape(-1, 1), random_indices.reshape(-1, 1)]).reshape(-1)
        # selected_indices = jnp.vstack([pf_indices, random_indices])
        print(selected_indices.shape)
        
    if plot:
        # plot pareto front
        pareto_mean = predictive_mean[pf_indices]
        pareto_std = predictive_std[pf_indices]
        ax.scatter(pareto_mean, pareto_std, color="g", s=5, label="Pareto front")
        selected_mean = predictive_mean[selected_indices]
        selected_std = predictive_std[selected_indices]
        ax.scatter(selected_mean, selected_std, color="r", s=5, label="Selected points")
        ax.legend()

    if return_front:
        # print("Returning pareto front")
        if len(pf_indices) < n_points:
            # Complete with random other points
            n_random = n_points - len(pf_indices)
            # randomly select indices
            random_indices = jax.random.choice(rng, indices, shape=(n_random,), replace=False)
            print(pf_indices.shape, random_indices.shape)
            # concatenate 
            selected_indices = jnp.vstack([pf_indices.reshape(-1, 1), random_indices.reshape(-1, 1)]).reshape(-1)
        else:
            selected_indices = pf_indices
        return points[selected_indices]

    pf_points = points[selected_indices].astype(jnp.float32)
    return pf_points
    
def random_pareto(opt_posterior, train_data, bounds, n_points=100, n_samples=1000, plot=False, return_front=False):
    rng = jax.random.PRNGKey(0)
    minval = jnp.array([bounds[0][0], bounds[1][0]])
    maxval = jnp.array([bounds[0][1], bounds[1][1]])
    points = jax.random.uniform(rng, (n_samples, 2), minval=minval, maxval=maxval)
    return get_pareto(points, opt_posterior, train_data, bounds, n_points=n_points, plot=plot, return_front=return_front)

def centroids_pareto(repertoire, opt_posterior, train_data, bounds, n_points=100, plot=False, return_front=False):
    return get_pareto(repertoire.centroids, opt_posterior, train_data, bounds, n_points=n_points, plot=plot, return_front=return_front)
        
def archive_pareto(repertoire, opt_posterior, train_data, bounds, n_points=100, plot=False, return_front=False):
    is_empty = repertoire.fitnesses == -jnp.inf
    archive = repertoire.centroids[~is_empty]
    return get_pareto(archive, opt_posterior, train_data, bounds, n_points=n_points, plot=plot, return_front=return_front)

# Compute pair-wise distances
def dist(x1, x2):
    return jnp.sqrt(jnp.sum((x1 - x2) ** 2))

def dist_matrix(points):
    # Compute the distance matrix using vmap
    return jax.vmap(lambda x: jax.vmap(lambda y: dist(x, y))(points))(points)

# Crowded wrapper
def crowded(func):
    def wrapped(*args, **kwargs):
        # print("Wow it's crowded in here")
        kwargs["return_front"]=True
        front = func(*args, **kwargs)
        # shuffle front
        rng = jax.random.PRNGKey(0)
        shuffled = jax.random.permutation(rng, front)
        # Compute the distance matrix
        d = dist_matrix(shuffled)
        # Only get the upper triangular part
        d = jnp.triu(d, k=1)
        # Replace 0 with inf
        d = jnp.where(d == 0, jnp.inf, d)
        # Min per line
        d = jnp.min(d, axis=1)
        # print(d)
        # Get top 10 points
        top_indices = jnp.argsort(d)[:kwargs["n_points"]]
        top_points = front[top_indices]
        return top_points
    return wrapped

crowded_centroids_pareto = crowded(centroids_pareto)
crowded_archive_pareto = crowded(archive_pareto)