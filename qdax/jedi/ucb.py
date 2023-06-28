import jax.numpy as jnp

def gp_ucb(repertoire, opt_posterior, train_data, beta, bounds, n_points=100, plot=False):
    points = repertoire.centroids.astype(jnp.float64)
    # rng = jax.random.PRNGKey(0)

    # Evaluate the points with the GP
    latent_dist = opt_posterior.predict(points, train_data=train_data)
    predictive_dist = opt_posterior.likelihood(latent_dist)

    predictive_mean = predictive_dist.mean()
    # Normalize with mean
    predictive_mean = predictive_mean - predictive_mean.mean()
    predictive_std = predictive_dist.stddev()

    ucb = predictive_mean + jnp.sqrt(beta) * predictive_std
    # top n_points indices
    argmax_ucb = jnp.argsort(ucb)[-n_points:]
    selected_points = points[argmax_ucb].astype(jnp.float32)

    return selected_points