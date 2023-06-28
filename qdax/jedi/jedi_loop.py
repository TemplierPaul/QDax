import jax 
import jax.numpy as jnp
from jax.tree_util import tree_flatten, tree_unflatten, tree_map
from qdax.jedi.evolution_strategies.aria_es import MixedES

def closest_genotype(repertoire, target_bd):
    is_empty = repertoire.fitnesses == -jnp.inf
    useful_centroids = repertoire.centroids[~is_empty]
    indices = jnp.arange(len(repertoire.centroids))[~is_empty]

    # get closes centroid to target_bd
    distances = jnp.linalg.norm(useful_centroids - target_bd, axis=1)
    cent_idx = distances.argmin()
    start_bd = useful_centroids[cent_idx]
    start_genome = tree_map(lambda x: x[indices[cent_idx]], repertoire.genotypes)

    return start_genome, start_bd

def stochastic_closest(repertoire, target_bd, top_n=5):
    raise NotImplementedError("TODO")

def aim_for(target, repertoire, es, config, base_es_emitter_state, random_key):
    target = jnp.array(target).astype(jnp.float32)
    # Select closest genotype in repertoire
    start, start_bd = closest_genotype(repertoire, target)
    # add a dimension
    start = jax.tree_map(
        lambda x: x[None, ...],
        start,
    )
    # print(net_shape(start))

    # Run ES
    if isinstance(es, MixedES):
        final_offspring, metrics, es_emitter_state, random_key = es.step(
            repertoire,
            start,
            target,
            base_es_emitter_state,
            random_key,
            n_steps=config["es_gens"],
        )
    else:
        final_offspring, metrics, es_emitter_state, random_key = es.step(
            start,
            target,
            base_es_emitter_state,
            random_key,
            n_steps=config["es_gens"],
        )
    return metrics, es_emitter_state, random_key