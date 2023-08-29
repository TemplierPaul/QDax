from qdax.jedi.evolution_strategies.base_es import BaseES

import jax
import jax.numpy as jnp


class ARIA_ES(BaseES):
    @property
    def name(self):
        return "ARIA_ES"

    def step(
        self, repertoire, start, target_bd, es_emitter_state, random_key, n_steps=1
    ):
        (
            centroids,
            offspring,
            target_bd,
            es_emitter_state,
            random_key,
        ), metrics = jax.lax.scan(
            self.scan_step,
            (repertoire.centroids, start, target_bd, es_emitter_state, random_key),
            None,
            length=n_steps,
        )
        # Remove the batch dimension
        metrics = jax.tree_map(lambda x: x.squeeze(), metrics)
        return offspring, metrics, es_emitter_state, random_key

    def scan_step(self, carry, unused):
        centroids, start, target_bd, es_emitter_state, random_key = carry

        @jax.jit
        def bd_scores_fn(fitnesses, descriptors) -> jnp.ndarray:
            # Get index of target_bd in repertoire
            target_idx = jnp.argmin(jnp.linalg.norm(centroids - target_bd, axis=-1))
            # Compute distance from descriptors to repertoire
            centroids_dists = jnp.linalg.norm(
                descriptors[:, None, :] - centroids[None, :, :], axis=-1
            )
            # Get argmin for each descriptor
            bd_index = jnp.argmin(centroids_dists, axis=-1)
            # Compute distance of each descriptor to target_bd
            dists = jnp.linalg.norm(descriptors - target_bd, axis=-1)
            fmin = jnp.min(fitnesses)
            # Score: fitnesses if argmin is target_idx, distance to target_bd otherwise
            scores = jnp.where(bd_index == target_idx, fitnesses, fmin - dists)
            return scores

        offspring, optimizer_state, random_key, extra_scores = self.emitter._es_emitter(
            parent=start,
            random_key=random_key,
            scores_fn=bd_scores_fn,
            optimizer_state=es_emitter_state.optimizer_state,
        )
        # evaluate offspring
        (
            offspring_fitnesses,
            offspring_descriptors,
            offspring_extra_scores,
            random_key,
        ) = self.emitter._rollout_fn(offspring, random_key)

        metrics = {
            "genotype": offspring,
            "fitness": offspring_fitnesses,
            "descriptor": offspring_descriptors,
            "population_fitness": extra_scores["population_fitness"],
            "population_descriptors": extra_scores["population_descriptors"],
            "population_networks": extra_scores["population_networks"],
        }

        es_emitter_state = es_emitter_state.replace(optimizer_state=optimizer_state)

        return (centroids, offspring, target_bd, es_emitter_state, random_key), metrics
