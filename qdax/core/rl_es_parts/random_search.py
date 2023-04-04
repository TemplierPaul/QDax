from __future__ import annotations
from qdax.core.emitters.vanilla_es_emitter import VanillaESConfig, VanillaESEmitterState, VanillaESEmitter, NoveltyArchive
from qdax.core.rl_es_parts.es_utils import ESRepertoire, ESMetrics

from dataclasses import dataclass
from functools import partial
from typing import Callable, Tuple

import jax
import jax.numpy as jnp
import optax
from qdax.core.emitters.emitter import EmitterState

from qdax.types import Descriptor, ExtraScores, Fitness, Genotype, RNGKey

@dataclass
class RandomConfig(VanillaESConfig):
    """Configuration for the random search emitter.

    Args:
        nses_emitter: if True, use NSES, if False, use ES
        sample_number: num of samples for gradient estimate
        sample_sigma: std to sample the samples for gradient estimate
    """

    nses_emitter: bool = False
    sample_number: int = 1000
    sample_sigma: float = 0.02

class RandomEmitterState(VanillaESEmitterState):
    """State for the random search emitter."""

    offspring: Genotype
    generation_count: int
    novelty_archive: NoveltyArchive
    random_key: RNGKey
    optimizer_state: optax.OptState = None # Not used by random search
    metrics: ESMetrics = ESMetrics()

class RandomEmitter(VanillaESEmitter):
    '''Random search emitter.'''

    @partial(
        jax.jit,
        static_argnames=("self", "scores_fn"),
    )
    def _es_emitter(
        self,
        parent: Genotype,
        optimizer_state: optax.OptState,
        random_key: RNGKey,
        scores_fn: Callable[[Fitness, Descriptor], jnp.ndarray],
        actor: Genotype=None,
    ) -> Tuple[Genotype, optax.OptState, RNGKey]:
        """Main es component, given a parent and a way to infer the score from
        the fitnesses and descriptors fo its es-samples, return its
        approximated-gradient-generated offspring.

        Args:
            parent: the considered parent.
            scores_fn: a function to infer the score of its es-samples from
                their fitness and descriptors.
            random_key

        Returns:
            The approximated-gradients-generated offspring and a new random_key.
        """

        random_key, subkey = jax.random.split(random_key)

        # Sampling mirror noise
        sample_number = self._config.sample_number if not self._config.actor_injection else self._config.sample_number - 1

        # Sampling noise
        sample_number = sample_number 
        sample_noise = jax.tree_map(
            lambda x: jax.random.normal(
                key=subkey,
                shape=jnp.repeat(x, sample_number, axis=0).shape,
            ),
            parent,
        )

        # Applying noise
        samples = jax.tree_map(
            lambda x: jnp.repeat(x, sample_number, axis=0),
            parent,
        )
        samples = jax.tree_map(
            lambda mean, noise: mean + self._config.sample_sigma * noise,
            samples,
            sample_noise,
        )

        # Evaluating samples
        fitnesses, descriptors, extra_scores, random_key = self._scoring_fn(
            samples, random_key
        )

        extra_scores["population_fitness"] = fitnesses

        # Get the one with highest fitness
        best_index = jnp.argmax(fitnesses)
        
        # Get the best sample
        offspring = jax.tree_map(
            lambda x: x[best_index],
            samples,
        )

        return offspring, optimizer_state, random_key, extra_scores