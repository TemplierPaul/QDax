from __future__ import annotations

from functools import partial
from typing import Any, Callable, Optional, Tuple

import jax

from qdax.core.containers.age_repertoire import AgeMapElitesRepertoire
from qdax.core.emitters.emitter import Emitter, EmitterState
from qdax.types import (
    Centroid,
    Descriptor,
    ExtraScores,
    Fitness,
    Genotype,
    Metrics,
    RNGKey,
)

import jax.numpy as jnp
from qdax.core.map_elites import MAPElites

class AgeMAPElites(MAPElites):
    @partial(jax.jit, static_argnames=("self",))
    def init(
        self,
        init_genotypes: Genotype,
        centroids: Centroid,
        random_key: RNGKey,
    ) -> Tuple[AgeMapElitesRepertoire, Optional[EmitterState], RNGKey]:
        """
        Initialize a Map-Elites repertoire with an initial population of genotypes.
        Requires the definition of centroids that can be computed with any method
        such as CVT or Euclidean mapping.

        Args:
            init_genotypes: initial genotypes, pytree in which leaves
                have shape (batch_size, num_features)
            centroids: tesselation centroids of shape (batch_size, num_descriptors)
            random_key: a random key used for stochastic operations.

        Returns:
            An initialized MAP-Elite repertoire with the initial state of the emitter,
            and a random key.
        """
        # score initial genotypes
        fitnesses, descriptors, extra_scores, random_key = self._scoring_function(
            init_genotypes, random_key
        )

        # Make all ages -inf
        ages = jnp.full(fitnesses.shape, -jnp.inf)

        # print("ME init ages shape: ", ages.shape)

        # init the repertoire
        repertoire = AgeMapElitesRepertoire.init(
            genotypes=init_genotypes,
            fitnesses=fitnesses,
            ages=ages,
            descriptors=descriptors,
            centroids=centroids,
            extra_scores=extra_scores,
        )

        # get initial state of the emitter
        emitter_state, random_key = self._emitter.init(
            init_genotypes=init_genotypes, random_key=random_key
        )

        # update emitter state
        emitter_state = self._emitter.state_update(
            emitter_state=emitter_state,
            repertoire=repertoire,
            genotypes=init_genotypes,
            fitnesses=fitnesses,
            descriptors=descriptors,
            extra_scores=extra_scores,
        )

        return repertoire, emitter_state, random_key

    @partial(jax.jit, static_argnames=("self",))
    def update(
        self,
        repertoire: AgeMapElitesRepertoire,
        emitter_state: Optional[EmitterState],
        random_key: RNGKey,
    ) -> Tuple[AgeMapElitesRepertoire, Optional[EmitterState], Metrics, RNGKey]:
        """
        Performs one iteration of the MAP-Elites algorithm.
        1. A batch of genotypes is sampled in the repertoire and the genotypes
            are copied.
        2. The copies are mutated and crossed-over
        3. The obtained offsprings are scored and then added to the repertoire.


        Args:
            repertoire: the MAP-Elites repertoire
            emitter_state: state of the emitter
            random_key: a jax PRNG random key

        Returns:
            the updated MAP-Elites repertoire
            the updated (if needed) emitter state
            metrics about the updated repertoire
            a new jax PRNG key
        """
        # generate offsprings with the emitter
        genotypes, ages, random_key = self._emitter.emit(
            repertoire, emitter_state, random_key
        )

        # print("ME update ages shape: ", ages.shape)

        # scores the offsprings
        fitnesses, descriptors, extra_scores, random_key = self._scoring_function(
            genotypes, random_key
        )

        # add genotypes in the repertoire
        repertoire = repertoire.add(genotypes, descriptors, fitnesses, ages, extra_scores)

        # update emitter state after scoring is made
        emitter_state = self._emitter.state_update(
            emitter_state=emitter_state,
            repertoire=repertoire,
            genotypes=genotypes,
            fitnesses=fitnesses,
            descriptors=descriptors,
            extra_scores=extra_scores,
        )

        # update the metrics
        metrics = self._metrics_function(repertoire)

        return repertoire, emitter_state, metrics, random_key
    