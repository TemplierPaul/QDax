from __future__ import annotations

from typing import Optional

import jax.numpy as jnp

from abc import ABC, abstractmethod
from functools import partial
from typing import Any, Optional, Tuple

import jax

from qdax.core.containers.mapelites_repertoire import MapElitesRepertoire
from qdax.core.emitters.cma_emitter import CMAEmitter, CMAEmitterState
from qdax.core.emitters.cma_pool_emitter import CMAPoolEmitter, CMAPoolEmitterState
from qdax.types import Descriptor, ExtraScores, Fitness, Genotype
from qdax.core.cmaes import CMAES, CMAESState

from qdax.types import Centroid, Descriptor, ExtraScores, Fitness, Genotype, RNGKey
from jax.tree_util import tree_flatten, tree_unflatten, tree_map
from qdax.core.emitters.emitter import Emitter, EmitterState
from qdax.core.containers.mapelites_repertoire import (
    MapElitesRepertoire,
    get_cells_indices,
)


def net_shape(net):
    return jax.tree_map(lambda x: x.shape, net)


class CMAMEPolicies(CMAEmitter):
    def __init__(
        self,
        batch_size: int,
        sigma_g: float,
        centroids: Centroid,
        min_count: Optional[int] = None,
        max_count: Optional[float] = None,
    ):
        """
        Class for the emitter of CMA ME from "Covariance Matrix Adaptation for the
        Rapid Illumination of Behavior Space" by Fontaine et al.

        Args:
            batch_size: number of solutions sampled at each iteration
            genotype_dim: dimension of the genotype space.
            centroids: centroids used for the repertoire.
            sigma_g: standard deviation for the coefficients - called step size.
            min_count: minimum number of CMAES opt step before being considered for
                reinitialisation.
            max_count: maximum number of CMAES opt step authorized.
        """
        self._batch_size = batch_size
        self._sigma_g = sigma_g

        # Delay until we have genomes
        self._cmaes = None

        # minimum number of emitted solution before an emitter can be re-initialized
        if min_count is None:
            min_count = 0

        self._min_count = min_count

        if max_count is None:
            max_count = jnp.inf

        self._max_count = max_count

        self._centroids = centroids

        self._cma_initial_state = None

        self.tree_def = None
        self.layer_sizes = None
        self.split_indices = None

    def init(
        self, init_genotypes: Genotype, random_key: RNGKey
    ) -> Tuple[CMAEmitterState, RNGKey]:
        """
        Initializes the CMA-MEGA emitter


        Args:
            init_genotypes: initial genotypes to add to the grid.
            random_key: a random key to handle stochastic operations.

        Returns:
            The initial state of the emitter.
        """
        # Initialisation requires one initial genotype
        if jax.tree_util.tree_leaves(init_genotypes)[0].shape[0] > 1:
            init_genotypes = jax.tree_util.tree_map(
                lambda x: x[0],
                init_genotypes,
            )

        # Add one dimension to the genotype
        init_genotypes = jax.tree_util.tree_map(
            lambda x: jnp.expand_dims(x, axis=0),
            init_genotypes,
        )

        flat_variables, tree_def = tree_flatten(init_genotypes)
        self.layer_shapes = [x.shape[1:] for x in flat_variables]

        sizes = [x.size for x in flat_variables]
        sizes = jnp.array(sizes)

        self.tree_def = tree_def
        self.layer_sizes = sizes.tolist()
        self.split_indices = jnp.cumsum(jnp.array(self.layer_sizes))[:-1].tolist()

        genotype_dim = jnp.sum(sizes)

        # print("Genotype dim", genotype_dim)

        # define a CMAES instance
        self._cmaes = CMAES(
            population_size=self._batch_size,
            search_dim=genotype_dim,
            # no need for fitness function in that specific case
            fitness_function=None,  # type: ignore
            num_best=self._batch_size,
            init_sigma=self._sigma_g,
            mean_init=None,  # will be init at zeros in cmaes
            bias_weights=True,
            delay_eigen_decomposition=True,
        )

        self._cma_initial_state = self._cmaes.init()

        # return the initial state
        random_key, subkey = jax.random.split(random_key)

        num_centroids = self._centroids.shape[0]
        default_fitnesses = -jnp.inf * jnp.ones(shape=num_centroids)

        return (
            CMAEmitterState(
                random_key=subkey,
                cmaes_state=self._cma_initial_state,
                previous_fitnesses=default_fitnesses,
                emit_count=0,
            ),
            random_key,
        )

    @partial(
        jax.jit,
        static_argnames=("self",),
    )
    def flatten(self, network):
        flat_variables, _ = tree_flatten(network)
        # print("Flatten", flat_variables)
        vect = jnp.concatenate([jnp.ravel(x) for x in flat_variables])
        return vect

    @partial(
        jax.jit,
        static_argnames=("self",),
    )
    def unflatten(self, vect):
        """Unflatten a vector of floats into a network"""
        # print("Unflatten", vect.shape)
        split_genome = jnp.split(vect, self.split_indices)
        # Reshape to the original shape
        split_genome = [x.reshape(s) for x, s in zip(split_genome, self.layer_shapes)]

        # Unflatten the tree
        new_net = tree_unflatten(self.tree_def, split_genome)
        return new_net

    @partial(jax.jit, static_argnames=("self",))
    def emit(
        self,
        repertoire: Optional[MapElitesRepertoire],
        emitter_state: CMAEmitterState,
        random_key: RNGKey,
    ) -> Tuple[Genotype, RNGKey]:
        """
        Emits new individuals. Interestingly, this method does not directly modifies
        individuals from the repertoire but sample from a distribution. Hence the
        repertoire is not used in the emit function.

        Args:
            repertoire: a repertoire of genotypes (unused).
            emitter_state: the state of the CMA-MEGA emitter.
            random_key: a random key to handle random operations.

        Returns:
            New genotypes and a new random key.
        """
        # emit from CMA-ES
        offsprings, random_key = self._cmaes.sample(
            cmaes_state=emitter_state.cmaes_state, random_key=random_key
        )
        # print("Emit Offsprings", offsprings.shape)
        # Unflatten
        offsprings = jax.vmap(self.unflatten)(offsprings)

        return offsprings, random_key

    @partial(
        jax.jit,
        static_argnames=("self",),
    )
    def state_update(
        self,
        emitter_state: CMAEmitterState,
        repertoire: MapElitesRepertoire,
        genotypes: Genotype,
        fitnesses: Fitness,
        descriptors: Descriptor,
        extra_scores: Optional[ExtraScores] = None,
    ) -> Optional[EmitterState]:
        """
        Updates the CMA-ME emitter state.

        Note: we use the update_state function from CMAES, a function that assumes
        that the candidates are already sorted. We do this because we have to sort
        them in this function anyway, in order to apply the right weights to the
        terms when update theta.

        Args:
            emitter_state: current emitter state
            repertoire: the current genotypes repertoire
            genotypes: the genotypes of the batch of emitted offspring (unused).
            fitnesses: the fitnesses of the batch of emitted offspring.
            descriptors: the descriptors of the emitted offspring.
            extra_scores: unused

        Returns:
            The updated emitter state.
        """

        genotypes = jax.vmap(self.flatten)(genotypes)

        # retrieve elements from the emitter state
        cmaes_state = emitter_state.cmaes_state

        # Compute the improvements - needed for re-init condition
        indices = get_cells_indices(descriptors, repertoire.centroids)
        improvements = fitnesses - emitter_state.previous_fitnesses[indices]

        ranking_criteria = fitnesses

        # get the indices
        sorted_indices = jnp.flip(jnp.argsort(ranking_criteria))

        # sort the candidates
        sorted_candidates = jax.tree_util.tree_map(
            lambda x: x[sorted_indices], genotypes
        )
        sorted_improvements = improvements[sorted_indices]

        # compute reinitialize condition
        emit_count = emitter_state.emit_count + 1

        # check if the criteria are too similar
        sorted_criteria = ranking_criteria[sorted_indices]
        flat_criteria_condition = (
            jnp.linalg.norm(sorted_criteria[0] - sorted_criteria[-1]) < 1e-12
        )

        # check all conditions
        reinitialize = (
            jnp.all(improvements < 0) * (emit_count > self._min_count)
            + (emit_count > self._max_count)
            + self._cmaes.stop_condition(cmaes_state)
            + flat_criteria_condition
        )

        # If true, draw randomly and re-initialize parameters
        def update_and_reinit(
            operand: Tuple[
                CMAESState, CMAEmitterState, MapElitesRepertoire, int, RNGKey
            ],
        ) -> Tuple[CMAEmitterState, RNGKey]:
            return self._update_and_init_emitter_state(*operand)

        def update_wo_reinit(
            operand: Tuple[
                CMAESState, CMAEmitterState, MapElitesRepertoire, int, RNGKey
            ],
        ) -> Tuple[CMAEmitterState, RNGKey]:
            """Update the emitter when no reinit event happened.

            Here lies a divergence compared to the original implementation. We
            are getting better results when using no mask and doing the update
            with the whole batch of individuals rather than keeping only the one
            than were added to the archive.

            Interestingly, keeping the best half was not doing better. We think that
            this might be due to the small batch size used.

            This applies for the setting from the paper CMA-ME. Those facts might
            not be true with other problems and hyperparameters.

            To replicate the code described in the paper, replace:
            `mask = jnp.ones_like(sorted_improvements)`

            by:
            ```
            mask = sorted_improvements >= 0
            mask = mask + 1e-6
            ```

            RMQ: the addition of 1e-6 is here to fix a numerical
            instability.
            """

            (cmaes_state, emitter_state, repertoire, emit_count, random_key) = operand

            # Update CMA Parameters
            mask = jnp.ones_like(sorted_improvements)

            cmaes_state = self._cmaes.update_state_with_mask(
                cmaes_state, sorted_candidates, mask=mask
            )

            emitter_state = emitter_state.replace(
                cmaes_state=cmaes_state,
                emit_count=emit_count,
            )

            return emitter_state, random_key

        # Update CMA Parameters
        emitter_state, random_key = jax.lax.cond(
            reinitialize,
            update_and_reinit,
            update_wo_reinit,
            operand=(
                cmaes_state,
                emitter_state,
                repertoire,
                emit_count,
                emitter_state.random_key,
            ),
        )

        # update the emitter state
        emitter_state = emitter_state.replace(
            random_key=random_key, previous_fitnesses=repertoire.fitnesses
        )

        return emitter_state

    def _update_and_init_emitter_state(
        self,
        cmaes_state: CMAESState,
        emitter_state: CMAEmitterState,
        repertoire: MapElitesRepertoire,
        emit_count: int,
        random_key: RNGKey,
    ) -> Tuple[CMAEmitterState, RNGKey]:
        """Update the emitter state in the case of a reinit event.
        Reinit the cmaes state and use an individual from the repertoire
        as the starting mean.

        Args:
            cmaes_state: current cmaes state
            emitter_state: current cmame state
            repertoire: most recent repertoire
            emit_count: counter of the emitter
            random_key: key to handle stochastic events

        Returns:
            The updated emitter state.
        """

        # re-sample
        random_genotype, random_key = repertoire.sample(random_key, 1)

        new_mean = self.flatten(random_genotype)

        # remove the batch dim
        # new_mean = jax.tree_util.tree_map(lambda x: x.squeeze(0), random_genotype)

        cmaes_init_state = self._cma_initial_state.replace(mean=new_mean, num_updates=0)

        emitter_state = emitter_state.replace(
            cmaes_state=cmaes_init_state, emit_count=0
        )

        return emitter_state, random_key

    def _ranking_criteria(
        self,
        emitter_state: CMAEmitterState,
        repertoire: MapElitesRepertoire,
        genotypes: Genotype,
        fitnesses: Fitness,
        descriptors: Descriptor,
        extra_scores: Optional[ExtraScores],
        improvements: jnp.ndarray,
    ) -> jnp.ndarray:
        return fitnesses


class CMAPoolPolicies(CMAPoolEmitter):
    def init(
        self, init_genotypes: Genotype, random_key: RNGKey
    ) -> Tuple[CMAPoolEmitterState, RNGKey]:
        """
        Initializes the CMA-MEGA emitter


        Args:
            init_genotypes: initial genotypes to add to the grid.
            random_key: a random key to handle stochastic operations.

        Returns:
            The initial state of the emitter.
        """

        # def scan_emitter_init(
        #     carry: RNGKey, unused: Any
        # ) -> Tuple[RNGKey, CMAEmitterState]:
        #     random_key = carry
        #     emitter_state, random_key = self._emitter.init(init_genotypes, random_key)
        #     return random_key, emitter_state

        # init all the emitter states
        print(f"Creating {self._num_states} CMA emitters")
        emitter_states = []
        for _ in range(self._num_states):
            emitter_state, random_key = self._emitter.init(init_genotypes, random_key)
            emitter_states.append(emitter_state)

        emitter_states = tree_map(lambda *args: jnp.stack(args), *emitter_states)

        # define the emitter state of the pool
        emitter_state = CMAPoolEmitterState(
            current_index=0, emitter_states=emitter_states
        )

        return (
            emitter_state,
            random_key,
        )
