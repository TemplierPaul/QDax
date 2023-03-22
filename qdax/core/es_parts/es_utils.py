from __future__ import annotations
from typing import Callable, List, Optional, Tuple, Union
import jax
import jax.numpy as jnp
from functools import partial

from qdax.types import Centroid, Descriptor, ExtraScores, Fitness, Genotype, RNGKey

# from qdax.core.containers.mapelites_repertoire import compute_cvt_centroids
# from qdax.core.emitters.vanilla_es_emitter import VanillaESConfig, VanillaESEmitter
from qdax.core.map_elites import MAPElites
from qdax.core.containers.mapelites_repertoire import MapElitesRepertoire, get_cells_indices
# from qdax.utils.metrics import CSVLogger, default_qd_metrics
from qdax.core.emitters.emitter import Emitter, EmitterState

class ESRepertoire(MapElitesRepertoire):
    """ A MapElitesRepertoire for ES that keeps the fitness of the last added ES center for logging
    """
    last_fitness: float

    @jax.jit
    def add(
        self,
        batch_of_genotypes: Genotype,
        batch_of_descriptors: Descriptor,
        batch_of_fitnesses: Fitness,
        batch_of_extra_scores: Optional[ExtraScores] = None,
    ) -> ESRepertoire:
        """
        Add a batch of elements to the repertoire.

        Args:
            batch_of_genotypes: a batch of genotypes to be added to the repertoire.
                Similarly to the self.genotypes argument, this is a PyTree in which
                the leaves have a shape (batch_size, num_features)
            batch_of_descriptors: an array that contains the descriptors of the
                aforementioned genotypes. Its shape is (batch_size, num_descriptors)
            batch_of_fitnesses: an array that contains the fitnesses of the
                aforementioned genotypes. Its shape is (batch_size,)
            batch_of_extra_scores: unused tree that contains the extra_scores of
                aforementioned genotypes.

        Returns:
            The updated MAP-Elites repertoire.
        """

        batch_of_indices = get_cells_indices(batch_of_descriptors, self.centroids)
        batch_of_indices = jnp.expand_dims(batch_of_indices, axis=-1)
        batch_of_fitnesses = jnp.expand_dims(batch_of_fitnesses, axis=-1)

        num_centroids = self.centroids.shape[0]

        # get fitness segment max
        best_fitnesses = jax.ops.segment_max(
            batch_of_fitnesses,
            batch_of_indices.astype(jnp.int32).squeeze(axis=-1),
            num_segments=num_centroids,
        )

        cond_values = jnp.take_along_axis(best_fitnesses, batch_of_indices, 0)

        # put dominated fitness to -jnp.inf
        batch_of_fitnesses = jnp.where(
            batch_of_fitnesses == cond_values, x=batch_of_fitnesses, y=-jnp.inf
        )

        # get addition condition
        repertoire_fitnesses = jnp.expand_dims(self.fitnesses, axis=-1)
        current_fitnesses = jnp.take_along_axis(
            repertoire_fitnesses, batch_of_indices, 0
        )
        addition_condition = batch_of_fitnesses > current_fitnesses

        # assign fake position when relevant : num_centroids is out of bound
        batch_of_indices = jnp.where(
            addition_condition, x=batch_of_indices, y=num_centroids
        )

        # create new repertoire
        new_repertoire_genotypes = jax.tree_util.tree_map(
            lambda repertoire_genotypes, new_genotypes: repertoire_genotypes.at[
                batch_of_indices.squeeze(axis=-1)
            ].set(new_genotypes),
            self.genotypes,
            batch_of_genotypes,
        )

        # compute new fitness and descriptors
        new_fitnesses = self.fitnesses.at[batch_of_indices.squeeze(axis=-1)].set(
            batch_of_fitnesses.squeeze(axis=-1)
        )
        new_descriptors = self.descriptors.at[batch_of_indices.squeeze(axis=-1)].set(
            batch_of_descriptors
        )

        last_fitness = jnp.mean(batch_of_fitnesses)

        return ESRepertoire(
            genotypes=new_repertoire_genotypes,
            fitnesses=new_fitnesses,
            descriptors=new_descriptors,
            centroids=self.centroids,
            last_fitness=last_fitness,
        )

    @classmethod
    def init_default(
        cls,
        genotype: Genotype,
        centroids: Centroid,
    ) -> ESRepertoire:
        """Initialize a Map-Elites repertoire with an initial population of
        genotypes. Requires the definition of centroids that can be computed
        with any method such as CVT or Euclidean mapping.

        Note: this function has been kept outside of the object MapElites, so
        it can be called easily called from other modules.

        Args:
            genotype: the typical genotype that will be stored.
            centroids: the centroids of the repertoire

        Returns:
            A repertoire filled with default values.
        """

        # get number of centroids
        num_centroids = centroids.shape[0]

        # default fitness is -inf
        default_fitnesses = -jnp.inf * jnp.ones(shape=num_centroids)

        # default genotypes is all 0
        default_genotypes = jax.tree_util.tree_map(
            lambda x: jnp.zeros(shape=(num_centroids,) + x.shape, dtype=x.dtype),
            genotype,
        )

        # default descriptor is all zeros
        default_descriptors = jnp.zeros_like(centroids)

        return cls(
            genotypes=default_genotypes,
            fitnesses=default_fitnesses,
            descriptors=default_descriptors,
            centroids=centroids,
            last_fitness=-jnp.inf,
        )
    
class ES(MAPElites):
    """ Map-Elite structure to run a standalone ES
    """

    @partial(jax.jit, static_argnames=("self",))
    def init(
        self,
        init_genotypes: Genotype,
        centroids: Centroid,
        random_key: RNGKey,
    ) -> Tuple[ESRepertoire, Optional[EmitterState], RNGKey]:
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

        # init the repertoire
        repertoire = ESRepertoire.init(
            genotypes=init_genotypes,
            fitnesses=fitnesses,
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

