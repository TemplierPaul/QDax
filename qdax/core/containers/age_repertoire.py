from __future__ import annotations

import warnings
from functools import partial
from typing import Callable, List, Optional, Tuple, Union

import flax
import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree
from numpy.random import RandomState
from sklearn.cluster import KMeans

from qdax.types import Centroid, Descriptor, ExtraScores, Fitness, Genotype, RNGKey, Metrics

from qdax.core.containers.mapelites_repertoire import MapElitesRepertoire, get_cells_indices

from typing_extensions import TypeAlias

Age: TypeAlias = jnp.ndarray

class AgeMapElitesRepertoire(MapElitesRepertoire):
    ages: Age

    def save(self, path: str = "./") -> None:
        """Saves the repertoire on disk in the form of .npy files.

        Flattens the genotypes to store it with .npy format. Supposes that
        a user will have access to the reconstruction function when loading
        the genotypes.

        Args:
            path: Path where the data will be saved. Defaults to "./".
        """

        def flatten_genotype(genotype: Genotype) -> jnp.ndarray:
            flatten_genotype, _ = ravel_pytree(genotype)
            return flatten_genotype

        # flatten all the genotypes
        flat_genotypes = jax.vmap(flatten_genotype)(self.genotypes)

        # save data
        jnp.save(path + "genotypes.npy", flat_genotypes)
        jnp.save(path + "fitnesses.npy", self.fitnesses)
        jnp.save(path + "descriptors.npy", self.descriptors)
        jnp.save(path + "centroids.npy", self.centroids)
        jnp.save(path + "ages.npy", self.ages)

    @classmethod
    def load(cls, reconstruction_fn: Callable, path: str = "./") -> AgeMapElitesRepertoire:
        """Loads an Age MAP Elites Repertoire.

        Args:
            reconstruction_fn: Function to reconstruct a PyTree
                from a flat array.
            path: Path where the data is saved. Defaults to "./".

        Returns:
            An Age MAP Elites Repertoire.
        """

        flat_genotypes = jnp.load(path + "genotypes.npy")
        genotypes = jax.vmap(reconstruction_fn)(flat_genotypes)

        fitnesses = jnp.load(path + "fitnesses.npy")
        descriptors = jnp.load(path + "descriptors.npy")
        centroids = jnp.load(path + "centroids.npy")
        ages = jnp.load(path + "ages.npy")

        return cls(
            genotypes=genotypes,
            fitnesses=fitnesses,
            descriptors=descriptors,
            centroids=centroids,
            ages=ages,
        )
    
    @partial(jax.jit, static_argnames=("num_samples",))
    def sample(self, random_key: RNGKey, num_samples: int) -> Tuple[Genotype, Age, RNGKey]:
        """Sample elements in the repertoire.

        Args:
            random_key: a jax PRNG random key
            num_samples: the number of elements to be sampled

        Returns:
            samples: a batch of genotypes sampled in the repertoire
            ages: a batch of the ages of the sampled genotypes
            random_key: an updated jax PRNG random key
        """

        repertoire_empty = self.fitnesses == -jnp.inf
        p = (1.0 - repertoire_empty) / jnp.sum(1.0 - repertoire_empty)

        random_key, subkey = jax.random.split(random_key)
        samples = jax.tree_util.tree_map(
            lambda x: jax.random.choice(subkey, x, shape=(num_samples,), p=p),
            self.genotypes,
        )

        # Get the corresponding ages
        ages = jax.tree_util.tree_map(
            lambda x: jax.random.choice(subkey, x, shape=(num_samples,), p=p),
            self.ages,
        )

        return samples, ages, random_key
    
    @jax.jit
    def add(
        self,
        batch_of_genotypes: Genotype,
        batch_of_descriptors: Descriptor,
        batch_of_fitnesses: Fitness,
        batch_of_ages: Age,
        batch_of_extra_scores: Optional[ExtraScores] = None,
    ) -> AgeMapElitesRepertoire:
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
            batch_of_ages: an array that contains the ages of the
                aforementioned genotypes. Its shape is (batch_size,)
            batch_of_extra_scores: unused tree that contains the extra_scores of
                aforementioned genotypes.

        Returns:
            The updated MAP-Elites repertoire.
        """

        batch_of_indices = get_cells_indices(batch_of_descriptors, self.centroids)
        batch_of_indices = jnp.expand_dims(batch_of_indices, axis=-1) # BD index of each genotype
        batch_of_fitnesses = jnp.expand_dims(batch_of_fitnesses, axis=-1) # fitness of each genotype
        batch_of_ages = jnp.expand_dims(batch_of_ages, axis=-1) # age of each genotype

        num_centroids = self.centroids.shape[0] # number of BD cells in the map

        # get fitness segment max
        best_fitnesses = jax.ops.segment_max(
            batch_of_fitnesses,
            batch_of_indices.astype(jnp.int32).squeeze(axis=-1),
            num_segments=num_centroids,
        ) # for each BD in the new batch, get the max fitness, shape (num_centroids, 1)

        cond_values = jnp.take_along_axis(best_fitnesses, batch_of_indices, 0) # for each genotype, get the max fitness of its BD cell, shape (batch_size, 1)

        # put dominated fitness to -jnp.inf
        batch_of_fitnesses = jnp.where(
            batch_of_fitnesses == cond_values, x=batch_of_fitnesses, y=-jnp.inf
        ) # for each genotype, if its fitness is not the max fitness of its BD cell, set it to -jnp.inf

        # get addition condition
        repertoire_fitnesses = jnp.expand_dims(self.fitnesses, axis=-1) # fitness of each genotype in the repertoire
        current_fitnesses = jnp.take_along_axis(
            repertoire_fitnesses, batch_of_indices, 0
        ) # for each genotype in the new batch, get the fitness of the genotype in the repertoire, shape (batch_size, 1)
        addition_condition = batch_of_fitnesses > current_fitnesses # for each genotype in the new batch, check if its fitness is better than the fitness of the genotype in the repertoire, shape (batch_size, 1)

        # assign fake position when relevant : num_centroids is out of bound
        batch_of_indices = jnp.where(
            addition_condition, x=batch_of_indices, y=num_centroids
        ) # for each genotype in the new batch, set its BD index to num_centroids if it is not better than the genotype in the repertoire

        # create new repertoire
        new_repertoire_genotypes = jax.tree_util.tree_map(
            lambda repertoire_genotypes, new_genotypes: repertoire_genotypes.at[
                batch_of_indices.squeeze(axis=-1)
            ].set(new_genotypes),
            self.genotypes,
            batch_of_genotypes,
        ) # for each genotype in the new batch, set its genotype in the repertoire if it is better than the genotype in the repertoire

        # compute new fitness and descriptors
        new_fitnesses = self.fitnesses.at[batch_of_indices.squeeze(axis=-1)].set(
            batch_of_fitnesses.squeeze(axis=-1)
        )
        new_descriptors = self.descriptors.at[batch_of_indices.squeeze(axis=-1)].set(
            batch_of_descriptors
        )
        new_ages = self.ages.at[batch_of_indices.squeeze(axis=-1)].set(
            batch_of_ages.squeeze(axis=-1)
        )

        return AgeMapElitesRepertoire(
            genotypes=new_repertoire_genotypes,
            fitnesses=new_fitnesses,
            descriptors=new_descriptors,
            centroids=self.centroids,
            ages=new_ages,
        )
    
    @classmethod
    def init(
        cls,
        genotypes: Genotype,
        fitnesses: Fitness,
        descriptors: Descriptor,
        centroids: Centroid,
        ages: Age,
        extra_scores: Optional[ExtraScores] = None,
    ) -> AgeMapElitesRepertoire:
        """
        Initialize a Map-Elites repertoire with an initial population of genotypes.
        Requires the definition of centroids that can be computed with any method
        such as CVT or Euclidean mapping.

        Note: this function has been kept outside of the object MapElites, so it can
        be called easily called from other modules.

        Args:
            genotypes: initial genotypes, pytree in which leaves
                have shape (batch_size, num_features)
            fitnesses: fitness of the initial genotypes of shape (batch_size,)
            descriptors: descriptors of the initial genotypes
                of shape (batch_size, num_descriptors)
            centroids: tesselation centroids of shape (batch_size, num_descriptors)
            extra_scores: unused extra_scores of the initial genotypes

        Returns:
            an initialized MAP-Elite repertoire
        """
        warnings.warn(
            (
                "This type of repertoire does not store the extra scores "
                "computed by the scoring function"
            ),
            stacklevel=2,
        )

        # retrieve one genotype from the population
        first_genotype = jax.tree_util.tree_map(lambda x: x[0], genotypes)

        # create a repertoire with default values
        repertoire = cls.init_default(genotype=first_genotype, centroids=centroids)

        # add initial population to the repertoire
        new_repertoire = repertoire.add(
            genotypes, 
            descriptors, 
            fitnesses, 
            ages, 
            extra_scores
            )

        return new_repertoire  # type: ignore
    
    @classmethod
    def init_default(
        cls,
        genotype: Genotype,
        centroids: Centroid,
    ) -> AgeMapElitesRepertoire:
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

        default_ages = -jnp.inf * jnp.ones(shape=num_centroids)

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
            ages=default_ages,
        )
    

def age_qd_metrics(repertoire: AgeMapElitesRepertoire, qd_offset: float) -> Metrics:
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

    # get metrics
    repertoire_empty = repertoire.fitnesses == -jnp.inf
    qd_score = jnp.sum(repertoire.fitnesses, where=~repertoire_empty)
    qd_score += qd_offset * jnp.sum(1.0 - repertoire_empty)
    coverage = 100 * jnp.mean(1.0 - repertoire_empty)
    max_fitness = jnp.max(repertoire.fitnesses)

    # From actor: age is not -inf and cell is not empty
    from_actor = repertoire.ages >= 0
    from_actor_ratio = jnp.sum(from_actor) / jnp.sum(~repertoire_empty)

    # Average age of cells that are not empty and have an age >= 0
    average_age = jnp.sum(repertoire.ages, where=repertoire.ages >= 0) / jnp.sum(
        repertoire.ages >= 0
    )

    return {
        "qd_score": qd_score, 
        "max_fitness": max_fitness, 
        "coverage": coverage, 
        "from_actor_ratio": from_actor_ratio,
        "average_age": average_age,
        }