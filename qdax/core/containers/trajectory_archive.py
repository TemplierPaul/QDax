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

from qdax.types import Centroid, Descriptor, ExtraScores, Fitness, Genotype, RNGKey


class Trajectory(flax.struct.PyTreeNode):
    """Class for the repertoire in Map Elites.

    Args:
        genotypes: a PyTree containing all the genotypes in the repertoire ordered
            by the centroids. Each leaf has a shape (num_centroids, num_features). The
            PyTree can be a simple Jax array or a more complex nested structure such
            as to represent parameters of neural network in Flax.
        fitnesses: an array that contains the fitness of solutions in each cell of the
            repertoire, ordered by centroids. The array shape is (num_centroids,).
        descriptors: an array that contains the descriptors of solutions in each cell
            of the repertoire, ordered by centroids. The array shape
            is (num_centroids, num_descriptors).
        centroids: an array the contains the centroids of the tesselation. The array
            shape is (num_centroids, num_descriptors).
    """

    genotypes: Genotype
    fitnesses: Fitness
    descriptors: Descriptor
    generations: int

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
        jnp.save(path + "generations.npy", self.generations)

    @classmethod
    def load(cls, reconstruction_fn: Callable, path: str = "./") -> Trajectory:
        """Loads a MAP Elites Repertoire.

        Args:
            reconstruction_fn: Function to reconstruct a PyTree
                from a flat array.
            path: Path where the data is saved. Defaults to "./".

        Returns:
            A MAP Elites Repertoire.
        """

        flat_genotypes = jnp.load(path + "genotypes.npy")
        genotypes = jax.vmap(reconstruction_fn)(flat_genotypes)

        fitnesses = jnp.load(path + "fitnesses.npy")
        descriptors = jnp.load(path + "descriptors.npy")
        generations = jnp.load(path + "generations.npy")

        return cls(
            genotypes=genotypes,
            fitnesses=fitnesses,
            descriptors=descriptors,
            generations=generations,
        )
    
    @jax.jit
    def add(
        self,
        genotype: Genotype,
        fitnesse: Fitness,
        descriptor: Descriptor,
        generation: int,
    ) -> Trajectory:
        """
        Add a new genotype to the trajectory
        """
        pass

    @classmethod
    def init(
        cls,
        genotypes: Genotype,
        fitnesses: Fitness,
        descriptors: Descriptor,
        size: int,
    ) -> Trajectory:
        """
        Initialize a new trajectory
        """
        first_genotype = jax.tree_util.tree_map(lambda x: x[0], genotypes)

        default_fitnesses = -jnp.inf * jnp.ones(shape=size)

        default_generations = -jnp.inf * jnp.ones(shape=size)

        # default genotypes is all 0
        default_genotypes = jax.tree_util.tree_map(
            lambda x: jnp.zeros(shape=(size,) + x.shape, dtype=x.dtype),
            first_genotype,
        )

        # default descriptor is all zeros
        default_descriptors = jnp.zeros_like(centroids)

        return cls(
            genotypes=default_genotypes,
            fitnesses=default_fitnesses,
            descriptors=default_descriptors,
            generations=default_generations,
        )
    
    def compute_PCA(self) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Compute the PCA of the genomes in the trajectory

        Returns:
            The mean and the eigenvectors of the PCA
        """
        pass


def frobenius_norm(A):
    return jnp.sqrt(jnp.sum(A**2))


def pca_ratio(W, X):
    return frobenius_norm(X @ W) ** 2 / frobenius_norm(X) ** 2

from jax.scipy.linalg import svd


class PCA(flax.struct.PyTreeNode):
    n_dim: int
    center: Genotype
    principal_components: jnp.ndarray
    explained_variance: jnp.ndarray

    @classmethod
    def init(
            cls, 
            genomes: Genotype,
            n_dim: int = None,
            center_last=True,
            ): 
        """
        Compute the PCA of a set of genomes
        """
        
        # compute the PCA
        center, principal_components, explained_variance = cls.compute_PCA(genomes, center_last=center_last)
        
        if n_dim is None:
            n_dim = principal_components.shape[0]
        else:
            # limit the number of dimensions
            principal_components = principal_components[:n_dim]
            explained_variance = explained_variance[:n_dim]

        explained_variance = explained_variance.sum()
        print(f"PCA variance: {explained_variance}")

        # pca = cls(
        #     n_dim=n_dim,
        #     center=center,
        #     principal_components=principal_components,
        #     explained_variance=jnp.zeros(shape=(n_dim,))
        # )

        # # compute the explained variance
        # explained_variance = pca.get_explained_variance(genomes, n_dim)

        pca = cls(
            n_dim=n_dim,
            center=center,
            principal_components=principal_components,
            explained_variance=explained_variance,
        )
        var = pca.get_explained_variance(genomes, n_dim)
        print(f"Computed variance:: {var}")

        return pca

    @classmethod
    # @jax.jit
    def compute_PCA(cls, genomes: Genotype, center_last: bool) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """
        Compute the PCA of the genomes in the trajectory

        Returns:
            The mean and the eigenvectors of the PCA
        """
        # center on last generation
        if center_last:
            center = genomes[-1]
        else:
            center = jnp.mean(genomes, axis=0)
        norm_genomes = genomes - center

        # compute the PCA
        _, s, principal_components = svd(norm_genomes, full_matrices=False)

        # sort the principal components by decreasing variance
        idx = jnp.argsort(s)[::-1]
        principal_components = principal_components[idx]
        s = s[idx]

        # compute the explained variance ratio
        explained_variance = s ** 2 / jnp.sum(s ** 2)

        return center, principal_components, explained_variance

    # @jax.jit
    def get_explained_variance(self, genomes: Genotype, n_dim: int) -> jnp.ndarray:
        """
        Compute the explained variance of the genomes
        """
        norm_genomes = genomes - self.center
        # compute the variance of the genomes explained by the principal components

        components = self.principal_components[:n_dim].T
        ratio = pca_ratio(components, norm_genomes)

        # ratio = pca_ratio(self.principal_components.T, norm_genomes)
        # print(ratio)
        return ratio
    
    def transform(self, genomes: Genotype, n_dim=None) -> jnp.ndarray:
        """
        Transform the genomes into the PCA space
        """
        norm_genomes = genomes - self.center
        components = self.principal_components[:n_dim].T
        return norm_genomes @ components
    
    def inverse_transform(self, genomes: jnp.ndarray) -> jnp.ndarray:
        """
        Transform the genomes into the original space
        """
        n_dim = genomes.shape[1]
        components = self.principal_components[:n_dim]
        return genomes @ components + self.center