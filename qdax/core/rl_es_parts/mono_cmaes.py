from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Callable, Tuple

import flax
import jax
import jax.numpy as jnp
from jax.tree_util import tree_flatten, tree_unflatten, tree_map
import optax

from qdax.core.emitters.vanilla_es_emitter import VanillaESConfig, VanillaESEmitterState, VanillaESEmitter, NoveltyArchive
from qdax.types import Descriptor, ExtraScores, Fitness, Genotype, RNGKey
from qdax.core.rl_es_parts.es_utils import ESRepertoire, ESMetrics
from qdax.core.emitters.emitter import Emitter, EmitterState
from qdax.core.cmaes import CMAES, CMAESState

@jax.jit
def jax_cumsum(x):
    _, cumsum = jax.lax.scan(
        lambda c, y: (c + y, c + y), 
        jnp.zeros_like(x), 
        x)
    return cumsum

@dataclass
class MonoCMAESConfig(VanillaESConfig):
    """Configuration for the CMAES with mono solution emitter."""
    nses_emitter: bool = False
    sample_number: int = 1000
    sample_sigma: float = 1e-3
    actor_injection: bool = False


class MonoCMAESState(VanillaESEmitterState):
    """Emitter State for the ES or NSES emitter.

    Args:
        optimizer_state: current optimizer state
        offspring: offspring generated through gradient estimate
        generation_count: generation counter used to update the novelty archive
        novelty_archive: used to compute novelty for explore
        random_key: key to handle stochastic operations
    """

    optimizer_state: CMAESState
    offspring: Genotype
    generation_count: int
    novelty_archive: NoveltyArchive
    random_key: RNGKey
    metrics: ESMetrics


class MonoCMAESEmitter(VanillaESEmitter):
    """
    Emitter allowing to reproduce an ES or NSES emitter with
    a passive archive. This emitter never sample from the reperoire.

    Uses CMAES as optimizer.
    """
    
    def __init__(
        self,
        config: MonoCMAESConfig,
        scoring_fn: Callable[
            [Genotype, RNGKey], Tuple[Fitness, Descriptor, ExtraScores, RNGKey]
        ],
        total_generations: int = 1,
        num_descriptors: int = 2,
    ) -> None:
        """Initialise the ES or NSES emitter.
        WARNING: total_generations and num_descriptors are required for NSES.

        Args:
            config: algorithm config
            scoring_fn: used to evaluate the samples for the gradient estimate.
            total_generations: total number of generations for which the
                emitter will run, allow to initialise the novelty archive.
            num_descriptors: dimension of the descriptors, used to initialise
                the empty novelty archive.
        """

        self._config = config
        self._scoring_fn = scoring_fn
        self._total_generations = total_generations
        self._num_descriptors = num_descriptors
        
        # define a CMAES instance
        self._cmaes = None
        self.tree_def = None
        self.layer_sizes = None
        self.split_indices = None

        # Actor injection not available yet
        if self._config.actor_injection:
            raise NotImplementedError("Actor injection not available for CMAES yet.")

    # @partial(
    #     jax.jit,
    #     static_argnames=("self",),
    # )
    def init(
        self, init_genotypes: Genotype, random_key: RNGKey
    ) -> Tuple[MonoCMAESState, RNGKey]:
        """Initializes the emitter state.

        Args:
            init_genotypes: The initial population.
            random_key: A random key.

        Returns:
            The initial state of the VanillaESEmitter, a new random key.
        """
        # Initialisation requires one initial genotype
        # print("init_genotypes", init_genotypes)

        if jax.tree_util.tree_leaves(init_genotypes)[0].shape[0] > 1:
            init_genotypes = jax.tree_util.tree_map(
                lambda x: x[0],
                init_genotypes,
            )

        flat_variables, tree_def = tree_flatten(init_genotypes)
        self.layer_shapes = [x.shape[1:] for x in flat_variables]
        # print("layer_shapes", self.layer_shapes)

        vect = jnp.concatenate([jnp.ravel(x) for x in flat_variables])
        sizes = [x.size for x in flat_variables]
        sizes = jnp.array(sizes)

        self.tree_def = tree_def
        self.layer_sizes = sizes.tolist()
        # print("layer_sizes", self.layer_sizes)
        self.split_indices = jnp.cumsum(jnp.array(self.layer_sizes))[:-1].tolist()
        # print("split_indices", self.split_indices)



        genotype_dim = len(vect)
        # print("genotype_dim", genotype_dim)

        self._cmaes = CMAES(
            population_size=self._config.sample_number,
            search_dim=genotype_dim,
            # no need for fitness function in that specific case
            fitness_function=None,  # type: ignore
            num_best= self._config.sample_number // 2,
            init_sigma= self._config.sample_sigma,
            mean_init=None,  # will be init at zeros in cmaes
            bias_weights=True,
            delay_eigen_decomposition=True,
        )

        # Create empty Novelty archive
        novelty_archive = NoveltyArchive.init(
            self._total_generations, self._num_descriptors
        )

        metrics = ESMetrics(
            es_updates=0,
            rl_updates=0,
            evaluations=0,
            actor_fitness=-jnp.inf,
            center_fitness=-jnp.inf,
            fitness=-jnp.inf,
        )

        return (
            MonoCMAESState(
                optimizer_state=self._cmaes.init(),
                offspring=init_genotypes,
                generation_count=0,
                novelty_archive=novelty_archive,
                random_key=random_key,
                metrics=metrics,
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

    
    @partial(
        jax.jit,
        static_argnames=("self",),
    )
    def emit(
        self,
        repertoire: ESRepertoire,
        emitter_state: VanillaESEmitterState,
        random_key: RNGKey,
    ) -> Tuple[Genotype, RNGKey]:
        """Return the offspring generated through gradient update.

        Params:
            repertoire: unused
            emitter_state
            random_key: a jax PRNG random key

        Returns:
            a new gradient offspring
            a new jax PRNG key
        """

        offspring_genome = emitter_state.optimizer_state.mean
        offspring = self.unflatten(offspring_genome)

        offspring = jax.tree_map(
            lambda x: jnp.expand_dims(x, axis=0), 
            offspring
        )

        # print("Init offspring", jax.tree_map(lambda x: x.shape, offspring))

        return offspring, random_key
    

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
        # print("Parent", jax.tree_map(lambda x: x.shape, parent))

        # Parent genome
        parent_genome = self.flatten(parent)
        # print("parent_genome", parent_genome.shape)

        samples = jax.random.multivariate_normal(
                key=subkey,
                shape=(self._config.sample_number,),
                mean=parent_genome,
                # Idendity matrix
                # cov=jnp.eye(parent_genome.shape[0])
                cov=(optimizer_state.sigma**2) * optimizer_state.cov_matrix
        )

        # print("samples", samples.shape)
        
        # Turn each sample into a network
        networks = jax.vmap(self.unflatten)(samples)

        # print("Population", jax.tree_map(lambda x: x.shape, networks))
        
        # print("networks", networks.shape)
        
        # Evaluating samples
        fitnesses, descriptors, extra_scores, random_key = self._scoring_fn(
            networks, random_key
        )

        extra_scores["population_fitness"] = fitnesses

        # Computing rank with normalisation
        scores = scores_fn(fitnesses, descriptors)

        # print("scores", scores.shape)


        # Sort samples by scores (descending order)
        idx_sorted = jnp.argsort(-scores)

        sorted_candidates = samples[idx_sorted[: self._cmaes._num_best]]

        new_cmaes_state = self._cmaes.update_state(optimizer_state, sorted_candidates)

        # print("CMA state updated")

        offspring_genome = new_cmaes_state.mean
        offspring = self.unflatten(offspring_genome)
        # print("offspring", offspring)

        # print("Offspring", jax.tree_map(lambda x: x.shape, offspring))


        offspring = jax.tree_map(
            lambda x: jnp.expand_dims(x, axis=0), 
            offspring
        )

        # print("Expanded", jax.tree_map(lambda x: x.shape, offspring))
        return offspring, new_cmaes_state, random_key, extra_scores
