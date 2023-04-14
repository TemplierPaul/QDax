"""
Emitter and utils necessary to create an ES or NSES emitter with
a passive archive. This emitter enevr sample from the reperoire.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Callable, Tuple

import flax
import jax
import jax.numpy as jnp
import optax

# from qdax.core.containers.mapelites_repertoire import ESRepertoire
from qdax.core.rl_es_parts.es_utils import ESRepertoire, ESMetrics
from qdax.core.emitters.emitter import Emitter, EmitterState
from qdax.types import Descriptor, ExtraScores, Fitness, Genotype, RNGKey
from qdax.core.cmaes import CMAESState
from jax.flatten_util import ravel_pytree

def flatten_genotype(genotype: Genotype) -> jnp.ndarray:
        flatten_genotype, _ = ravel_pytree(genotype)
        return flatten_genotype

class NoveltyArchive(flax.struct.PyTreeNode):
    """Novelty Archive used by NS-ES.

    Args:
        archive: content of the archive
        size: total size of the archive
        position: current position in the archive
    """

    archive: jnp.ndarray
    size: int = flax.struct.field(pytree_node=False)
    position: jnp.ndarray = flax.struct.field()

    @classmethod
    def init(
        cls,
        size: int,
        num_descriptors: int,
    ) -> NoveltyArchive:
        archive = jnp.zeros((size, num_descriptors))
        return cls(archive=archive, size=size, position=jnp.array(0, dtype=int))

    @jax.jit
    def update(
        self,
        descriptor: Descriptor,
    ) -> NoveltyArchive:
        """Update the content of the novelty archive with newly generated descriptor.

        Args:
            descriptor: new descriptor generated by NS-ES
        Returns:
            The updated NoveltyArchive
        """

        new_archive = jax.lax.dynamic_update_slice_in_dim(
            self.archive,
            descriptor,
            self.position,
            axis=0,
        )
        new_position = (self.position + 1) % self.size
        return NoveltyArchive(
            archive=new_archive, size=self.size, position=new_position
        )

    @partial(jax.jit, static_argnames=("num_nearest_neighbors",))
    def novelty(
        self,
        descriptors: Descriptor,
        num_nearest_neighbors: int,
    ) -> jnp.ndarray:
        """Compute the novelty of the given descriptors as the average distance
        to the k nearest neighbours in the archive.

        Args:
            descriptors: the descriptors to compute novelty for
            num_nearest_neighbors: k used to compute the k-nearest-neighbours
        Returns:
            the novelty of each descriptor in descriptors.
        """

        # Compute all distances with archive content
        def distance(x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
            return jnp.sqrt(jnp.sum(jnp.square(x - y)))

        distances = jax.vmap(
            jax.vmap(partial(distance), in_axes=(None, 0)), in_axes=(0, None)
        )(descriptors, self.archive)

        # Filter distance with empty slot of archive
        indices = jnp.arange(0, self.size, step=1) < self.position + 1
        distances = jax.vmap(lambda distance: jnp.where(indices, distance, jnp.inf))(
            distances
        )

        # Find k nearest neighbours
        _, indices = jax.lax.top_k(-distances, num_nearest_neighbors)

        # Compute novelty as average distance with k neirest neirghbours
        distances = jnp.where(distances == jnp.inf, jnp.nan, distances)
        novelty = jnp.nanmean(jnp.take_along_axis(distances, indices, axis=1), axis=1)
        return novelty


@dataclass
class VanillaESConfig:
    """Configuration for the ES or NSES emitter.

    Args:
        nses_emitter: if True, use NSES, if False, use ES
        sample_number: num of samples for gradient estimate
        sample_sigma: std to sample the samples for gradient estimate
        sample_mirror: if True, use mirroring sampling
        sample_rank_norm: if True, use normalisation
        adam_optimizer: if True, use ADAM, if False, use SGD
        learning_rate
        l2_coefficient: coefficient for regularisation
            novelty_nearest_neighbors
    """

    nses_emitter: bool = False
    sample_number: int = 1000
    sample_sigma: float = 0.02
    sample_mirror: bool = True
    sample_rank_norm: bool = True
    adam_optimizer: bool = True
    learning_rate: float = 0.01
    l2_coefficient: float = 0.02
    novelty_nearest_neighbors: int = 10
    actor_injection: bool = False
    nb_injections: int = 1


class VanillaESEmitterState(EmitterState):
    """Emitter State for the ES or NSES emitter.

    Args:
        optimizer_state: current optimizer state
        offspring: offspring generated through gradient estimate
        generation_count: generation counter used to update the novelty archive
        novelty_archive: used to compute novelty for explore
        random_key: key to handle stochastic operations
    """

    offspring: Genotype
    generation_count: int
    novelty_archive: NoveltyArchive
    random_key: RNGKey
    optimizer_state: optax.OptState
    initial_center: Genotype
    metrics: ESMetrics = ESMetrics()

    def save(self, path):
        """Saves the state to a file."""
        flat_genotypes = flatten_genotype(self.offspring)
        jnp.save(path + "_offspring.npy", flat_genotypes)
        print("Saved offspring to", path + "_offspring.npy")

        

class VanillaESEmitter(Emitter):
    """
    Emitter allowing to reproduce an ES or NSES emitter with
    a passive archive. This emitter enevr sample from the reperoire.

    One can choose between ES and NSES by setting nses_emitter boolean.
    """

    def __init__(
        self,
        config: VanillaESConfig,
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

        # Initialise optimizer
        if self._config.adam_optimizer:
            self._optimizer = optax.adam(learning_rate=config.learning_rate)
        else:
            self._optimizer = optax.sgd(learning_rate=config.learning_rate)

        # Actor injection
        if self._config.actor_injection:
            self._actor_injection = self._inject_actor
        else:
            self._actor_injection = lambda x, a, p: x

    @property
    def batch_size(self) -> int:
        """
        Returns:
            the batch size emitted by the emitter.
        """
        return 1

    @partial(
        jax.jit,
        static_argnames=("self",),
    )
    def init(
        self, init_genotypes: Genotype, random_key: RNGKey
    ) -> Tuple[VanillaESEmitterState, RNGKey]:
        """Initializes the emitter state.

        Args:
            init_genotypes: The initial population.
            random_key: A random key.

        Returns:
            The initial state of the VanillaESEmitter, a new random key.
        """
        # Initialisation requires one initial genotype
        if jax.tree_util.tree_leaves(init_genotypes)[0].shape[0] > 1:
            init_genotypes = jax.tree_util.tree_map(
                lambda x: x[0],
                init_genotypes,
            )

        # Initialise optimizer
        initial_optimizer_state = self._optimizer.init(init_genotypes)

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
        )

        return (
            VanillaESEmitterState(
                optimizer_state=initial_optimizer_state,
                offspring=init_genotypes,
                generation_count=0,
                novelty_archive=novelty_archive,
                random_key=random_key,
                initial_center=init_genotypes,
                metrics=metrics,
            ),
            random_key,
        )

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

        return emitter_state.offspring, random_key

    @partial(
        jax.jit,
        static_argnames=("self"),
    )
    def _sample(
        self, 
        parent: Genotype,
        # total_sample_number: int,
        random_key: RNGKey,
    ) -> Tuple[Genotype, RNGKey]:
        """Sample a batch of genotypes from the parent.

        Args:
            parent: the considered parent.
            number: the number of genotypes to sample.
            random_key

        Returns:
            The sampled noises and a new random_key.
        """
        random_key, subkey = jax.random.split(random_key)
        total_sample_number = self._config.sample_number

        sample_noise = jax.tree_map(
            lambda x: jax.random.normal(
                key=subkey,
                shape=jnp.repeat(x, total_sample_number, axis=0).shape,
            ),
            parent,
        )

        return sample_noise, random_key
    
    @partial(
        jax.jit,
        static_argnames=("self"),
    )
    def _sample_cauchy(
        self, 
        parent: Genotype,
        # total_sample_number: int,
        random_key: RNGKey,
    ) -> Tuple[Genotype, RNGKey]:
        """Sample a batch of genotypes from the parent.

        Args:
            parent: the considered parent.
            number: the number of genotypes to sample.
            random_key

        Returns:
            The sampled noises and a new random_key.
        """
        random_key, subkey = jax.random.split(random_key)
        total_sample_number = self._config.sample_number

        sample_noise = jax.tree_util.tree_map(
            lambda x: jax.random.cauchy(
                key=subkey,
                shape=jnp.repeat(x, total_sample_number, axis=0).shape,
            ),
            parent,
        )

        return sample_noise, random_key
    
    @partial(
        jax.jit,
        static_argnames=("self"),
    )
    def _sample_mirror(
        self, 
        parent: Genotype,
        # total_sample_number: int,
        random_key: RNGKey,
    ) -> Tuple[Genotype, RNGKey]:
        """
        Sample a batch of genotypes from the parent, with a mirror noise.

        Args:
            parent: the considered parent.
            number: the number of genotypes to sample.
            random_key
        
        Returns:
            The sampled noises and a new random_key.
        """
        random_key, subkey = jax.random.split(random_key)
        total_sample_number = self._config.sample_number

        sample_number = total_sample_number // 2
        half_sample_noise = jax.tree_util.tree_map(
            lambda x: jax.random.normal(
                key=subkey,
                shape=jnp.repeat(x, sample_number, axis=0).shape,
            ),
            parent,
        )
        sample_noise = jax.tree_util.tree_map(
            lambda x: jnp.concatenate(
                [jnp.expand_dims(x, axis=1), jnp.expand_dims(-x, axis=1)], axis=1
            ).reshape(jnp.repeat(x, 2, axis=0).shape),
            half_sample_noise,
        )

        return sample_noise, random_key
    
    @partial(
        jax.jit,
        static_argnames=("self"),
    )
    def _inject_actor(
        self, 
        sample_noise: Genotype,
        actor: Genotype,
        parent: Genotype,
    ) -> Genotype:
        """
        Replace the last genotype of the sample_noise by the actor minus the parent.
        """
        raise NotImplementedError
        # Get the noise that recreates the actor
        actor_noise = jax.tree_util.tree_map(
            lambda x, y: (x - y)/self._config.sample_sigma,
            actor,
            parent,
        )

        # Replace the last one 
        sample_noise = jax.tree_util.tree_map(
            lambda x, y: jnp.concatenate([x[:-1], y], axis=0),
            sample_noise,
            actor_noise,
        )

        return sample_noise

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

        raise NotImplementedError

        return offspring, optimizer_state, random_key, extra_scores

    @partial(
        jax.jit,
        static_argnames=("self",),
    )
    def state_update(
        self,
        emitter_state: VanillaESEmitterState,
        repertoire: ESRepertoire,
        genotypes: Genotype,
        fitnesses: Fitness,
        descriptors: Descriptor,
        extra_scores: ExtraScores,
    ) -> VanillaESEmitterState:
        """Generate the gradient offspring for the next emitter call. Also
        update the novelty archive and generation count from current call.

        Args:
            emitter_state: current emitter state.
            repertoire: unused.
            genotypes: the genotypes of the batch of emitted offspring.
            fitnesses: the fitnesses of the batch of emitted offspring.
            descriptors: the descriptors of the emitted offspring.
            extra_scores: a dictionary with other values outputted by the
                scoring function.

        Returns:
            The modified emitter state.
        """

        assert jax.tree_util.tree_leaves(genotypes)[0].shape[0] == 1, (
            "ERROR: Vanilla-ES generates 1 offspring per generation, "
            + "batch_size should be 1, the inputed batch has size:"
            + str(jax.tree_util.tree_leaves(genotypes)[0].shape[0])
        )

        # Updating novelty archive
        novelty_archive = emitter_state.novelty_archive.update(descriptors)

        # Define scores for es process
        def scores(fitnesses: Fitness, descriptors: Descriptor) -> jnp.ndarray:
            if self._config.nses_emitter:
                return novelty_archive.novelty(
                    descriptors, self._config.novelty_nearest_neighbors
                )
            else:
                return fitnesses

        # Run es process
        offspring, optimizer_state, random_key, extra_scores = self._es_emitter(
            parent=genotypes,
            optimizer_state=emitter_state.optimizer_state,
            random_key=emitter_state.random_key,
            scores_fn=scores,
            actor=genotypes,
        )

        metrics = self.get_metrics(
            emitter_state,
            offspring,
            extra_scores,
            fitnesses,
            random_key=random_key,
        )


        return emitter_state.replace(  # type: ignore
            optimizer_state=optimizer_state,
            offspring=offspring,
            novelty_archive=novelty_archive,
            generation_count=emitter_state.generation_count + 1,
            random_key=random_key,
            metrics=metrics,
        )

    @partial(
        jax.jit,
        static_argnames=("self",),
    )
    def get_metrics(
            self, 
            emitter_state: VanillaESEmitterState,
            offspring: Genotype,
            extra_scores: ExtraScores,
            fitnesses: Fitness,
            # evaluations: int,
            random_key: RNGKey,
        ) -> ESMetrics:

        metrics = emitter_state.metrics

        # Log fitness from the center
        center_fitness = jnp.mean(fitnesses)

        metrics = metrics.replace(
            center_fitness=center_fitness,
            evaluations=metrics.evaluations + self._config.sample_number,
        )

        # Population fitness stats
        if "population_fitness" in extra_scores:
            pop_mean = jnp.mean(extra_scores["population_fitness"])
            pop_median = jnp.median(extra_scores["population_fitness"])
            pop_std = jnp.std(extra_scores["population_fitness"])
            pop_min = jnp.min(extra_scores["population_fitness"])
            pop_max = jnp.max(extra_scores["population_fitness"]) 
            metrics = metrics.replace(
                pop_mean=pop_mean,
                pop_median=pop_median,
                pop_std=pop_std,
                pop_min=pop_min,
                pop_max=pop_max,
            )
        
        # Evaluating offspring multiple time
        multi_offspring = jax.tree_util.tree_map(
            lambda x: jnp.repeat(x, self._config.sample_number, axis=0),
            offspring,
        )

        off_fitnesses, _, _, random_key = self._scoring_fn(
            multi_offspring, random_key
        )
         
        metrics = metrics.replace(
            center_mean=jnp.mean(off_fitnesses),
            center_median=jnp.median(off_fitnesses),
            center_std=jnp.std(off_fitnesses),
            center_min=jnp.min(off_fitnesses),
            center_max=jnp.max(off_fitnesses),
        )
        
        return metrics