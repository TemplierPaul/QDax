from dataclasses import dataclass
from functools import partial
from typing import Any, Optional, Tuple
from typing import Callable, Tuple


import flax.linen as nn
import jax
import optax
from jax import numpy as jnp
from qdax.core.emitters.esrl_emitter import ESRLConfig, ESRLEmitterState, ESRLEmitter
from qdax.types import Descriptor, ExtraScores, Fitness, Genotype, Params, RNGKey
from qdax.core.emitters.vanilla_es_emitter import (
    VanillaESConfig,
    VanillaESEmitterState,
    VanillaESEmitter,
    NoveltyArchive,
)
from qdax.core.emitters.qpg_emitter import (
    QualityPGConfig,
    QualityPGEmitterState,
    QualityPGEmitter,
)
from qdax.core.rl_es_parts.es_utils import ESRepertoire, ESMetrics
from jax.tree_util import tree_flatten, tree_unflatten, tree_map


@dataclass
class NoESConfig(VanillaESConfig):
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
    novelty_nearest_neighbors: int = 10
    explo_noise: float = 0.0


class NoESEmitterState(VanillaESEmitterState):
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
    optimizer_state: optax.OptState = None  # Not used by canonical ES
    initial_center: Genotype = None
    metrics: ESMetrics = ESMetrics()


class NoESEmitter(VanillaESEmitter):
    def __init__(
        self,
        config: NoESConfig,
        rollout_fn: Callable[
            [Genotype, RNGKey], Tuple[Fitness, Descriptor, ExtraScores, RNGKey]
        ],
        eval_fn: Callable[
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
        self._eval_fn = eval_fn
        self._rollout_fn = rollout_fn
        self._total_generations = total_generations
        self._num_descriptors = num_descriptors

        # Add a wrapper to the scoring function to handle the surrogate data
        extended_scoring = lambda networks, random_key, extra: self._rollout_fn(
            networks, random_key
        )

        self._es_emitter = partial(
            self._base_es_emitter,
            fitness_function=extended_scoring,
            surrogate_data=None,
        )
        self._es_emitter = partial(
            jax.jit,
            static_argnames=("scores_fn"),
        )(self._es_emitter)

        self.tree_def = None
        self.layer_sizes = None
        self.split_indices = None

    @property
    def config_string(self):
        """Returns a string describing the config."""
        s = f"MultiRL {self._config.sample_number} "
        if self._config.explo_noise > 0:
            s += f"| explo {self._config.explo_noise} "
        return s

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

        flat_variables, tree_def = tree_flatten(init_genotypes)
        self.layer_shapes = [x.shape[1:] for x in flat_variables]
        print("layer_shapes", self.layer_shapes)

        sizes = [x.size for x in flat_variables]
        sizes = jnp.array(sizes)

        print("sizes", sizes)

        self.tree_def = tree_def
        self.layer_sizes = sizes.tolist()
        print("layer_sizes", self.layer_sizes)
        self.split_indices = jnp.cumsum(jnp.array(self.layer_sizes))[:-1].tolist()
        print("split_indices", self.split_indices)

        return (
            NoESEmitterState(
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
        static_argnames=("self", "scores_fn", "fitness_function"),
    )
    def _base_es_emitter(
        self,
        parent: Genotype,
        optimizer_state: optax.OptState,
        random_key: RNGKey,
        scores_fn: Callable[[Fitness, Descriptor], jnp.ndarray],
        fitness_function: Callable[[Genotype], RNGKey],
        surrogate_data=None,
        actor: Genotype = None,
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
        sample_number = self._config.sample_number

        # print("actor shape", jax.tree_map(lambda x: x.shape, actor))

        # Repeat the actor n times
        networks = jax.tree_util.tree_map(
            lambda x: jnp.repeat(x[None, ...], sample_number, axis=0),
            actor,
        )
        # print("repeated actor shape", jax.tree_map(lambda x: x.shape, networks))

        # Evaluating the actor
        fitnesses, descriptors, extra_scores, random_key = fitness_function(
            networks, random_key, surrogate_data
        )

        extra_scores["population_fitness"] = fitnesses

        return actor, optimizer_state, random_key, extra_scores


class MultiActorTD3(ESRLEmitter):
    @partial(
        jax.jit,
        static_argnames=("self",),
    )
    def emit(
        self,
        repertoire: ESRepertoire,
        emitter_state: ESRLEmitterState,
        random_key: RNGKey,
    ) -> Tuple[Genotype, RNGKey]:
        """Return the offspring generated through RL+ES update.

        Params:
            repertoire: unused
            emitter_state
            random_key: a jax PRNG random key

        Returns:
            a new gradient offspring
            a new jax PRNG key
        """
        actor = emitter_state.rl_state.actor_params
        # print("actor shape", jax.tree_map(lambda x: x.shape, actor))
        # add a dimension to the actor
        actor = jax.tree_map(lambda x: x[None, ...], actor)
        # print("reshaped actor shape", jax.tree_map(lambda x: x.shape, actor))

        return actor, random_key

    @partial(
        jax.jit,
        static_argnames=("self",),
    )
    def state_update(
        self,
        emitter_state: ESRLEmitterState,
        repertoire: ESRepertoire,
        genotypes: Genotype,
        fitnesses: Fitness,
        descriptors: Descriptor,
        extra_scores: ExtraScores,
    ) -> ESRLEmitterState:
        """Generate the gradient offspring for the next emitter call. Also
        update the novelty archive and generation count from current call.

        Chooses between ES and RL update.

        Params:
            emitter_state
            repertoire: unused
            genotypes: the genotypes of the offspring
            fitnesses: the fitnesses of the offspring
            descriptors: the descriptors of the offspring
            extra_scores: the extra scores of the offspring

        Returns:
            the updated emitter state
        """

        key, emitter_state = emitter_state.get_key()

        # Choose between ES and RL update with probability 0.5
        # cond = jax.random.choice(key,
        #                          jnp.array([True, False]),
        #                          p=jnp.array([self._config.es_proba, 1-self._config.es_proba]))

        # Do RL if the ES has done more steps than RL
        # cond = emitter_state.metrics.es_updates <= emitter_state.metrics.rl_updates

        emitter_state, pop_extra_scores = self.es_state_update(
            emitter_state,
            repertoire,
            genotypes,
            fitnesses,
            descriptors,
            extra_scores,
        )

        offspring = emitter_state.rl_state.actor_params
        # add dimension to the offspring
        offspring = jax.tree_map(lambda x: x[None, ...], offspring)

        metrics = self.get_metrics(
            emitter_state,
            offspring,
            pop_extra_scores,
            fitnesses,
            new_evaluations=self._config.es_config.sample_number,
            random_key=key,
        )

        # Actor evaluation
        key, emitter_state = emitter_state.get_key()

        actor_genome = emitter_state.rl_state.actor_params
        actor_fitness, _ = self.multi_eval(actor_genome, key)

        metrics = metrics.replace(
            actor_fitness=actor_fitness,
            # center_fitness=center_fitness,
        )

        emitter_state = emitter_state.set_metrics(metrics)

        return emitter_state

    @partial(
        jax.jit,
        static_argnames=("self",),
    )
    def es_state_update(
        self,
        emitter_state: ESRLEmitterState,
        repertoire: ESRepertoire,
        genotypes: Genotype,
        fitnesses: Fitness,
        descriptors: Descriptor,
        extra_scores: ExtraScores,
    ) -> ESRLEmitterState:
        """Generate the gradient offspring for the next emitter call. Also
        update the novelty archive and generation count from current call.

        Params:
            emitter_state
            repertoire: unused
            genotypes: the genotypes of the offspring
            fitnesses: the fitnesses of the offspring
            descriptors: the descriptors of the offspring
            extra_scores: the extra scores of the offspring

        Returns:
            the updated emitter state
        """
        random_key, emitter_state = emitter_state.get_key()

        actor = emitter_state.rl_state.actor_params
        # print("es_state_update actor shape", jax.tree_map(lambda x: x.shape, actor))
        # print(
        # "es_state_update genotypes shape",
        #     jax.tree_map(lambda x: x.shape, genotypes),
        # )

        # Add dimension to actor so it looks like a population of 1
        genotypes = jax.tree_map(lambda x: x[None, ...], actor)

        assert jax.tree_util.tree_leaves(genotypes)[0].shape[0] == 1, (
            "ERROR: ES generates 1 offspring per generation, "
            + "batch_size should be 1, the inputed batch has size:"
            + str(jax.tree_util.tree_leaves(genotypes)[0].shape[0])
        )

        # Updating novelty archive
        novelty_archive = emitter_state.es_state.novelty_archive.update(descriptors)

        # Define scores for es process
        def scores(fitnesses: Fitness, descriptors: Descriptor) -> jnp.ndarray:
            return fitnesses

        base_optim_state = emitter_state.es_state.optimizer_state
        # Run es process
        offspring, optimizer_state, new_random_key, extra_scores = self.true_es_emitter(
            parent=genotypes,
            optimizer_state=base_optim_state,
            random_key=random_key,
            scores_fn=scores,
            actor=actor,
        )

        offspring = emitter_state.rl_state.actor_params
        # add dimension to the offspring
        offspring = jax.tree_map(lambda x: x[None, ...], offspring)
        random_key = new_random_key

        # Update ES emitter state
        es_state = emitter_state.es_state.replace(
            offspring=offspring,
            optimizer_state=optimizer_state,
            random_key=random_key,
            novelty_archive=novelty_archive,
            generation_count=emitter_state.es_state.generation_count + 1,
        )

        # Update QPG emitter to train RL agent
        # Update random key
        rl_state = emitter_state.rl_state.replace(
            random_key=random_key,
            es_center=offspring,
        )

        rl_state = self.rl_emitter.state_update(
            emitter_state=rl_state,
            repertoire=repertoire,
            genotypes=genotypes,
            fitnesses=fitnesses,
            descriptors=descriptors,
            extra_scores=extra_scores,
        )

        # metrics = self.es_emitter.get_metrics(
        #     es_state,
        #     offspring,
        #     extra_scores,
        #     fitnesses,
        #     # evaluations=emitter_state.metrics.evaluations,
        #     random_key=random_key,
        # )

        metrics = emitter_state.metrics.replace(
            es_updates=emitter_state.metrics.es_updates + 1,
            rl_updates=emitter_state.metrics.rl_updates,
        )

        # Share random key between ES and RL emitters

        state = ESRLEmitterState(es_state, rl_state)
        state = state.set_metrics(metrics)
        state = state.set_key(random_key)
        # print("ES offspring", jax.tree_map(lambda x: x.shape, offspring))

        return state, extra_scores
