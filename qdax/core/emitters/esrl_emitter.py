from dataclasses import dataclass
from functools import partial
from typing import Any, Optional, Tuple
from typing import Callable, Tuple


import flax.linen as nn
import jax
import optax
from jax import numpy as jnp

from qdax.core.containers.repertoire import Repertoire
from qdax.core.emitters.emitter import Emitter, EmitterState
from qdax.core.neuroevolution.buffers.buffer import QDTransition, ReplayBuffer
from qdax.core.neuroevolution.losses.td3_loss import make_td3_loss_fn
from qdax.core.neuroevolution.networks.networks import QModule
from qdax.environments.base_wrappers import QDEnv
from qdax.types import Descriptor, ExtraScores, Fitness, Genotype, Params, RNGKey
from qdax.core.emitters.vanilla_es_emitter import VanillaESConfig, VanillaESEmitterState, VanillaESEmitter, NoveltyArchive
from qdax.core.emitters.qpg_emitter import QualityPGConfig, QualityPGEmitterState, QualityPGEmitter
from qdax.core.rl_es_parts.es_utils import ESRepertoire, ESMetrics
from qdax.core.cmaes import CMAESState


@dataclass
class ESRLConfig:
    """Configuration for ESRL Emitter"""

    es_config: VanillaESConfig
    rl_config: QualityPGConfig
    es_proba: float = 0.5


class ESRLEmitterState(EmitterState):
    """Contains training state for the learner."""

    es_state: VanillaESEmitterState
    rl_state: QualityPGEmitterState
    # metrics: ESMetrics

    def set_key(self, key: RNGKey) -> "ESRLEmitterState":
        """Sets the random key."""
        es_state = self.es_state.replace(random_key=key)
        rl_state = self.rl_state.replace(random_key=key)
        return self.replace(es_state=es_state, rl_state=rl_state)
    
    def get_key(self) -> RNGKey:
        """Returns the random key."""
        key = self.es_state.random_key
        key, subkey = jax.random.split(key)
        return subkey, self.set_key(key)
    
    @property
    def metrics(self) -> ESMetrics:
        """Returns the metrics."""
        return self.es_state.metrics
    
    @metrics.setter
    def metrics(self, metrics: ESMetrics) -> None:
        """Sets the metrics."""
        self.es_state = self.es_state.replace(metrics=metrics)

class ESRLEmitter(Emitter):
    """
    A policy gradient emitter used to implement the Policy Gradient Assisted MAP-Elites
    (PGA-Map-Elites) algorithm.
    """
    def __init__(
        self,
        config: ESRLConfig,
        es_emitter: VanillaESEmitter,
        rl_emitter: QualityPGEmitter,
    ) -> None:
        
        self.config = config
        self.es_emitter = es_emitter
        self.rl_emitter = rl_emitter

    @property
    def batch_size(self) -> int:
        """
        Returns:
            the batch size emitted by the emitter.
        """
        return 1
    
    # @partial(
    #     jax.jit,
    #     static_argnames=("self",),
    # )
    def init(
        self, init_genotypes: Genotype, random_key: RNGKey
    ) -> Tuple[ESRLEmitterState, RNGKey]:
        """
        Initializes the emitter state.

        Args:
            init_genotypes: the initial genotypes to use.
            random_key: the random key to use for initialization.

        Returns:
            the initialized emitter state.
        """
        # print("RLES init_genotypes", jax.tree_map(lambda x: x.shape, init_genotypes))

        es_state, random_key = self.es_emitter.init(init_genotypes, random_key)
        rl_state, random_key = self.rl_emitter.init(init_genotypes, random_key)
        # Make sure the random key is the same for both emitters

        es_state = es_state.replace(random_key=random_key)
        rl_state = rl_state.replace(random_key=random_key)

        metrics = ESMetrics(
            es_updates=0,
            rl_updates=0,
            evaluations=0,
            actor_fitness=-jnp.inf,
            center_fitness=-jnp.inf,
            fitness=-jnp.inf,
        )

        state = ESRLEmitterState(es_state, rl_state, metrics)
        state = state.set_key(random_key)

        return state, random_key
    
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

        return emitter_state.es_state.offspring, random_key
    
    @partial(
        jax.jit,
        static_argnames=("self",),
    )
    def choose_es_update(self, 
        emitter_state: ESRLEmitterState,
        fitnesses: Fitness,
        descriptors: Descriptor,
        extra_scores: ExtraScores,
    ) -> bool:
        """Choose between ES and RL update with probability 0.5.

        Params:
            emitter_state
            fitnesses: the fitnesses of the offspring
            descriptors: the descriptors of the offspring
            extra_scores: the extra scores of the offspring

        Returns:
            True if ES update False else
        """
        cond = emitter_state.metrics.es_updates <= emitter_state.metrics.rl_updates

        return cond


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
        #                          p=jnp.array([self.config.es_proba, 1-self.config.es_proba]))

        # Do RL if the ES has done more steps than RL
        # cond = emitter_state.metrics.es_updates <= emitter_state.metrics.rl_updates

        cond = self.choose_es_update(emitter_state, fitnesses, descriptors, extra_scores)

        emitter_state = jax.lax.cond(
            cond,
            self.es_state_update,
            self.rl_state_update,
            # Arguments to the two functions:
            emitter_state, 
            repertoire, 
            genotypes, 
            fitnesses, 
            descriptors, 
            extra_scores
        )

        key, emitter_state = emitter_state.get_key()

        actor_genome = emitter_state.rl_state.actor_params
        actor_fitness, _ = self.multi_eval(actor_genome, key)

        # es_center = emitter_state.es_state.offspring
        # es_center_fitness, _ = self.multi_eval(genotypes, key)
        center_fitness = jnp.mean(fitnesses)

        metrics = emitter_state.metrics.replace(
            actor_fitness=actor_fitness,
            center_fitness=center_fitness,
        )

        emitter_state = emitter_state.replace(
            metrics=metrics,
        )

        return emitter_state
    
    @partial(
        jax.jit,
        static_argnames=("self",),
    )
    def multi_eval(
        self, 
        genome: Genotype,
        random_key: RNGKey,
    ) -> Tuple[Genotype, RNGKey]:
        """Evaluate the genome multiple times and return the mean fitness.

        Params:
            genome: the genome to evaluate
            scoring_fn: the scoring function to use

        Returns:
            the mean fitness
            the new random key
        """

        genotypes = jax.tree_util.tree_map(
            lambda x: jnp.repeat(
                jnp.expand_dims(x, axis=0), 
                48, axis=0), 
            genome
            )
        scoring_fn = self.es_emitter._scoring_fn
        fitnesses, descriptors, extra_scores, random_key = scoring_fn(
            genotypes, random_key
        )
        return jnp.mean(fitnesses), random_key
        
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

        assert jax.tree_util.tree_leaves(genotypes)[0].shape[0] == 1, (
            "ERROR: ES generates 1 offspring per generation, "
            + "batch_size should be 1, the inputed batch has size:"
            + str(jax.tree_util.tree_leaves(genotypes)[0].shape[0])
        )

        # Updating novelty archive
        novelty_archive = emitter_state.es_state.novelty_archive.update(descriptors)

        # Define scores for es process
        def scores(fitnesses: Fitness, descriptors: Descriptor) -> jnp.ndarray:
            if self.es_emitter._config.nses_emitter:
                return novelty_archive.novelty(
                    descriptors, self.es_emitter._config.novelty_nearest_neighbors
                )
            else:
                return fitnesses

        # Run es process
        offspring, optimizer_state, random_key, extra_scores = self.es_emitter._es_emitter(
            parent=genotypes,
            optimizer_state=emitter_state.es_state.optimizer_state,
            random_key=random_key,
            scores_fn=scores,
            actor=emitter_state.rl_state.actor_params,
        )

        center_fitness = jnp.mean(fitnesses)
        metrics = emitter_state.metrics.replace(
            center_fitness=center_fitness,
            evaluations=emitter_state.metrics.evaluations + self._config.sample_number,
        )

        if "population_fitness" in extra_scores:
            pop_mean = jnp.mean(extra_scores["population_fitness"])
            pop_std = jnp.std(extra_scores["population_fitness"])
            pop_min = jnp.min(extra_scores["population_fitness"])
            pop_max = jnp.max(extra_scores["population_fitness"]) 
            metrics = metrics.replace(
                pop_mean=pop_mean,
                pop_std=pop_std,
                pop_min=pop_min,
                pop_max=pop_max,
            )

        # Log sigma if using cmaes
        if isinstance(emitter_state.es_state.optimizer_state, CMAESState):
            metrics = metrics.replace(
                sigma=emitter_state.es_state.optimizer_state.sigma,
            )

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
        )

        rl_state = self.rl_emitter.state_update(
            emitter_state = rl_state,
            repertoire = repertoire,
            genotypes = genotypes,
            fitnesses = fitnesses,
            descriptors = descriptors,
            extra_scores = extra_scores,
        )

        metrics = ESMetrics(
            es_updates=emitter_state.metrics.es_updates + 1,
            rl_updates=emitter_state.metrics.rl_updates,
            evaluations=emitter_state.metrics.evaluations + self.config.es_config.sample_number,
        )
        # Share random key between ES and RL emitters

        state = ESRLEmitterState(es_state, rl_state, metrics)
        state = state.set_key(random_key)
        return state
    
    @partial(
        jax.jit,
        static_argnames=("self",),
    )
    def rl_state_update(
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

        assert jax.tree_util.tree_leaves(genotypes)[0].shape[0] == 1, (
            "ERROR: ES generates 1 offspring per generation, "
            + "batch_size should be 1, the inputed batch has size:"
            + str(jax.tree_util.tree_leaves(genotypes)[0].shape[0])
        )

        # Updating novelty archive
        novelty_archive = emitter_state.es_state.novelty_archive.update(descriptors)
            
        # Update QPG emitter to train RL agent
        # Update random key
        rl_state = emitter_state.rl_state.replace(
            random_key=random_key,
        )

        new_rl_state = self.rl_emitter.state_update(
            emitter_state = rl_state,
            repertoire = repertoire,
            genotypes = genotypes,
            fitnesses = fitnesses,
            descriptors = descriptors,
            extra_scores = extra_scores,
        )

        offspring = self.rl_emitter.emit_pg(
            emitter_state = new_rl_state,
            parents=genotypes,
        )

        random_key = new_rl_state.random_key

        # Update ES emitter state
        es_state = emitter_state.es_state.replace(
            offspring=offspring,
            optimizer_state=emitter_state.es_state.optimizer_state,
            random_key=random_key,
            novelty_archive=novelty_archive,
            generation_count=emitter_state.es_state.generation_count + 1,
        )

        metrics = ESMetrics(
            es_updates=emitter_state.metrics.es_updates,
            rl_updates=emitter_state.metrics.rl_updates + 1,
            evaluations=emitter_state.metrics.evaluations,
        )

        state = ESRLEmitterState(es_state, rl_state, metrics)
        state = state.set_key(random_key)
        return state

