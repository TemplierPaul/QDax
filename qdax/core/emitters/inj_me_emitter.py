# PGA-ME but with no PG mutation: only actor injection in standard ME

from dataclasses import dataclass
from functools import partial
from typing import Any, Optional, Tuple, Callable

import flax.linen as nn
import jax
import optax
from jax import numpy as jnp

from qdax.core.emitters.multi_emitter import MultiEmitter, MultiEmitterState
from qdax.core.emitters.qpg_emitter import QualityPGConfig, QualityPGEmitter, QualityPGEmitterState
from qdax.core.emitters.pga_me_emitter import PGAMEConfig, PGAMEEmitter

from qdax.core.emitters.standard_emitters import MixingEmitter
from qdax.environments.base_wrappers import QDEnv
from qdax.types import Params, RNGKey

from qdax.core.containers.repertoire import Repertoire
from qdax.core.emitters.emitter import Emitter, EmitterState
from qdax.core.neuroevolution.buffers.buffer import QDTransition, ReplayBuffer
from qdax.core.neuroevolution.losses.td3_loss import make_td3_loss_fn
from qdax.core.neuroevolution.networks.networks import QModule
from qdax.environments.base_wrappers import QDEnv
from qdax.types import Descriptor, ExtraScores, Fitness, Genotype, Params, RNGKey

from qdax.core.containers.age_repertoire import AgeMapElitesRepertoire, Age

class QualityInjectionEmitter(QualityPGEmitter):
    @partial(
        jax.jit,
        static_argnames=("self",),
    )
    def emit(
        self,
        repertoire: AgeMapElitesRepertoire,
        emitter_state: QualityPGEmitterState,
        random_key: RNGKey,
    ) -> Tuple[Genotype, Age, RNGKey]:
        """Do a step of PG emission.

        Args:
            repertoire: the current repertoire of genotypes
            emitter_state: the state of the emitter used
            random_key: a random key

        Returns:
            A batch of offspring, the new emitter state and a new key.
        """
        # get the actor (greedy actor)
        offspring_actor = self.emit_actor(emitter_state)

        # add dimension for concatenation
        offspring_actor = jax.tree_util.tree_map(
            lambda x: jnp.expand_dims(x, axis=0), offspring_actor
        )

        age = jnp.zeros((1,))

        # print("Injection age shape: ", age.shape)

        return offspring_actor, age, random_key

class AgeMixingEmitter(MixingEmitter):
    @partial(
        jax.jit,
        static_argnames=("self",),
    )
    def emit(
        self,
        repertoire: Repertoire,
        emitter_state: Optional[EmitterState],
        random_key: RNGKey,
    ) -> Tuple[Genotype, Age, RNGKey]:
        """
        Emitter that performs both mutation and variation. Two batches of
        variation_percentage * batch_size genotypes are sampled in the repertoire,
        copied and cross-over to obtain new offsprings. One batch of
        (1.0 - variation_percentage) * batch_size genotypes are sampled in the
        repertoire, copied and mutated.

        Note: this emitter has no state. A fake none state must be added
        through a function redefinition to make this emitter usable with MAP-Elites.

        Params:
            repertoire: the MAP-Elites repertoire to sample from
            emitter_state: void
            random_key: a jax PRNG random key

        Returns:
            a batch of offsprings
            a new jax PRNG key
        """
        n_variation = int(self._batch_size * self._variation_percentage)
        n_mutation = self._batch_size - n_variation

        if n_variation > 0:
            x1, ages_variation, random_key = repertoire.sample(random_key, n_variation)
            x2, _, random_key = repertoire.sample(random_key, n_variation)

            # print("Variation GA ages shape: ", ages_variation.shape)

            x_variation, random_key = self._variation_fn(x1, x2, random_key)

        if n_mutation > 0:
            x1, ages_mutation, random_key = repertoire.sample(random_key, n_mutation)

            # print("Mutation GA ages shape: ", ages_mutation.shape)

            x_mutation, random_key = self._mutation_fn(x1, random_key)

        if n_variation == 0:
            genotypes = x_mutation
            ages = ages_mutation
        elif n_mutation == 0:
            genotypes = x_variation
            ages = ages_variation
        else:
            genotypes = jax.tree_util.tree_map(
                lambda x_1, x_2: jnp.concatenate([x_1, x_2], axis=0),
                x_variation,
                x_mutation,
            )
            ages = jnp.concatenate([ages_variation, ages_mutation], axis=0)

        # Increment age
        ages += 1

        # print("GA ages shape: ", ages.shape)

        return genotypes, ages, random_key

# Copied from qdax.core.emitters.pga_me_emitter
class InjectionMEEmitter(MultiEmitter):
    def __init__(
        self,
        config: PGAMEConfig,
        policy_network: nn.Module,
        env: QDEnv,
        variation_fn: Callable[[Params, Params, RNGKey], Tuple[Params, RNGKey]],
    ) -> None:
        
        self._config = config
        self._policy_network = policy_network
        self._env = env
        self._variation_fn = variation_fn

        ga_batch_size = config.env_batch_size - 1
        qpg_batch_size = 1

        qpg_config = QualityPGConfig(
            env_batch_size=qpg_batch_size,
            num_critic_training_steps=config.num_critic_training_steps,
            num_pg_training_steps=config.num_pg_training_steps,
            replay_buffer_size=config.replay_buffer_size,
            critic_hidden_layer_size=config.critic_hidden_layer_size,
            critic_learning_rate=config.critic_learning_rate,
            actor_learning_rate=config.greedy_learning_rate,
            policy_learning_rate=config.policy_learning_rate,
            noise_clip=config.noise_clip,
            policy_noise=config.policy_noise,
            discount=config.discount,
            reward_scaling=config.reward_scaling,
            batch_size=config.batch_size,
            soft_tau_update=config.soft_tau_update,
            policy_delay=config.policy_delay,
        )

        # define the quality emitter
        q_emitter = QualityInjectionEmitter(
            config=qpg_config, policy_network=policy_network, env=env
        )

        # define the GA emitter
        ga_emitter = AgeMixingEmitter(
            mutation_fn=lambda x, r: (x, r),
            variation_fn=variation_fn,
            variation_percentage=1.0,
            batch_size=ga_batch_size,
        )

        super().__init__(emitters=(q_emitter, ga_emitter))

    @partial(jax.jit, static_argnames=("self",))
    def emit(
        self,
        repertoire: Optional[AgeMapElitesRepertoire],
        emitter_state: Optional[MultiEmitterState],
        random_key: RNGKey,
    ) -> Tuple[Genotype, Age,  RNGKey]:
        """Emit new population. Use all the sub emitters to emit subpopulation
        and gather them.

        Args:
            repertoire: a repertoire of genotypes.
            emitter_state: the current state of the emitter.
            random_key: key for random operations.

        Returns:
            Offsprings and a new random key.
        """
        assert emitter_state is not None
        assert len(emitter_state.emitter_states) == len(self.emitters)

        # prepare subkeys for each sub emitter
        random_key, subkey = jax.random.split(random_key)
        subkeys = jax.random.split(subkey, len(self.emitters))

        # emit from all emitters and gather offsprings
        all_offsprings = []
        all_ages = []
        for emitter, sub_emitter_state, subkey_emitter in zip(
            self.emitters,
            emitter_state.emitter_states,
            subkeys,
        ):
            genotype, ages, _ = emitter.emit(repertoire, sub_emitter_state, subkey_emitter)
            batch_size = jax.tree_util.tree_leaves(genotype)[0].shape[0]
            assert batch_size == emitter.batch_size
            all_offsprings.append(genotype)
            all_ages.append(ages)

        # concatenate offsprings together
        offsprings = jax.tree_util.tree_map(
            lambda *x: jnp.concatenate(x, axis=0), *all_offsprings
        )
        ages = jnp.concatenate(all_ages, axis=0)

        # print("MultiEmitter ages shape: ", ages.shape)

        return offsprings, ages, random_key