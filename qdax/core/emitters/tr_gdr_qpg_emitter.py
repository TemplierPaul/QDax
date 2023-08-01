from dataclasses import dataclass
from functools import partial
from typing import Any, Optional, Tuple

import flax.linen as nn
import jax
import optax
from jax import numpy as jnp

from qdax.core.containers.repertoire import Repertoire
from qdax.core.emitters.emitter import Emitter, EmitterState
from qdax.core.neuroevolution.buffers.buffer import QDTransition, ReplayBuffer
from qdax.core.neuroevolution.losses.td3_loss import make_td3_loss_fn
from qdax.core.neuroevolution.losses.elastic_pg_loss import elastic_td3_loss_fn
from qdax.core.neuroevolution.losses.tr_gdr_loss import tr_gdr_loss_fn
from qdax.core.neuroevolution.networks.networks import QModule
from qdax.environments.base_wrappers import QDEnv
from qdax.types import Descriptor, ExtraScores, Fitness, Genotype, Params, RNGKey

from qdax.core.emitters.qpg_emitter import QualityPGConfig, QualityPGEmitterState, QualityPGEmitter
from qdax.core.emitters.custom_qpg_emitter import CustomQualityPGEmitterState, CustomQualityPGEmitter

from qdax.core.emitters.vanilla_es_emitter import flatten_genotype
from jax.tree_util import tree_flatten, tree_unflatten, tree_map
from flax.struct import PyTreeNode

# from jax.debug import print as jprint

@dataclass
class TRGDRConfig:
    tr_gdr_alpha: float = 0.99
    tr_gdr_lambda_lr: float = 1e-3
    tr_gdr_scaling: float = 1

class TRGDRState(PyTreeNode):
    tr_gdr_lambda_opt_state: optax.OptState
    tr_gdr_d: float = 0.0
    tr_gdr_lambda: float = 0.0

class TRGDREmitterState(CustomQualityPGEmitterState):
    gdr_state: TRGDRState

class TRGDREmitter(CustomQualityPGEmitter):
    """
    A policy gradient emitter used to implement the Policy Gradient Assisted MAP-Elites
    (PGA-Map-Elites) algorithm, with L2 regularization on the actor network to keep it close to the ES distribution.
    """

    def __init__(
        self,
        config: QualityPGConfig,
        gdr_config: TRGDRConfig,
        policy_network: nn.Module,
        env: QDEnv,
    ) -> None:
        self._config = config
        self._gdr_config = gdr_config
        self._env = env
        self._policy_network = policy_network
        self.policy_fn = policy_network.apply

        # Init Critics
        critic_network = QModule(
            n_critics=2, hidden_layer_sizes=self._config.critic_hidden_layer_size
        )
        self._critic_network = critic_network
        self.critic_fn = critic_network.apply

        self._policy_loss_fn, self._critic_loss_fn, self._lambda_loss_fn = tr_gdr_loss_fn(
            policy_fn=policy_network.apply,
            critic_fn=critic_network.apply,
            reward_scaling=self._config.reward_scaling,
            discount=self._config.discount,
            noise_clip=self._config.noise_clip,
            policy_noise=self._config.policy_noise,
        )

        # Init optimizers
        self._actor_optimizer = optax.adam(
            learning_rate=self._config.actor_learning_rate
        )
        self._critic_optimizer = optax.adam(
            learning_rate=self._config.critic_learning_rate
        )
        self._policies_optimizer = optax.adam(
            learning_rate=self._config.policy_learning_rate
        )
        self._lambda_optimizer = optax.adam(
            learning_rate=self._gdr_config.tr_gdr_lambda_lr
        )

        self.critic_tree_def = None
        self.critic_layer_sizes = None
        self.critic_split_indices = None

    @property
    def config_string(self):
        s = f"TD3 {self._config.num_critic_training_steps} - PG {self._config.num_pg_training_steps} "
        s += f"- lr A {self._config.actor_learning_rate} / C {self._config.critic_learning_rate}"
        s += f"- TR-GDR {self._gdr_config.tr_gdr_alpha}"
        return s
    
    def init(
        self, init_genotypes: Genotype, random_key: RNGKey
    ) -> Tuple[TRGDRState, RNGKey]:
        """Initializes the emitter state.

        Args:
            init_genotypes: The initial population.
            random_key: A random key.

        Returns:
            The initial state of the PGAMEEmitter, a new random key.
        """

        observation_size = self._env.observation_size
        action_size = self._env.action_size
        descriptor_size = self._env.state_descriptor_length

        # Initialise critic, greedy actor and population
        random_key, subkey = jax.random.split(random_key)
        fake_obs = jnp.zeros(shape=(observation_size,))
        fake_action = jnp.zeros(shape=(action_size,))
        critic_params = self._critic_network.init(
            subkey, obs=fake_obs, actions=fake_action
        )
        target_critic_params = jax.tree_util.tree_map(lambda x: x, critic_params)

        actor_params = jax.tree_util.tree_map(lambda x: x[0], init_genotypes)
        target_actor_params = jax.tree_util.tree_map(lambda x: x[0], init_genotypes)

        # Prepare init optimizer states
        critic_optimizer_state = self._critic_optimizer.init(critic_params)
        actor_optimizer_state = self._actor_optimizer.init(actor_params)

        # Initialize replay buffer
        self.get_dummy_batch = lambda p, n: QDTransition.dummy_batch(
            observation_dim=observation_size,
            action_dim=action_size,
            descriptor_dim=descriptor_size,
            population=p,
            length=n,
        )
        
        dummy_transition = QDTransition.init_dummy(
            observation_dim=observation_size,
            action_dim=action_size,
            descriptor_dim=descriptor_size,
        )

        replay_buffer = ReplayBuffer.init(
            buffer_size=self._config.replay_buffer_size, transition=dummy_transition
        )
        # print("Replay buffer position", replay_buffer.current_position)

        # Trust region GDR

        flat_variables, tree_def = tree_flatten(actor_params)
        vect = jnp.concatenate([jnp.ravel(x) for x in flat_variables])
        dim = len(vect)

        from scipy.stats import chi
        quant = self._gdr_config.tr_gdr_alpha
        sigma = self._gdr_config.tr_gdr_scaling

        tr_gdr_d = chi.ppf(quant, dim, scale=sigma)
        print(f"TR-GDR d = {tr_gdr_d}")
        tr_gdr_lambda = 0.0
        tr_gdr_lambda_opt_state = self._lambda_optimizer.init(tr_gdr_lambda)

        # Initial training state
        random_key, subkey = jax.random.split(random_key)

        flat_variables, tree_def = tree_flatten(critic_params)
        self.critic_layer_shapes = [x.shape for x in flat_variables]
        # print("critic layer shapes", self.critic_layer_shapes)

        # vect = jnp.concatenate([jnp.ravel(x) for x in flat_variables])
        sizes = [x.size for x in flat_variables]
        sizes = jnp.array(sizes)

        # print("critic sizes", sizes)

        self.critic_tree_def = tree_def
        self.critic_layer_sizes = sizes.tolist()
        # print("layer_sizes", self.critic_layer_sizes)
        self.critic_split_indices = jnp.cumsum(jnp.array(self.critic_layer_sizes))[:-1].tolist()
        # print("split_indices", self.split_indices)

        gdr_state = TRGDRState(
            tr_gdr_d=tr_gdr_d,
            tr_gdr_lambda=tr_gdr_lambda,
            tr_gdr_lambda_opt_state=tr_gdr_lambda_opt_state,
        )
        
        emitter_state = TRGDREmitterState(
            critic_params=critic_params,
            critic_optimizer_state=critic_optimizer_state,
            actor_params=actor_params,
            actor_opt_state=actor_optimizer_state,
            target_critic_params=target_critic_params,
            target_actor_params=target_actor_params,
            random_key=subkey,
            steps=jnp.array(0),
            replay_buffer=replay_buffer,
            es_center=init_genotypes,
            gdr_state=gdr_state,                        
        )

        return emitter_state, random_key
    
    @partial(jax.jit, static_argnames=("self",))
    def _update_actor(
        self,
        actor_params: Params,
        actor_opt_state: optax.OptState,
        target_actor_params: Params,
        critic_params: Params,
        transitions: QDTransition,
        es_center: Params,
        gdr_state: TRGDRState,
    ) -> Tuple[optax.OptState, Params, Params]:

        # Update greedy actor
        policy_loss, policy_gradient = jax.value_and_grad(self._policy_loss_fn)(
            actor_params,
            critic_params,
            transitions,
            es_center=es_center,
            gdr_state=gdr_state,
        )
        
        (
            policy_updates,
            actor_optimizer_state,
        ) = self._actor_optimizer.update(policy_gradient, actor_opt_state)
        new_actor_params = optax.apply_updates(actor_params, policy_updates)

        # Update trust region GDR lambda
        # lambda_loss, lambda_gradient = jax.value_and_grad(self._lambda_loss_fn)(
        #     gdr_state.tr_gdr_lambda,
        #     actor_params,
        #     es_center=es_center,
        #     gdr_state=gdr_state,
        # )
        # (
        #     lambda_updates,
        #     lambda_optimizer_state,
        # ) = self._lambda_optimizer.update(lambda_gradient, gdr_state.tr_gdr_lambda_opt_state)   
        # tr_gdr_lambda = optax.apply_updates(gdr_state.tr_gdr_lambda, lambda_updates)
        
        # Get square vector
        dist = jax.tree_util.tree_map(
            lambda x, y: ((x - y) ** 2), 
            actor_params,
            es_center
        )

        # Sum it
        dist_sum = jax.tree_util.tree_reduce(
            lambda x, y: x.sum() + y.sum(), 
            dist
        )
        # Square root to get distance
        g = jnp.sqrt(dist_sum) - gdr_state.tr_gdr_d

        # detach
        g = jax.lax.stop_gradient(g)
        tr_gdr_lambda = g

        # Update lambda
        # tr_gdr_lambda = gdr_state.tr_gdr_lambda + self._gdr_config.tr_gdr_lambda_lr * g
        lambda_optimizer_state = gdr_state.tr_gdr_lambda_opt_state

        # Clip at 0
        # tr_gdr_lambda = jnp.clip(tr_gdr_lambda, a_min=0.0, a_max=None)

        actor_params = new_actor_params
        
        gdr_state = gdr_state.replace(
            tr_gdr_lambda=tr_gdr_lambda, 
            tr_gdr_lambda_opt_state=lambda_optimizer_state
            )

        # Soft update of target greedy actor
        target_actor_params = jax.tree_map(
            lambda x1, x2: (1.0 - self._config.soft_tau_update) * x1
            + self._config.soft_tau_update * x2,
            target_actor_params,
            actor_params,
        )

        return (
            actor_optimizer_state,
            actor_params,
            target_actor_params,
            gdr_state,
            )
    
    @partial(jax.jit, static_argnames=("self",))
    def _train_critics(
        self, emitter_state: CustomQualityPGEmitterState
    ) -> CustomQualityPGEmitterState:
        """Apply one gradient step to critics and to the greedy actor
        (contained in carry in training_state), then soft update target critics
        and target actor.

        Those updates are very similar to those made in TD3.

        Args:
            emitter_state: actual emitter state

        Returns:
            New emitter state where the critic and the greedy actor have been
            updated. Optimizer states have also been updated in the process.
        """

        # Sample a batch of transitions in the buffer
        random_key = emitter_state.random_key
        replay_buffer = emitter_state.replay_buffer
        transitions, random_key = replay_buffer.sample(
            random_key, sample_size=self._config.batch_size
        )
        
        # es_center = emitter_state.es_center
        es_center = jax.tree_util.tree_map(lambda x: x[0], emitter_state.es_center)

        # Update Critic
        (
            critic_optimizer_state,
            critic_params,
            target_critic_params,
            random_key,
        ) = self._update_critic(
            critic_params=emitter_state.critic_params,
            target_critic_params=emitter_state.target_critic_params,
            target_actor_params=es_center, # ES center as target policy
            critic_optimizer_state=emitter_state.critic_optimizer_state,
            transitions=transitions,
            random_key=random_key,
        )

        # Update greedy actor
        (actor_optimizer_state, actor_params, target_actor_params, gdr_state,) = jax.lax.cond(
            emitter_state.steps % self._config.policy_delay == 0,
            lambda x: self._update_actor(*x),
            lambda _: (
                emitter_state.actor_opt_state,
                emitter_state.actor_params,
                emitter_state.target_actor_params,
                emitter_state.gdr_state, 
            ),
            operand=(
                emitter_state.actor_params,
                emitter_state.actor_opt_state,
                emitter_state.target_actor_params,
                emitter_state.critic_params,
                transitions,
                es_center,
                emitter_state.gdr_state, 
            ),
        )

        # Create new training state
        new_emitter_state = emitter_state.replace(
            critic_params=critic_params,
            critic_optimizer_state=critic_optimizer_state,
            actor_params=actor_params,
            actor_opt_state=actor_optimizer_state,
            target_critic_params=target_critic_params,
            target_actor_params=target_actor_params,
            random_key=random_key,
            steps=emitter_state.steps + 1,
            replay_buffer=replay_buffer,
            gdr_state=gdr_state,
        )

        return new_emitter_state  # type: ignore