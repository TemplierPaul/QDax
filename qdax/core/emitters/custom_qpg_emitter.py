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
from qdax.core.neuroevolution.networks.networks import QModule
from qdax.environments.base_wrappers import QDEnv
from qdax.types import Descriptor, ExtraScores, Fitness, Genotype, Params, RNGKey

from qdax.core.emitters.qpg_emitter import QualityPGConfig, QualityPGEmitterState, QualityPGEmitter

from qdax.core.emitters.vanilla_es_emitter import flatten_genotype
from jax.tree_util import tree_flatten, tree_unflatten, tree_map

@dataclass
class CustomQualityPGConfig(QualityPGConfig):
    elastic_pull: float = 0.01
    surrogate_batch: int = 1024

class CustomQualityPGEmitterState(QualityPGEmitterState):
    es_center: Params

    def save(self, path):
        """Saves the state to a file."""
        flat_genotypes = flatten_genotype(self.actor_params)
        jnp.save(path + "_actor.npy", flat_genotypes)
        print("Saved actor to " + path + "_actor.npy")

        flat_genotypes = flatten_genotype(self.critic_params)
        jnp.save(path + "_critic.npy", flat_genotypes)
        print("Saved critic to " + path + "_critic.npy")


class CustomQualityPGEmitter(Emitter):
    """
    A policy gradient emitter used to implement the Policy Gradient Assisted MAP-Elites
    (PGA-Map-Elites) algorithm, with L2 regularization on the actor network to keep it close to the ES distribution.
    """

    def __init__(
        self,
        config: QualityPGConfig,
        policy_network: nn.Module,
        env: QDEnv,
    ) -> None:
        self._config = config
        self._env = env
        self._policy_network = policy_network
        self.policy_fn = policy_network.apply

        # Init Critics
        critic_network = QModule(
            n_critics=2, hidden_layer_sizes=self._config.critic_hidden_layer_size
        )
        self._critic_network = critic_network
        self.critic_fn = critic_network.apply

        # Set up the losses and optimizers - return the opt states
        self._policy_loss_fn, self._critic_loss_fn = elastic_td3_loss_fn(
            policy_fn=policy_network.apply,
            critic_fn=critic_network.apply,
            reward_scaling=self._config.reward_scaling,
            discount=self._config.discount,
            noise_clip=self._config.noise_clip,
            policy_noise=self._config.policy_noise,
            elastic_pull=self._config.elastic_pull,
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

        self.critic_tree_def = None
        self.critic_layer_sizes = None
        self.critic_split_indices = None

    @property
    def config_string(self):
        s = f"TD3 {self._config.num_critic_training_steps} - PG {self._config.num_pg_training_steps} "
        if self._config.elastic_pull > 0:
            s += f"- \u03B5 {self._config.elastic_pull}" # \u03B5 is epsilon
        return s

    @property
    def batch_size(self) -> int:
        """
        Returns:
            the batch size emitted by the emitter.
        """
        return self._config.env_batch_size

    @property
    def use_all_data(self) -> bool:
        """Whether to use all data or not when used along other emitters.

        QualityPGEmitter uses the transitions from the genotypes that were generated
        by other emitters.
        """
        return True

    def init(
        self, init_genotypes: Genotype, random_key: RNGKey
    ) -> Tuple[CustomQualityPGEmitterState, RNGKey]:
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

        # Initial training state
        random_key, subkey = jax.random.split(random_key)
        emitter_state = CustomQualityPGEmitterState(
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
        )

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

        return emitter_state, random_key
    
    @partial(
        jax.jit,
        static_argnames=("self",),
    )
    def flatten_critic(self, network):
        flat_variables, _ = tree_flatten(network)
        # print("Flatten", flat_variables)
        vect = jnp.concatenate([jnp.ravel(x) for x in flat_variables])
        return vect
    

    @partial(
        jax.jit,
        static_argnames=("self",),
    )
    def unflatten_critic(self, vect):
        """Unflatten a vector of floats into a network"""
        # print("Unflatten", vect.shape)
        split_genome = jnp.split(vect, self.critic_split_indices)
        # Reshape to the original shape
        split_genome = [x.reshape(s) for x, s in zip(split_genome, self.critic_layer_shapes)]

        # Unflatten the tree
        new_net = tree_unflatten(self.critic_tree_def, split_genome)
        return new_net

    @partial(
        jax.jit,
        static_argnames=("self",),
    )
    def emit(
        self,
        repertoire: Repertoire,
        emitter_state: CustomQualityPGEmitterState,
        random_key: RNGKey,
    ) -> Tuple[Genotype, RNGKey]:
        """Do a step of PG emission.

        Args:
            repertoire: the current repertoire of genotypes
            emitter_state: the state of the emitter used
            random_key: a random key

        Returns:
            A batch of offspring, the new emitter state and a new key.
        """

        batch_size = self._config.env_batch_size

        # sample parents
        mutation_pg_batch_size = int(batch_size - 1)
        parents, random_key = repertoire.sample(random_key, mutation_pg_batch_size)

        # apply the pg mutation
        offsprings_pg = self.emit_pg(emitter_state, parents)

        # get the actor (greedy actor)
        offspring_actor = self.emit_actor(emitter_state)

        # add dimension for concatenation
        offspring_actor = jax.tree_util.tree_map(
            lambda x: jnp.expand_dims(x, axis=0), offspring_actor
        )

        # gather offspring
        genotypes = jax.tree_util.tree_map(
            lambda x, y: jnp.concatenate([x, y], axis=0),
            offsprings_pg,
            offspring_actor,
        )

        return genotypes, random_key

    @partial(
        jax.jit,
        static_argnames=("self",),
    )
    def emit_pg(
        self, emitter_state: CustomQualityPGEmitterState, parents: Genotype
    ) -> Genotype:
        """Emit the offsprings generated through pg mutation.

        Args:
            emitter_state: current emitter state, contains critic and
                replay buffer.
            parents: the parents selected to be applied gradients in order
                to mutate towards better performance.

        Returns:
            A new set of offsprings.
        """
        mutation_fn = partial(
            self._mutation_function_pg,
            emitter_state=emitter_state,
        )
        offsprings = jax.vmap(mutation_fn)(parents)

        return offsprings

    @partial(
        jax.jit,
        static_argnames=("self",),
    )
    def emit_actor(self, emitter_state: CustomQualityPGEmitterState) -> Genotype:
        """Emit the greedy actor.

        Simply needs to be retrieved from the emitter state.

        Args:
            emitter_state: the current emitter state, it stores the
                greedy actor.

        Returns:
            The parameters of the actor.
        """
        return emitter_state.actor_params

    @partial(jax.jit, static_argnames=("self",))
    def state_update(
        self,
        emitter_state: CustomQualityPGEmitterState,
        repertoire: Optional[Repertoire],
        genotypes: Optional[Genotype],
        fitnesses: Optional[Fitness],
        descriptors: Optional[Descriptor],
        extra_scores: ExtraScores,
    ) -> CustomQualityPGEmitterState:
        """This function gives an opportunity to update the emitter state
        after the genotypes have been scored.

        Here it is used to fill the Replay Buffer with the transitions
        from the scoring of the genotypes, and then the training of the
        critic/actor happens. Hence the params of critic/actor are updated,
        as well as their optimizer states.

        Args:
            emitter_state: current emitter state.
            repertoire: the current genotypes repertoire
            genotypes: unused here - but compulsory in the signature.
            fitnesses: unused here - but compulsory in the signature.
            descriptors: unused here - but compulsory in the signature.
            extra_scores: extra information coming from the scoring function,
                this contains the transitions added to the replay buffer.

        Returns:
            New emitter state where the replay buffer has been filled with
            the new experienced transitions.
        """
        # get the transitions out of the dictionary
        assert "transitions" in extra_scores.keys(), "Missing transitions or wrong key"
        transitions = extra_scores["transitions"]

        # add transitions in the replay buffer
        replay_buffer = emitter_state.replay_buffer.insert(transitions)
        emitter_state = emitter_state.replace(replay_buffer=replay_buffer)

        def scan_train_critics(
            carry: CustomQualityPGEmitterState, unused: Any
        ) -> Tuple[CustomQualityPGEmitterState, Any]:
            emitter_state = carry
            new_emitter_state = self._train_critics(emitter_state)
            return new_emitter_state, ()

        # Train critics and greedy actor
        emitter_state, _ = jax.lax.scan(
            scan_train_critics,
            emitter_state,
            (),
            length=self._config.num_critic_training_steps,
        )

        return emitter_state  # type: ignore

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
            target_actor_params=emitter_state.target_actor_params,
            critic_optimizer_state=emitter_state.critic_optimizer_state,
            transitions=transitions,
            random_key=random_key,
        )

        # Update greedy actor
        (actor_optimizer_state, actor_params, target_actor_params,) = jax.lax.cond(
            emitter_state.steps % self._config.policy_delay == 0,
            lambda x: self._update_actor(*x),
            lambda _: (
                emitter_state.actor_opt_state,
                emitter_state.actor_params,
                emitter_state.target_actor_params,
            ),
            operand=(
                emitter_state.actor_params,
                emitter_state.actor_opt_state,
                emitter_state.target_actor_params,
                emitter_state.critic_params,
                transitions,
                es_center,
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
        )

        return new_emitter_state  # type: ignore

    @partial(jax.jit, static_argnames=("self",))
    def _update_critic(
        self,
        critic_params: Params,
        target_critic_params: Params,
        target_actor_params: Params,
        critic_optimizer_state: Params,
        transitions: QDTransition,
        random_key: RNGKey,
    ) -> Tuple[Params, Params, Params, RNGKey]:

        # compute loss and gradients
        random_key, subkey = jax.random.split(random_key)
        critic_loss, critic_gradient = jax.value_and_grad(self._critic_loss_fn)(
            critic_params,
            target_actor_params,
            target_critic_params,
            transitions,
            subkey,
        )
        critic_updates, critic_optimizer_state = self._critic_optimizer.update(
            critic_gradient, critic_optimizer_state
        )

        # update critic
        critic_params = optax.apply_updates(critic_params, critic_updates)

        # Soft update of target critic network
        target_critic_params = jax.tree_map(
            lambda x1, x2: (1.0 - self._config.soft_tau_update) * x1
            + self._config.soft_tau_update * x2,
            target_critic_params,
            critic_params,
        )

        return critic_optimizer_state, critic_params, target_critic_params, random_key

    @partial(jax.jit, static_argnames=("self",))
    def _update_actor(
        self,
        actor_params: Params,
        actor_opt_state: optax.OptState,
        target_actor_params: Params,
        critic_params: Params,
        transitions: QDTransition,
        es_center: Params,
    ) -> Tuple[optax.OptState, Params, Params]:

        # Update greedy actor
        policy_loss, policy_gradient = jax.value_and_grad(self._policy_loss_fn)(
            actor_params,
            critic_params,
            transitions,
            es_center=es_center,
        )
        (
            policy_updates,
            actor_optimizer_state,
        ) = self._actor_optimizer.update(policy_gradient, actor_opt_state)
        actor_params = optax.apply_updates(actor_params, policy_updates)

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
        )

    @partial(jax.jit, static_argnames=("self",))
    def _mutation_function_pg(
        self,
        policy_params: Genotype,
        emitter_state: CustomQualityPGEmitterState,
    ) -> Genotype:
        """Apply pg mutation to a policy via multiple steps of gradient descent.
        First, update the rewards to be diversity rewards, then apply the gradient
        steps.

        Args:
            policy_params: a policy, supposed to be a differentiable neural
                network.
            emitter_state: the current state of the emitter, containing among others,
                the replay buffer, the critic.

        Returns:
            The updated params of the neural network.
        """

        # Define new policy optimizer state
        policy_optimizer_state = self._policies_optimizer.init(policy_params)

        def scan_train_policy(
            carry: Tuple[CustomQualityPGEmitterState, Genotype, optax.OptState],
            unused: Any,
        ) -> Tuple[Tuple[CustomQualityPGEmitterState, Genotype, optax.OptState], Any]:
            emitter_state, policy_params, policy_optimizer_state = carry
            (
                new_emitter_state,
                new_policy_params,
                new_policy_optimizer_state,
            ) = self._train_policy(
                emitter_state,
                policy_params,
                policy_optimizer_state,
            )
            return (
                new_emitter_state,
                new_policy_params,
                new_policy_optimizer_state,
            ), ()

        (emitter_state, policy_params, policy_optimizer_state,), _ = jax.lax.scan(
            scan_train_policy,
            (emitter_state, policy_params, policy_optimizer_state),
            (),
            length=self._config.num_pg_training_steps,
        )

        return policy_params

    @partial(jax.jit, static_argnames=("self",))
    def _train_policy(
        self,
        emitter_state: CustomQualityPGEmitterState,
        policy_params: Params,
        policy_optimizer_state: optax.OptState,
    ) -> Tuple[CustomQualityPGEmitterState, Params, optax.OptState]:
        """Apply one gradient step to a policy (called policy_params).

        Args:
            emitter_state: current state of the emitter.
            policy_params: parameters corresponding to the weights and bias of
                the neural network that defines the policy.

        Returns:
            The new emitter state and new params of the NN.
        """

        # Sample a batch of transitions in the buffer
        random_key = emitter_state.random_key
        replay_buffer = emitter_state.replay_buffer
        transitions, random_key = replay_buffer.sample(
            random_key, sample_size=self._config.batch_size
        )

        # update policy
        policy_optimizer_state, policy_params = self._update_policy(
            critic_params=emitter_state.critic_params,
            policy_optimizer_state=policy_optimizer_state,
            policy_params=policy_params,
            transitions=transitions,
        )

        # Create new training state
        new_emitter_state = emitter_state.replace(
            random_key=random_key,
            replay_buffer=replay_buffer,
        )

        return new_emitter_state, policy_params, policy_optimizer_state

    @partial(jax.jit, static_argnames=("self",))
    def _update_policy(
        self,
        critic_params: Params,
        policy_optimizer_state: optax.OptState,
        policy_params: Params,
        transitions: QDTransition,
    ) -> Tuple[optax.OptState, Params]:

        # compute loss
        _policy_loss, policy_gradient = jax.value_and_grad(self._policy_loss_fn)(
            policy_params,
            critic_params,
            transitions,
            es_center=policy_params, # No L2 regu on ES agents
        )
        # Compute gradient and update policies
        (
            policy_updates,
            policy_optimizer_state,
        ) = self._policies_optimizer.update(policy_gradient, policy_optimizer_state)
        policy_params = optax.apply_updates(policy_params, policy_updates)

        return policy_optimizer_state, policy_params

    @partial(
        jax.jit,
        static_argnames=("self",),
    )
    def surrogate_eval(
        self, 
        policy_params: Params,
        critic_params: Params,
        transitions: QDTransition,
    ) -> Fitness:
        """Evaluate the surrogate fitness of a genotype.

        Args:
            genotype: the genotype to evaluate.

        Returns:
            The surrogate fitness of the genotype.
        """
        action = self.policy_fn(policy_params, transitions.obs)
        q_value = self.critic_fn(
            critic_params, obs=transitions.obs, actions=action  # type: ignore
        )
        q1_action = jnp.take(q_value, jnp.asarray([0]), axis=-1)
        surrogate = jnp.mean(q1_action)
        return surrogate

