from qdax.tasks.brax_envs import (
    make_policy_network_play_step_fn_brax,
    reset_based_scoring_function_brax_envs,
)
from qdax.core.rl_es_parts.canonical_es import CanonicalESConfig, CanonicalESEmitter

import jax 
import jax.numpy as jnp
import functools

from qdax import environments

class BaseES:
    def __init__(self, env, config):
        self.config = config
        self.random_key = jax.random.PRNGKey(0)

        self.env = env

    def init(self, policy_network):

        random_key, subkey = jax.random.split(self.random_key)
        keys = jax.random.split(subkey, num=1)
        fake_batch = jnp.zeros(shape=(1, self.env.observation_size))
        init_variables = jax.vmap(policy_network.init)(keys, fake_batch)

        play_reset_fn = self.env.reset

        # Prepare the scoring function
        bd_extraction_fn = environments.behavior_descriptor_extractor[self.config["env_name"]]

        play_step_fn = make_policy_network_play_step_fn_brax(self.env, policy_network)

        self.es_scoring_fn = functools.partial(
            reset_based_scoring_function_brax_envs,
            episode_length=self.config["episode_length"],
            play_reset_fn=play_reset_fn,
            play_step_fn=play_step_fn,
            behavior_descriptor_extractor=bd_extraction_fn,
        )

        es_config = CanonicalESConfig(
            nses_emitter=False,
            sample_number=self.config["sample_number"],
            canonical_mu=int(self.config["sample_number"] / 2),
            sample_sigma=self.config["sigma"],
            actor_injection=False,
            nb_injections=0,
            episode_length=self.config["episode_length"],
            return_population=True,
        )

        self.emitter = CanonicalESEmitter(
            config=es_config,
            rollout_fn=self.es_scoring_fn,
            eval_fn=self.es_scoring_fn,
            total_generations=self.config["es_gens"],
            num_descriptors=self.env.behavior_descriptor_length,
        )

        es_emitter_state, random_key = self.emitter.init(
            init_genotypes=init_variables,
            random_key=random_key,
        )
        return es_emitter_state, random_key

    def scan_step(self, carry, unused):
        raise NotImplementedError
        
    def step(self, start, target_bd, es_emitter_state, random_key, n_steps=1):
        (offspring, target_bd, es_emitter_state, random_key), metrics = jax.lax.scan(
            self.scan_step,
            (start, target_bd, es_emitter_state, random_key),
            None,
            length=n_steps,
        )
        # Remove the batch dimension
        metrics = jax.tree_map(lambda x: x.squeeze(), metrics)
        return offspring, metrics, es_emitter_state, random_key
    
    def evaluate(self, genotype, random_key):
        offspring_fitnesses, offspring_descriptors, extra_scores, random_key = self.emitter._rollout_fn(
                genotype,
                random_key
            )
        return offspring_fitnesses, offspring_descriptors, extra_scores, random_key
        

class BD_ES(BaseES):
    def scan_step(self, carry, unused):
        start, target_bd, es_emitter_state, random_key = carry
        @jax.jit
        def bd_scores_fn(fitnesses, descriptors) -> jnp.ndarray:
            # minimize distance to target_bd
            return  - jnp.linalg.norm(descriptors - target_bd, axis=1)

        offspring, optimizer_state, random_key, extra_scores = self.emitter._es_emitter(
            parent = start,
            random_key = random_key,
            scores_fn = bd_scores_fn,
            optimizer_state=es_emitter_state.optimizer_state
        )
        # evaluate offspring
        offspring_fitnesses, offspring_descriptors, offspring_extra_scores, random_key = self.emitter._rollout_fn(
            offspring,
            random_key
        )

        metrics = {
            "genotype": offspring,
            "fitness": offspring_fitnesses,
            "descriptor": offspring_descriptors,
            "population_fitness": extra_scores["population_fitness"],
            "population_descriptors": extra_scores["population_descriptors"],
            "population_networks": extra_scores["population_networks"],
        }

        es_emitter_state = es_emitter_state.replace(
            optimizer_state=optimizer_state
        )

        return (offspring, target_bd, es_emitter_state, random_key), metrics

class Fitness_ES(BaseES):
    def scan_step(self, carry, unused):
        start, target_bd, es_emitter_state, random_key = carry
        @jax.jit
        def bd_scores_fn(fitnesses, descriptors) -> jnp.ndarray:
            # minimize distance to target_bd
            return  fitnesses

        offspring, optimizer_state, random_key, extra_scores = self.emitter._es_emitter(
            parent = start,
            random_key = random_key,
            scores_fn = bd_scores_fn,
            optimizer_state=es_emitter_state.optimizer_state
        )
        # evaluate offspring
        offspring_fitnesses, offspring_descriptors, offspring_extra_scores, random_key = self.emitter._rollout_fn(
            offspring,
            random_key
        )

        metrics = {
            "genotype": offspring,
            "fitness": offspring_fitnesses,
            "descriptor": offspring_descriptors,
            "population_fitness": extra_scores["population_fitness"],
            "population_descriptors": extra_scores["population_descriptors"],
            "population_networks": extra_scores["population_networks"],
        }

        es_emitter_state = es_emitter_state.replace(
            optimizer_state=optimizer_state
        )

        return (offspring, target_bd, es_emitter_state, random_key), metrics
        