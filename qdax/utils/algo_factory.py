import functools
import time

import jax
import jax.numpy as jnp

from qdax.core.custom_repertoire_mapelites import CustomMAPElites
from qdax.core.containers.mapelites_repertoire import (
    compute_cvt_centroids,
    MapElitesRepertoire,
)
from qdax.core.containers.count_repertoire import (
    CountMapElitesRepertoire,
    count_qd_metrics,
)

from qdax import environments
from qdax.tasks.brax_envs import scoring_function_brax_envs as scoring_function

from qdax.core.neuroevolution.buffers.buffer import QDTransition
from qdax.core.neuroevolution.networks.networks import MLP

# Emitters
from qdax.core.emitters.mutation_operators import isoline_variation
from qdax.core.emitters.standard_emitters import MixingEmitter
from qdax.core.emitters.pga_me_emitter import PGAMEConfig, PGAMEEmitter
from qdax.core.emitters.mees_emitter import MEESConfig, MEESEmitter

# CMA-ME Emitters
from qdax.core.emitters.cma_me_policies import CMAMEPolicies, CMAPoolPolicies

# from qdax.core.emitters.cma_pool_emitter import CMAPoolEmitter

from qdax.utils.sampling import sampling

from qdax.utils.metrics import CSVLogger, default_qd_metrics

from qdax.core.emitters.vanilla_es_emitter import flatten_genotype


class MEFactory:
    def __init__(self, config):
        self.config = config

        self.env = None
        self.map_elites = None
        self.init_repertoire = None
        self.emitter_state = None
        self.random_key = None
        self.metrics_function = None
        self.init_variables = None
        self.scoring_fn = None
        self.policy_network = None
        self.centroids = None

        self.me_type = "None"

        self.get_env_steps = (
            lambda gen: gen * self.config["batch_size"] * self.config["episode_length"]
        )

        self.end_repertoire = None
        self.metrics = None

    def __repr__(self) -> str:
        s = f"MEFactory({self.me_type})"
        return s

    def __str__(self) -> str:
        return self.__repr__()

    def reset(self):
        print(f"Resetting {self}")
        self.end_repertoire = None
        self.metrics = None

    def _get_default(self):
        # Init environment
        env = environments.create(
            self.config["env_name"], episode_length=self.config["episode_length"]
        )

        # Init a random key
        random_key = jax.random.PRNGKey(self.config["seed"])

        activations = {
            "relu": jax.nn.relu,
            "tanh": jnp.tanh,
            "sigmoid": jax.nn.sigmoid,
            "elu": jax.nn.elu,
        }
        activation = activations[self.config["activation"]]

        # Init policy network
        policy_layer_sizes = self.config["policy_hidden_layer_sizes"] + (
            env.action_size,
        )
        policy_network = MLP(
            layer_sizes=policy_layer_sizes,
            kernel_init=jax.nn.initializers.lecun_uniform(),
            final_activation=jnp.tanh,
            activation=activation,
        )

        # Init population of controllers
        random_key, subkey = jax.random.split(random_key)

        fake_batch = jnp.zeros(shape=(self.config["batch_size"], env.observation_size))
        keys = jax.random.split(subkey, num=self.config["batch_size"])
        init_variables = jax.vmap(policy_network.init)(keys, fake_batch)

        # Create the initial environment states
        random_key, subkey = jax.random.split(random_key)
        keys = jnp.repeat(
            jnp.expand_dims(subkey, axis=0), repeats=self.config["batch_size"], axis=0
        )
        reset_fn = jax.jit(jax.vmap(env.reset))
        init_states = reset_fn(keys)

        # Define the fonction to play a step with the policy in the environment
        def play_step_fn(
            env_state,
            policy_params,
            random_key,
        ):
            """
            Play an environment step and return the updated state and the transition.
            """

            actions = policy_network.apply(policy_params, env_state.obs)

            state_desc = env_state.info["state_descriptor"]
            next_state = env.step(env_state, actions)

            transition = QDTransition(
                obs=env_state.obs,
                next_obs=next_state.obs,
                rewards=next_state.reward,
                dones=next_state.done,
                actions=actions,
                truncations=next_state.info["truncation"],
                state_desc=state_desc,
                next_state_desc=next_state.info["state_descriptor"],
            )

            return next_state, policy_params, random_key, transition

        # Prepare the scoring function
        bd_extraction_fn = environments.behavior_descriptor_extractor[
            self.config["env_name"]
        ]
        scoring_fn = functools.partial(
            scoring_function,
            init_states=init_states,
            episode_length=self.config["episode_length"],
            play_step_fn=play_step_fn,
            behavior_descriptor_extractor=bd_extraction_fn,
        )

        # Get minimum reward value to make sure qd_score are positive
        self.reward_offset = environments.reward_offset[self.config["env_name"]]
        self.qd_offset = self.reward_offset * self.config["episode_length"]

        metrics_function = functools.partial(
            count_qd_metrics,
            qd_offset=self.qd_offset,
        )

        # Compute the centroids
        centroids, random_key = compute_cvt_centroids(
            num_descriptors=env.behavior_descriptor_length,
            num_init_cvt_samples=self.config["num_init_cvt_samples"],
            num_centroids=self.config["num_centroids"],
            minval=self.config["min_bd"],
            maxval=self.config["max_bd"],
            random_key=random_key,
        )

        self.centroids = centroids
        self.scoring_fn = scoring_fn
        self.env = env
        self.random_key = random_key
        self.metrics_function = metrics_function
        self.init_variables = init_variables
        self.policy_network = policy_network

    def _make_mapelites_objects(self, emitter):
        # Instantiate MAP-Elites
        map_elites = CustomMAPElites(
            scoring_function=self.scoring_fn,
            emitter=emitter,
            metrics_function=self.metrics_function,
            repertoire_type=CountMapElitesRepertoire,
        )

        # Compute initial repertoire and emitter state
        repertoire, emitter_state, random_key = map_elites.init(
            self.init_variables, self.centroids, self.random_key
        )

        self.map_elites = map_elites
        self.init_repertoire = repertoire
        self.emitter_state = emitter_state
        self.random_key = random_key

        return self

    def get_mapelites(self):
        default_config = {
            "iso_sigma": 0.005,
            "line_sigma": 0.05,
            "batch_size": 100,
        }
        self.config = {
            **default_config,
            **self.config,
        }  # Merge default and user config, user config has priority

        self._get_default()

        # Define emitter
        variation_fn = functools.partial(
            isoline_variation,
            iso_sigma=self.config["iso_sigma"],
            line_sigma=self.config["line_sigma"],
        )

        mixing_emitter = MixingEmitter(
            mutation_fn=None,
            variation_fn=variation_fn,
            variation_percentage=1.0,
            batch_size=self.config["batch_size"],
        )
        self.me_type = "MAP-Elites"
        return self._make_mapelites_objects(mixing_emitter)

    def get_pgame(self):
        default_config = {
            "proportion_mutation_ga": 0.5,
            "iso_sigma": 0.005,
            "line_sigma": 0.05,
            "batch_size": 100,
        }
        td3_config = {
            "replay_buffer_size": 1000000,
            "critic_hidden_layer_size": (256, 256),
            "critic_learning_rate": 3e-4,
            "greedy_learning_rate": 3e-4,
            "policy_learning_rate": 1e-3,
            "noise_clip": 0.5,
            "policy_noise": 0.2,
            "discount": 0.99,
            "reward_scaling": 1.0,
            "transitions_batch_size": 256,
            "soft_tau_update": 0.005,
            "num_critic_training_steps": 300,
            "num_pg_training_steps": 100,
            "policy_delay": 2,
        }
        default_config = {**td3_config, **default_config}

        self.config = {
            **default_config,
            **self.config,
        }  # Merge default and user config, user config has priority

        self._get_default()

        # Define emitter
        # Define the PG-emitter config
        pga_emitter_config = PGAMEConfig(
            env_batch_size=self.config["batch_size"],
            batch_size=self.config["transitions_batch_size"],
            proportion_mutation_ga=self.config["proportion_mutation_ga"],
            critic_hidden_layer_size=self.config["critic_hidden_layer_size"],
            critic_learning_rate=self.config["critic_learning_rate"],
            greedy_learning_rate=self.config["greedy_learning_rate"],
            policy_learning_rate=self.config["policy_learning_rate"],
            noise_clip=self.config["noise_clip"],
            policy_noise=self.config["policy_noise"],
            discount=self.config["discount"],
            reward_scaling=self.config["reward_scaling"],
            replay_buffer_size=self.config["replay_buffer_size"],
            soft_tau_update=self.config["soft_tau_update"],
            num_critic_training_steps=self.config["num_critic_training_steps"],
            num_pg_training_steps=self.config["num_pg_training_steps"],
            policy_delay=self.config["policy_delay"],
        )

        # Define emitter
        variation_fn = functools.partial(
            isoline_variation,
            iso_sigma=self.config["iso_sigma"],
            line_sigma=self.config["line_sigma"],
        )

        pg_emitter = PGAMEEmitter(
            config=pga_emitter_config,
            policy_network=self.policy_network,
            env=self.env,
            variation_fn=variation_fn,
        )

        self.me_type = "PGA-ME"

        return self._make_mapelites_objects(pg_emitter)

    def get_cmame(self):
        default_config = {
            "sigma_g": 0.5,
            "pool_size": 10,
            "batch_size": 100,
        }

        self.config = {
            **default_config,
            **self.config,
        }  # Merge default and user config, user config has priority

        # if self.config["pool_size"] > 1:
        #     self.config["batch_size"] = (
        #         self.config["batch_size"] * self.config["pool_size"]
        #     )
        #     self.config["pool_size"] = 1
        #     import warnings

        #     warnings.warn(
        #         f"""`pool_size` > 1.
        #         Setting `pool_size` to 1 and `batch_size` to `batch_size * pool_size`."""
        #     )
        self._get_default()

        emitter_kwargs = {
            "batch_size": self.config["batch_size"],
            "sigma_g": self.config["sigma_g"],
            "centroids": self.centroids,
            "min_count": 1,
            "max_count": None,
        }

        emitter = CMAMEPolicies(**emitter_kwargs)

        emitter = CMAPoolPolicies(num_states=self.config["pool_size"], emitter=emitter)

        self.me_type = "CMA-ME"
        return self._make_mapelites_objects(emitter)

    def run(self):
        assert self.map_elites is not None, "No Map-Elites algorithm defined"
        assert (
            self.end_repertoire is None
        ), "Map-Elites algorithm already run, call reset() to run again."

        init_metrics = self.metrics_function(self.init_repertoire)

        log_period = 50
        num_loops = int(self.config["num_iterations"] / log_period)

        csv_logger = CSVLogger(
            "mapelites-logs.csv",
            header=["loop", "iteration", "time"] + list(init_metrics.keys()),
        )
        all_metrics = {}

        repertoire = self.init_repertoire

        from tqdm import tqdm

        # main loop
        map_elites_scan_update = self.map_elites.scan_update
        for i in tqdm(range(num_loops)):
            start_time = time.time()
            # main iterations
            (
                repertoire,
                emitter_state,
                random_key,
            ), metrics = jax.lax.scan(
                map_elites_scan_update,
                (repertoire, self.emitter_state, self.random_key),
                (),
                length=log_period,
            )
            timelapse = time.time() - start_time

            # log metrics
            logged_metrics = {
                "time": timelapse,
                "loop": 1 + i,
                "iteration": 1 + i * log_period,
            }
            for key, value in metrics.items():
                # take last value
                logged_metrics[key] = value[-1]

                # take all values
                if key in all_metrics.keys():
                    all_metrics[key] = jnp.concatenate([all_metrics[key], value])
                else:
                    all_metrics[key] = value

            csv_logger.log(logged_metrics)

        all_metrics["env_steps"] = self.get_env_steps(
            jnp.arange(self.config["num_iterations"])
        )

        self.end_repertoire = repertoire
        self.metrics = all_metrics

        return repertoire, all_metrics

    def get_mees(self):
        raise NotImplementedError("MEES is not implemented yet.")
        # MEES only works with batch_size=1 (ES center)
        if "batch_size" in self.config:
            self.config["sample_number"] = self.config["batch_size"]
            self.config["batch_size"] = 1
            # Warn
            import warnings

            warnings.warn(
                "MEES only works with batch_size=1 (ES center). Setting sample_number=batch_size and batch_size=1."
            )

        self.get_env_steps = (
            lambda gen: gen
            * self.config["sample_number"]
            * self.config["episode_length"]
        )

        default_config = {
            "proportion_mutation_ga": 0.5,
            "iso_sigma": 0.005,
            "line_sigma": 0.05,
            "batch_size": 1,
        }
        mees_config = {
            "sample_number": 1000,
            "sample_sigma": 0.02,
            "num_optimizer_steps": 10,
            "learning_rate": 0.01,
            "l2_coefficient": 0.02,
            "novelty_nearest_neighbors": 10,
            "last_updated_size": 5,
            "exploit_num_cell_sample": 2,
            "explore_num_cell_sample": 5,
            "adam_optimizer": True,
            "sample_mirror": True,
            "sample_rank_norm": True,
            "use_explore": True,
        }

        default_config = {**mees_config, **default_config}

        self.config = {
            **default_config,
            **self.config,
        }  # Merge default and user config, user config has priority

        self._get_default()

        assert (
            self.config["batch_size"] == 1
        ), "MEES only works with batch_size=1 (ES center)."

        # Define the MEES-emitter config
        mees_emitter_config = MEESConfig(
            sample_number=self.config["sample_number"],
            sample_sigma=self.config["sample_sigma"],
            sample_mirror=self.config["sample_mirror"],
            sample_rank_norm=self.config["sample_rank_norm"],
            num_optimizer_steps=self.config["num_optimizer_steps"],
            adam_optimizer=self.config["adam_optimizer"],
            learning_rate=self.config["learning_rate"],
            l2_coefficient=self.config["l2_coefficient"],
            novelty_nearest_neighbors=self.config["novelty_nearest_neighbors"],
            last_updated_size=self.config["last_updated_size"],
            exploit_num_cell_sample=self.config["exploit_num_cell_sample"],
            explore_num_cell_sample=self.config["explore_num_cell_sample"],
            use_explore=self.config["use_explore"],
        )

        # Prepare the scoring functions for the offspring generated folllowing
        # the approximated gradient (each of them is evaluated 30 times)
        sampling_fn = functools.partial(
            sampling,
            scoring_fn=self.scoring_fn,
            num_samples=30,
        )

        # Get the emitter
        mees_emitter = MEESEmitter(
            config=mees_emitter_config,
            total_generations=self.config["num_iterations"],
            scoring_fn=self.scoring_fn,
            num_descriptors=self.env.behavior_descriptor_length,
        )

        self.me_type = "ME-ES"

        # Instantiate MAP-Elites
        map_elites = CustomMAPElites(
            scoring_function=sampling_fn,
            emitter=mees_emitter,
            metrics_function=self.metrics_function,
            repertoire_type=CountMapElitesRepertoire,
        )

        # Compute the centroids
        centroids, random_key = compute_cvt_centroids(
            num_descriptors=self.env.behavior_descriptor_length,
            num_init_cvt_samples=self.config["num_init_cvt_samples"],
            num_centroids=self.config["num_centroids"],
            minval=self.config["min_bd"],
            maxval=self.config["max_bd"],
            random_key=self.random_key,
        )

        # Compute initial repertoire and emitter state
        repertoire, emitter_state, random_key = map_elites.init(
            self.init_variables, centroids, random_key
        )

        self.map_elites = map_elites
        self.init_repertoire = repertoire
        self.emitter_state = emitter_state
        self.random_key = random_key

        return self
