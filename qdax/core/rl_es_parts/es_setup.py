# Import after parsing arguments
import functools
import time
from typing import Dict

import jax

import jax.numpy as jnp
import flax.linen as nn
import functools
import argparse


# print("Device count:", jax.device_count(), jax.devices())
import jax.numpy as jnp

from qdax import environments
from qdax.core.containers.mapelites_repertoire import compute_cvt_centroids

# from qdax.core.map_elites import MAPElites
from qdax.core.neuroevolution.networks.networks import MLP
from qdax.tasks.brax_envs import (
    make_policy_network_play_step_fn_brax,
    reset_based_scoring_function_brax_envs,
)
from qdax.utils.metrics import CSVLogger
from qdax.utils.plotting import plot_map_elites_results

from qdax.core.rl_es_parts.open_es import OpenESEmitter, OpenESConfig
from qdax.core.rl_es_parts.canonical_es import CanonicalESConfig, CanonicalESEmitter
from qdax.core.rl_es_parts.random_search import RandomConfig, RandomEmitter
from qdax.core.rl_es_parts.mono_cmaes import MonoCMAESEmitter, MonoCMAESConfig
from qdax.core.rl_es_parts.es_utils import (
    ES,
    default_es_metrics,
    ESMetrics,
    make_stochastic_policy_network_play_step_fn_brax,
)

from qdax.core.emitters.multi_actor_td3 import NoESConfig, NoESEmitter, MultiActorTD3


from qdax.core.emitters.custom_qpg_emitter import (
    CustomQualityPGConfig,
    CustomQualityPGEmitter,
    ESTargetQualityPGEmitter,
)

from qdax.core.emitters.esrl_emitter import ESRLConfig, ESRLEmitter
from qdax.core.emitters.test_gradients import TestGradientsEmitter
from qdax.core.emitters.carlies_emitter import CARLIES
from qdax.core.emitters.surrogate_es_emitter import (
    SurrogateESConfig,
    SurrogateESEmitter,
)
from qdax.core.emitters.spearman_surrogate import SpearmanSurrogateEmitter
import wandb
from dataclasses import dataclass

print("Device count:", jax.device_count(), jax.devices())


@dataclass
class ESMaker:
    es = None
    env = None
    emitter = None
    emitter_state = None
    repertoire = None
    random_key = None
    wandb_run = None
    policy_network = None
    rollout_fn = None
    eval_fn = None

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


def setup_es(args):
    print("Imported modules")

    ###############
    # Environment #

    # Init environment
    env = environments.create(
        args.env_name,
        episode_length=args.episode_length,
        fixed_init_state=args.deterministic,
    )

    # Init a random key
    random_key = jax.random.PRNGKey(args.seed)

    # Init policy network
    if args.groupsort or args.activation == "sort":
        print("Using groupsort")
        if args.groupsort_k != 1:
            raise NotImplementedError(
                "Groupsort with k != 1 not implemented for MLPs"
            )
        args.groupsort = True
        args.activation = "sort"

    activations = {
        "relu": nn.relu,
        "tanh": jnp.tanh,
        "sigmoid": jax.nn.sigmoid,
        "sort": jnp.sort,
    }
    if args.activation not in activations:
        raise NotImplementedError(
            f"Activation {args.activation} not implemented, choose one of {activations.keys()}"
        )
    
    activation = activations[args.activation]

    policy_layer_sizes = args.policy_hidden_layer_sizes + (env.action_size,)
    policy_network = MLP(
        layer_sizes=policy_layer_sizes,
        kernel_init=jax.nn.initializers.lecun_uniform(),
        final_activation=jnp.tanh,
        activation=activation,
    )

    args.policy_network = f"MLP {args.policy_layer_number}x{args.policy_hidden_layer_sizes[0]} {args.activation} -> tanh"
    print("Policy network", args.policy_network)
    args.critic_network = f"MLP {args.critic_layer_number}x{args.critic_hidden_layer_sizes[0]} relu -> none"
    print("Critic network", args.critic_network)
    
    # Init population of controllers
    random_key, subkey = jax.random.split(random_key)
    keys = jax.random.split(subkey, num=1)
    fake_batch = jnp.zeros(shape=(1, env.observation_size))
    init_variables = jax.vmap(policy_network.init)(keys, fake_batch)

    # print("Init variables", jax.tree_map(lambda x: x.shape, init_variables))

    # Play reset fn
    # WARNING: use "env.reset" for stochastic environment,
    # use "lambda random_key: init_state" for deterministic environment
    play_reset_fn = env.reset

    # Prepare the scoring function
    bd_extraction_fn = environments.behavior_descriptor_extractor[args.env_name]

    play_step_fn = make_policy_network_play_step_fn_brax(env, policy_network)

    eval_scoring_fn = functools.partial(
        reset_based_scoring_function_brax_envs,
        episode_length=args.episode_length,
        play_reset_fn=play_reset_fn,
        play_step_fn=play_step_fn,
        behavior_descriptor_extractor=bd_extraction_fn,
    )

    if args.explo_noise > 0:
        play_step_fn = make_stochastic_policy_network_play_step_fn_brax(
            env,
            policy_network,
            args.explo_noise,
        )
        print(f"Using exploration noise {args.explo_noise} in rollouts")

        rollout_scoring_fn = functools.partial(
            reset_based_scoring_function_brax_envs,
            episode_length=args.episode_length,
            play_reset_fn=play_reset_fn,
            play_step_fn=play_step_fn,
            behavior_descriptor_extractor=bd_extraction_fn,
        )

    else:
        rollout_scoring_fn = eval_scoring_fn

    # Get minimum reward value to make sure qd_score are positive
    reward_offset = environments.reward_offset[args.env_name]

    # Define a metrics function
    metrics_function = functools.partial(
        default_es_metrics,
        qd_offset=reward_offset * args.episode_length,
    )

    #############
    # Algorithm #

    # ES emitter
    if args.es in ["open", "openai"]:
        es_config = OpenESConfig(
            nses_emitter=args.nses,
            sample_number=args.pop,
            sample_sigma=args.es_sigma,
            sample_mirror=args.sample_mirror,
            sample_rank_norm=args.sample_rank_norm,
            adam_optimizer=args.adam_optimizer,
            learning_rate=args.learning_rate,
            l2_coefficient=args.l2_coefficient,
            novelty_nearest_neighbors=args.novelty_nearest_neighbors,
            actor_injection=args.actor_injection,
            nb_injections=args.nb_injections,
            episode_length=args.episode_length,
            explo_noise=args.explo_noise,
            groupsort=args.groupsort,
            groupsort_k=args.groupsort_k,
        )

        es_emitter = OpenESEmitter(
            config=es_config,
            rollout_fn=rollout_scoring_fn,
            eval_fn=eval_scoring_fn,
            total_generations=args.num_gens,
            num_descriptors=env.behavior_descriptor_length,
        )
    elif args.es in ["canonical"]:
        # print(CanonicalESConfig.__dataclass_fields__.keys())
        es_config = CanonicalESConfig(
            nses_emitter=args.nses,
            sample_number=args.pop,
            canonical_mu=int(args.pop / 2),
            sample_sigma=args.es_sigma,
            # learning_rate=args.learning_rate,
            novelty_nearest_neighbors=args.novelty_nearest_neighbors,
            actor_injection=args.actor_injection,
            nb_injections=args.nb_injections,
            episode_length=args.episode_length,
            injection_clipping=args.injection_clip,
            explo_noise=args.explo_noise,
            groupsort=args.groupsort,
            groupsort_k=args.groupsort_k,
        )

        es_emitter = CanonicalESEmitter(
            config=es_config,
            rollout_fn=rollout_scoring_fn,
            eval_fn=eval_scoring_fn,
            total_generations=args.num_gens,
            num_descriptors=env.behavior_descriptor_length,
        )

    elif args.es in ["cmaes"]:
        es_config = MonoCMAESConfig(
            nses_emitter=args.nses,
            sample_number=args.pop,
            sample_sigma=args.es_sigma,
            actor_injection=args.actor_injection,
            nb_injections=args.nb_injections,
            episode_length=args.episode_length,
            explo_noise=args.explo_noise,
            groupsort=args.groupsort,
            groupsort_k=args.groupsort_k,
        )

        es_emitter = MonoCMAESEmitter(
            config=es_config,
            rollout_fn=rollout_scoring_fn,
            eval_fn=eval_scoring_fn,
            total_generations=args.num_gens,
            num_descriptors=env.behavior_descriptor_length,
        )

    elif args.es in ["random"]:
        es_config = RandomConfig(
            nses_emitter=args.nses,
            sample_number=args.pop,
            actor_injection=args.actor_injection,
            nb_injections=args.nb_injections,
            episode_length=args.episode_length,
            explo_noise=args.explo_noise,
            groupsort=args.groupsort,
            groupsort_k=args.groupsort_k,
        )

        es_emitter = RandomEmitter(
            config=es_config,
            rollout_fn=rollout_scoring_fn,
            eval_fn=eval_scoring_fn,
            total_generations=args.num_gens,
            num_descriptors=env.behavior_descriptor_length,
        )

    elif args.es in ["multiactor"]:
        es_config = NoESConfig(
            nses_emitter=args.nses,
            sample_number=args.pop,
            novelty_nearest_neighbors=args.novelty_nearest_neighbors,
            episode_length=args.episode_length,
            explo_noise=args.explo_noise,
            groupsort=args.groupsort,
            groupsort_k=args.groupsort_k,
        )

        es_emitter = NoESEmitter(
            config=es_config,
            rollout_fn=rollout_scoring_fn,
            eval_fn=eval_scoring_fn,
            total_generations=args.num_gens,
            num_descriptors=env.behavior_descriptor_length,
        )

    else:
        raise ValueError(f"Unknown ES type: {args.es}")

    if args.rl:
        rl_config = CustomQualityPGConfig(
            env_batch_size=100,
            num_critic_training_steps=args.critic_training,
            num_pg_training_steps=args.pg_training,
            # TD3 params
            replay_buffer_size=1000000,
            critic_hidden_layer_size=args.critic_hidden_layer_sizes,
            critic_learning_rate=args.critic_lr,
            actor_learning_rate=args.actor_lr,
            policy_learning_rate=args.actor_lr,
            noise_clip=0.5,
            policy_noise=0.2,
            discount=args.discount,
            reward_scaling=1.0,
            batch_size=256,
            soft_tau_update=0.005,
            policy_delay=2,
            elastic_pull=args.elastic_pull,
            surrogate_batch=args.surrogate_batch,
        )

        if args.es_target:
            rl_emitter = ESTargetQualityPGEmitter(
                config=rl_config,
                policy_network=policy_network,
                env=env,
            )
        else:
            rl_emitter = CustomQualityPGEmitter(
                config=rl_config,
                policy_network=policy_network,
                env=env,
            )

        # ESRL emitter
        esrl_emitter_type = ESRLEmitter
        if args.carlies:
            esrl_emitter_type = CARLIES
        elif args.testrl:
            esrl_emitter_type = TestGradientsEmitter
        elif args.es in ["multiactor"]:
            esrl_emitter_type = MultiActorTD3

        if args.surrogate:
            esrl_config = SurrogateESConfig(
                es_config=es_config,
                rl_config=rl_config,
                surrogate_omega=args.surrogate_omega,
            )
            esrl_emitter_type = SurrogateESEmitter
        elif args.spearman:
            esrl_config = ESRLConfig(
                es_config=es_config,
                rl_config=rl_config,
            )
            esrl_emitter_type = SpearmanSurrogateEmitter

        else:
            esrl_config = ESRLConfig(
                es_config=es_config,
                rl_config=rl_config,
            )

        emitter = esrl_emitter_type(
            config=esrl_config,
            es_emitter=es_emitter,
            rl_emitter=rl_emitter,
        )

        print(f"ES-RL emitter type: {esrl_emitter_type}")

    else:
        emitter = es_emitter

    args.config = emitter.config_string

    # Instantiate ES
    es = ES(
        scoring_function=eval_scoring_fn,
        emitter=emitter,
        metrics_function=metrics_function,
    )

    # Compute the centroids
    centroids, random_key = compute_cvt_centroids(
        num_descriptors=env.behavior_descriptor_length,
        num_init_cvt_samples=args.num_init_cvt_samples,
        num_centroids=args.num_centroids,
        minval=args.min_bd,
        maxval=args.max_bd,
        random_key=random_key,
    )

    # Compute initial repertoire and emitter state
    repertoire, emitter_state, random_key = es.init(
        init_variables, centroids, random_key
    )

    # print("After ES init:", emitter_state.rl_state.replay_buffer.current_position)

    # print("Initialized ES")
    print(es_emitter)
    
    print(emitter.config_string)

    entity = None
    project = args.wandb
    wandb_run = None
    if project != "":
        if "/" in project:
            entity, project = project.split("/")
        wandb_run = wandb.init(project=project, entity=entity, config={**vars(args)})

        print("Initialized wandb")

    return ESMaker(
        es=es,
        env=env,
        emitter=emitter,
        emitter_state=emitter_state,
        repertoire=repertoire,
        random_key=random_key,
        wandb_run=wandb_run,
        policy_network=policy_network,
        rollout_fn=rollout_scoring_fn,
        eval_fn=eval_scoring_fn,
        scoring_fn=eval_scoring_fn,
    )


# Setup parser for command-line arguments.
def get_es_parser():
    # Argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--env",
        type=str,
        default="walker2d_uni",
        help="Environment name",
        # choices=['ant_uni', 'hopper_uni', 'walker2d_uni', 'halfcheetah_uni', 'humanoid_uni', 'ant_omni', 'humanoid_omni', 'anttrap'],
        dest="env_name",
    )
    parser.add_argument(
        "--episode_length", type=int, default=1000, help="Number of steps per episode"
    )
    # parser.add_argument('--gen', type=int, default=10000, help='Generations', dest='num_iterations')
    parser.add_argument("--evals", type=int, default=1000000, help="Evaluations")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    # Networks
    parser.add_argument(
        "--policy_hidden_layer_sizes",
        type=int,
        default=128,
        help="Policy network hidden layer sizes",
    )
    parser.add_argument(
        "--policy_layer_number",
        type=int,
        default=2,
        help="Policy network hidden layer number",
    )
    parser.add_argument(
        "--critic_hidden_layer_sizes",
        type=int,
        default=128,
        help="critic network hidden layer sizes",
    )
    parser.add_argument(
        "--critic_layer_number",
        type=int,
        default=2,
        help="Critic network hidden layer number",
    )
    parser.add_argument(
        "--deterministic", default=False, action="store_true", help="Fixed init state"
    )
    parser.add_argument(
        "--groupsort", default=False, action="store_true", help="Groupsort activation function"
    )
    parser.add_argument(
        "--groupsort_k", type=int, default=1, help="Number of groups for groupsort"
    )
    # activation
    parser.add_argument(
        "--activation",
        type=str,
        default="relu",
        help="Activation function for policy network",
    )

    # Exploration noise, default 0
    parser.add_argument(
        "--explo_noise",
        type=float,
        default=0.0,
        help="Exploration noise (Gaussian)",
    )

    # Map-Elites
    parser.add_argument(
        "--num_init_cvt_samples",
        type=int,
        default=50000,
        help="Number of samples to use for CVT initialization",
    )
    parser.add_argument(
        "--num_centroids", type=int, default=1024, help="Number of centroids"
    )
    parser.add_argument(
        "--min_bd",
        type=float,
        default=0.0,
        help="Minimum value for the behavior descriptor",
    )
    parser.add_argument(
        "--max_bd",
        type=float,
        default=1.0,
        help="Maximum value for the behavior descriptor",
    )

    # ES
    # ES type
    parser.add_argument(
        "--es",
        type=str,
        default="es",
        help="ES type",
        choices=["open", "canonical", "cmaes", "random", "multiactor"],
    )
    parser.add_argument("--pop", type=int, default=512, help="Population size")
    parser.add_argument(
        "--es_sigma",
        type=float,
        default=0.01,
        help="Standard deviation of the Gaussian distribution",
    )
    parser.add_argument(
        "--sample_mirror", type=bool, default=True, help="Mirror sampling in ES"
    )
    parser.add_argument(
        "--sample_rank_norm", type=bool, default=True, help="Rank normalization in ES"
    )
    parser.add_argument(
        "--adam_optimizer",
        type=bool,
        default=True,
        help="Use Adam optimizer instead of SGD",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.01,
        help="Learning rate for ES optimizer",
    )
    parser.add_argument(
        "--l2_coefficient",
        type=float,
        default=0.02,
        help="L2 coefficient for Adam optimizer",
    )

    # NSES
    parser.add_argument(
        "--nses", default=False, action="store_true", help="Use NSES instead of ES"
    )
    parser.add_argument(
        "--novelty_nearest_neighbors",
        type=int,
        default=10,
        help="Number of nearest neighbors to use for novelty computation",
    )

    # RL
    parser.add_argument("--rl", default=False, action="store_true", help="Add RL")
    parser.add_argument(
        "--testrl", default=False, action="store_true", help="Add RL/ES testing"
    )
    parser.add_argument(
        "--carlies", default=False, action="store_true", help="Add CARLIES"
    )
    parser.add_argument(
        "--elastic_pull",
        type=float,
        default=0,
        help="Penalization for pulling the actor too far from the ES center",
    )
    parser.add_argument("--discount", type=float, default=0.99, help="Discount factor")
    parser.add_argument(
        "--actor_injection",
        action="store_true",
        default=False,
        help="Use actor injection",
    )
    parser.add_argument(
        "--injection_clip",
        action="store_true",
        default=False,
        help="Clip actor vector norm for injection",
    )
    parser.add_argument(
        "--nb_injections",
        type=int,
        default=1,
        help="Number of actors to inject if actor_injection is True",
    )
    parser.add_argument(
        "--critic_training",
        type=int,
        default=1000,
        help="Number of critic training steps",
    )
    parser.add_argument(
        "--pg_training", type=int, default=1000, help="Number of PG training steps"
    )
    parser.add_argument(
        "--actor_lr",
        type=float,
        default=3e-4,
        help="Learning rate for actor Adam optimizer",
    )
    parser.add_argument(
        "--critic_lr",
        type=float,
        default=3e-4,
        help="Learning rate for critic Adam optimizer",
    )
    parser.add_argument(
        "--es_target",
        action="store_true",
        default=False,
        help="Use ES center as critic target",
    )

    # RL + ES
    parser.add_argument(
        "--surrogate", default=False, action="store_true", help="Use surrogate"
    )
    parser.add_argument(
        "--surrogate_batch",
        type=int,
        default=1024,
        help="Number of samples to use for surrogate evaluation",
    )
    parser.add_argument(
        "--surrogate_omega",
        type=float,
        default=0.6,
        help="Probability of using surrogate",
    )
    parser.add_argument(
        "--spearman",
        default=False,
        action="store_true",
        help="Use surrogate with spearman-ajusted probability",
    )
    # parser.add_argument('--spearman_decay', type=float, default=1.0, help='Spearman decay')

    # File output
    parser.add_argument("--output", type=str, default="", help="Output file")
    parser.add_argument("--plot", default=False, action="store_true", help="Make plots")

    # Wandb
    parser.add_argument("--wandb", type=str, default="", help="Wandb project name")
    parser.add_argument("--tag", type=str, default="", help="Project tag")
    parser.add_argument("--jobid", type=str, default="", help="Job ID")

    # Log period
    parser.add_argument("--log_period", type=int, default=1, help="Log period")

    # Debug flag
    parser.add_argument(
        "--debug", default=False, action="store_true", help="Debug flag"
    )
    parser.add_argument(
        "--logall", default=False, action="store_true", help="Lot at each generation"
    )
    return parser


def fill_default(args):
    # Get default arguments from es_parser
    # If an arg is not in args, add it with the default value
    # Args and es_args are both from argparse
    es_parser = get_es_parser()
    es_args = vars(es_parser.parse_args([]))
    for key, value in es_args.items():
        if key not in args:
            print(f"Adding default argument {key} = {value}")
            # Add default argument to the namespace
            setattr(args, key, value)

    return args
