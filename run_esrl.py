import argparse

# Argparse
parser = argparse.ArgumentParser()
parser.add_argument(
    '--env', 
    type=str, 
    default="walker2d_uni", 
    help='Environment name', 
    choices=['ant_uni', 'hopper_uni', 'walker2d_uni', 'halfcheetah_uni', 'humanoid_uni', 'ant_omni', 'humanoid_omni'],
    dest='env_name'
    )
parser.add_argument('--episode_length', type=int, default=1000, help='Number of steps per episode')
parser.add_argument('--evals', type=int, default=1000000, help='Evaluations')
parser.add_argument('--seed', type=int, default=42, help='Random seed')
parser.add_argument('--policy_hidden_layer_sizes', type=int, nargs='+', default=[128, 128], help='Policy network hidden layer sizes')

# Map-Elites
parser.add_argument('--num_init_cvt_samples', type=int, default=50000, help='Number of samples to use for CVT initialization')
parser.add_argument('--num_centroids', type=int, default=1024, help='Number of centroids')
parser.add_argument('--min_bd', type=float, default=0.0, help='Minimum value for the behavior descriptor')
parser.add_argument('--max_bd', type=float, default=1.0, help='Maximum value for the behavior descriptor')

# ES
# ES type
parser.add_argument('--es', type=str, default='open', help='ES type', choices=['open', 'canonical'])
parser.add_argument('--pop', type=int, default=512, help='Population size')
parser.add_argument('--es_sigma', type=float, default=0.01, help='Standard deviation of the Gaussian distribution')
parser.add_argument('--sample_mirror', type=bool, default=True, help='Mirror sampling in ES')
parser.add_argument('--sample_rank_norm', type=bool, default=True, help='Rank normalization in ES')
parser.add_argument('--adam_optimizer', type=bool, default=True, help='Use Adam optimizer instead of SGD')
parser.add_argument('--learning_rate', type=float, default=0.01, help='Learning rate for Adam optimizer')
parser.add_argument('--l2_coefficient', type=float, default=0.02, help='L2 coefficient for Adam optimizer')

# ES + RL
parser.add_argument('--actor_injection', action="store_true", default=False, help='Use actor injection')

# NSES
parser.add_argument('--nses_emitter', type=bool, default=False, help='Use NSES instead of ES')
parser.add_argument('--novelty_nearest_neighbors', type=int, default=10, help='Number of nearest neighbors to use for novelty computation')

# File output
parser.add_argument('--output', type=str, default='output', help='Output file')
parser.add_argument('--plot', default=False, action="store_true", help='Make plots')

# Wandb
parser.add_argument('--wandb', type=str, default='', help='Wandb project name')
parser.add_argument('--tag', type=str, default='', help='Project tag')
parser.add_argument('--jobid', type=str, default='', help='Job ID')

# Log period
parser.add_argument('--log_period', type=int, default=1, help='Log period')

# Debug flag 
parser.add_argument('--debug', default=False, action="store_true", help='Debug flag')

# parse arguments
args = parser.parse_args()

if args.debug:
    # Cheap ES to debug
    debug_values = {
        'env_name': 'walker2d_uni',
        'episode_length': 100,
        "pop": 10,
        'evals': 100,
        'seed': 42,
        'policy_hidden_layer_sizes': (32, 32),
        "output": "debug"
    }
    for k, v in debug_values.items():
        setattr(args, k, v)

args.policy_hidden_layer_sizes = tuple(args.policy_hidden_layer_sizes)
args.num_gens = args.evals // args.pop

args.algo = "ESRL-Alt"
if args.actor_injection:
    args.algo += "-AI"

print("Parsed arguments:", args)

# Import after parsing arguments
import functools
import time
from typing import Dict

import jax
import jax.numpy as jnp

from qdax import environments
from qdax.core.containers.mapelites_repertoire import compute_cvt_centroids
from qdax.core.emitters.vanilla_es_emitter import VanillaESConfig, VanillaESEmitter
# from qdax.core.map_elites import MAPElites
from qdax.core.neuroevolution.networks.networks import MLP
from qdax.tasks.brax_envs import (
    make_policy_network_play_step_fn_brax,
    reset_based_scoring_function_brax_envs,
)
from qdax.utils.metrics import CSVLogger, default_qd_metrics
from qdax.utils.plotting import plot_map_elites_results

from qdax.core.rl_es_parts.es_utils import ES, default_es_metrics, ESMetrics
from qdax.core.rl_es_parts.open_es import OpenESEmitter, OpenESConfig
from qdax.core.rl_es_parts.canonical_es import CanonicalESConfig, CanonicalESEmitter

from qdax.core.emitters.qpg_emitter import QualityPGConfig, QualityPGEmitterState, QualityPGEmitter

from qdax.core.emitters.esrl_emitter import ESRLConfig, ESRLEmitter

import wandb
print("Imported modules")

entity = None
project = args.wandb
wandb_run = None
if project != "":
    if "/" in project:
        entity, project = project.split("/")
    wandb_run = wandb.init(
        project=project,
        entity=entity,
        config = {**vars(args)})

    print("Initialized wandb")

###############
# Environment #

# Init environment
env = environments.create(args.env_name, episode_length=args.episode_length)

# Init a random key
random_key = jax.random.PRNGKey(args.seed)

# Init policy network
policy_layer_sizes = args.policy_hidden_layer_sizes + (env.action_size,)
policy_network = MLP(
    layer_sizes=policy_layer_sizes,
    kernel_init=jax.nn.initializers.lecun_uniform(),
    final_activation=jnp.tanh,
)

# Init population of controllers
random_key, subkey = jax.random.split(random_key)
keys = jax.random.split(subkey, num=1)
fake_batch = jnp.zeros(shape=(1, env.observation_size))
init_variables = jax.vmap(policy_network.init)(keys, fake_batch)

# Play reset fn
# WARNING: use "env.reset" for stochastic environment,
# use "lambda random_key: init_state" for deterministic environment
play_reset_fn = env.reset

# Prepare the scoring function
bd_extraction_fn = environments.behavior_descriptor_extractor[args.env_name]
scoring_fn = functools.partial(
    reset_based_scoring_function_brax_envs,
    episode_length=args.episode_length,
    play_reset_fn=play_reset_fn,
    play_step_fn=make_policy_network_play_step_fn_brax(env, policy_network),
    behavior_descriptor_extractor=bd_extraction_fn,
)

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
    args.es = "open"
    es_config = OpenESConfig(
        nses_emitter=args.nses_emitter,
        sample_number=args.pop,
        sample_sigma=args.es_sigma,
        sample_mirror=args.sample_mirror,
        sample_rank_norm=args.sample_rank_norm,
        adam_optimizer=args.adam_optimizer,
        learning_rate=args.learning_rate,
        l2_coefficient=args.l2_coefficient,
        novelty_nearest_neighbors=args.novelty_nearest_neighbors,
        actor_injection = args.actor_injection,
    )

    es_emitter = OpenESEmitter(
        config=es_config,
        scoring_fn=scoring_fn,
        total_generations=args.num_gens,
        num_descriptors=env.behavior_descriptor_length,
    )
elif args.es in ["canonical"]:
    es_config = CanonicalESConfig(
        nses_emitter=args.nses_emitter,
        sample_number=args.pop,
        canonical_mu=int(args.pop / 2),
        sample_sigma=args.es_sigma,
        learning_rate=args.learning_rate,
        novelty_nearest_neighbors=args.novelty_nearest_neighbors,
        actor_injection = args.actor_injection,
    )

    es_emitter = CanonicalESEmitter(
        config=es_config,
        scoring_fn=scoring_fn,
        total_generations=args.num_gens,
        num_descriptors=env.behavior_descriptor_length,
    )

else:
    raise ValueError(f"Unknown ES type: {args.es}")

# QPG emitter
rl_config = QualityPGConfig(
    env_batch_size = 100,
    num_critic_training_steps = 1000,
    num_pg_training_steps = 1000,

    # TD3 params
    replay_buffer_size = 1000000,
    critic_hidden_layer_size = (256, 256),
    critic_learning_rate = 3e-4,
    actor_learning_rate = 3e-4,
    policy_learning_rate = 1e-3,
    noise_clip = 0.5,
    policy_noise = 0.2,
    discount = 0.99,
    reward_scaling = 1.0,
    batch_size = 256,
    soft_tau_update = 0.005,
    policy_delay = 2
)
    
rl_emitter = QualityPGEmitter(
    config=rl_config,
    policy_network=policy_network,
    env=env,
)

# RL-ES emitter
esrl_config = ESRLConfig(
    es_config=es_config,
    rl_config=rl_config,
)

esrl_emitter = ESRLEmitter(
    config=esrl_config,
    es_emitter=es_emitter,
    rl_emitter=rl_emitter,
)

# ES 
es = ES(
    scoring_function=scoring_fn,
    emitter=esrl_emitter,
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


#######
# Run #

log_period = args.log_period
num_loops = int(args.num_gens / log_period)

log_file = args.output
if not log_file.endswith(".csv"):
    log_file += ".csv"

# create log file
with open(log_file, "w+") as file:
    pass

plot_file = args.output
if not plot_file.endswith(".png"):
    plot_file += ".png"

# get all the fields in ESMetrics
header = ESMetrics.__dataclass_fields__.keys()

csv_logger = CSVLogger(
    log_file,
    header=["loop", "generation", 
            "qd_score",  "max_fitness", "coverage", 
            "time", "frames"] + list(header),
)
all_metrics: Dict[str, float] = {}

# main loop
es_scan_update = es.scan_update

# main iterations
from tqdm import tqdm
bar = tqdm(range(num_loops))
try:
    for i in bar:
        start_time = time.time()
        (repertoire, emitter_state, random_key,), metrics = jax.lax.scan(
            es_scan_update,
            (repertoire, emitter_state, random_key),
            (),
            length=log_period,
        )
        timelapse = time.time() - start_time

        # log metrics
        gen = 1 + i * log_period
        logged_metrics = {
            "time": timelapse, 
            "loop": 1 + i, 
            "generation": gen,
            }
        for key, value in metrics.items():
            # take last value
            logged_metrics[key] = value[-1]

            # take all values
            if key in all_metrics.keys():
                all_metrics[key] = jnp.concatenate([all_metrics[key], value])
            else:
                all_metrics[key] = value
        logged_metrics["frames"] = logged_metrics["evaluations"] * args.episode_length
        csv_logger.log(logged_metrics)
        if wandb_run:
            wandb_run.log(logged_metrics)


        # Update bar
        qd_score = logged_metrics["qd_score"]
        max_fitness = logged_metrics["max_fitness"]
        es_steps = logged_metrics["es_updates"]
        rl_steps = logged_metrics["rl_updates"]
        bar.set_description(f"Gen: {gen}, qd_score: {qd_score:.2f}, max_fitness: {max_fitness:.2f}, ES/RL:{es_steps}/{rl_steps} time: {timelapse:.2f}")
except KeyboardInterrupt:
    print("Interrupted by user")

#################
# Visualisation #
if args.plot:
    # create the x-axis array
    env_steps = jnp.arange(args.num_gens) * args.episode_length

    # create the plots and the grid
    fig, axes = plot_map_elites_results(
        env_steps=env_steps,
        metrics=all_metrics,
        repertoire=repertoire,
        min_bd=args.min_bd,
        max_bd=args.max_bd,
    )

    import matplotlib.pyplot as plt
    plt.savefig(plot_file)

    from qdax.utils.plotting import plot_2d_map_elites_repertoire

    fig, ax = plot_2d_map_elites_repertoire(
        centroids=repertoire.centroids,
        repertoire_fitnesses=repertoire.fitnesses,
        minval=args.min_bd,
        maxval=args.max_bd,
        repertoire_descriptors=repertoire.descriptors,
    )

    # Log the plot
    if wandb_run:
        wandb_run.log({"archive": wandb.Image(fig)})
        wandb.finish()
