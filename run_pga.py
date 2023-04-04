import argparse

# Argparse
parser = argparse.ArgumentParser()
parser.add_argument(
    '--env', 
    type=str, 
    default="walker2d_uni", 
    help='Environment name', 
    choices=['ant_uni', 'hopper_uni', 'walker2d_uni', 'halfcheetah_uni', 'humanoid_uni', 'ant_omni', 'humanoid_omni', 'anttrap'],
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
parser.add_argument('--es', type=str, default='open', help='ES type', choices=['open', 'canonical', 'cmaes'])
parser.add_argument('--pop', type=int, default=512, help='Population size')
parser.add_argument('--es_sigma', type=float, default=0.01, help='Standard deviation of the Gaussian distribution')
parser.add_argument('--sample_mirror', type=bool, default=True, help='Mirror sampling in ES')
parser.add_argument('--sample_rank_norm', type=bool, default=True, help='Rank normalization in ES')
parser.add_argument('--adam_optimizer', type=bool, default=True, help='Use Adam optimizer instead of SGD')
parser.add_argument('--learning_rate', type=float, default=0.01, help='Learning rate for Adam optimizer')
parser.add_argument('--l2_coefficient', type=float, default=0.02, help='L2 coefficient for Adam optimizer')

# NSES
parser.add_argument('--nses_emitter', type=bool, default=False, help='Use NSES instead of ES')
parser.add_argument('--novelty_nearest_neighbors', type=int, default=10, help='Number of nearest neighbors to use for novelty computation')

# RL
parser.add_argument('--rl', default=False, action="store_true", help='Add RL')
parser.add_argument('--actor_injection', action="store_true", default=False, help='Use actor injection')
parser.add_argument('--carlies', default=False, action="store_true", help='Add CARLIES')

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
        'policy_hidden_layer_sizes': (16, 16),
        "output": "debug"
    }
    for k, v in debug_values.items():
        setattr(args, k, v)

log_period = args.log_period
args.num_gens = args.evals // args.pop
num_loops = int(args.num_gens / log_period)

args.policy_hidden_layer_sizes = tuple(args.policy_hidden_layer_sizes)

args.algo = "PGA-ME"

args.config = f"PGA {args.pop}"
print("Parsed arguments:", args)


import os

# from IPython.display import clear_output
import functools
import time

import jax
import jax.numpy as jnp


from qdax.core.map_elites import MAPElites
from qdax.core.containers.mapelites_repertoire import compute_cvt_centroids
from qdax import environments
from qdax.tasks.brax_envs import scoring_function_brax_envs as scoring_function
from qdax.core.neuroevolution.buffers.buffer import QDTransition
from qdax.core.neuroevolution.networks.networks import MLP
from qdax.core.emitters.mutation_operators import isoline_variation
from qdax.utils.plotting import plot_map_elites_results

from qdax.core.emitters.pga_me_emitter import PGAMEConfig, PGAMEEmitter
from qdax.utils.metrics import CSVLogger, default_qd_metrics

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

#@title QD Training Definitions Fields
#@markdown ---
env_name = args.env_name
episode_length = 250 #@param {type:"integer"}
num_iterations = 4000 #@param {type:"integer"}
seed = 42 #@param {type:"integer"}
# policy_hidden_layer_sizes = (256, 256) #@param {type:"raw"}
iso_sigma = 0.005 #@param {type:"number"}
line_sigma = 0.05 #@param {type:"number"}
num_init_cvt_samples = 50000 #@param {type:"integer"}
num_centroids = 1024 #@param {type:"integer"}
min_bd = 0. #@param {type:"number"}
max_bd = 1.0 #@param {type:"number"}

#@title PGA-ME Emitter Definitions Fields
proportion_mutation_ga = 0.5

# TD3 params
env_batch_size = 100 #@param {type:"number"}
replay_buffer_size = 1000000 #@param {type:"number"}
critic_hidden_layer_size = (256, 256) #@param {type:"raw"}
critic_learning_rate = 3e-4 #@param {type:"number"}
greedy_learning_rate = 3e-4 #@param {type:"number"}
policy_learning_rate = 1e-3 #@param {type:"number"}
noise_clip = 0.5 #@param {type:"number"}
policy_noise = 0.2 #@param {type:"number"}
discount = 0.99 #@param {type:"number"}
reward_scaling = 1.0 #@param {type:"number"}
transitions_batch_size = 256 #@param {type:"number"}
soft_tau_update = 0.005 #@param {type:"number"}
num_critic_training_steps = 300 #@param {type:"number"}
num_pg_training_steps = 100 #@param {type:"number"}
policy_delay = 2 #@param {type:"number"}
#@markdown ---


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
keys = jax.random.split(subkey, num=env_batch_size)
fake_batch = jnp.zeros(shape=(env_batch_size, env.observation_size))
init_variables = jax.vmap(policy_network.init)(keys, fake_batch)

# Create the initial environment states
random_key, subkey = jax.random.split(random_key)
keys = jnp.repeat(jnp.expand_dims(subkey, axis=0), repeats=env_batch_size, axis=0)
reset_fn = jax.jit(jax.vmap(env.reset))
init_states = reset_fn(keys)

# play_reset_fn = env.reset

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
bd_extraction_fn = environments.behavior_descriptor_extractor[args.env_name]
scoring_fn = functools.partial(
    scoring_function,
    init_states=init_states,
    episode_length=args.episode_length,
    # play_reset_fn=play_reset_fn,
    play_step_fn=play_step_fn,
    behavior_descriptor_extractor=bd_extraction_fn,
)

# Get minimum reward value to make sure qd_score are positive
reward_offset = environments.reward_offset[args.env_name]

# Define a metrics function
metrics_function = functools.partial(
    default_qd_metrics,
    qd_offset=reward_offset * episode_length,
)

# Define the PG-emitter config
pga_emitter_config = PGAMEConfig(
    env_batch_size=env_batch_size,
    batch_size=transitions_batch_size,
    proportion_mutation_ga=proportion_mutation_ga,
    critic_hidden_layer_size=critic_hidden_layer_size,
    critic_learning_rate=critic_learning_rate,
    greedy_learning_rate=greedy_learning_rate,
    policy_learning_rate=policy_learning_rate,
    noise_clip=noise_clip,
    policy_noise=policy_noise,
    discount=discount,
    reward_scaling=reward_scaling,
    replay_buffer_size=replay_buffer_size,
    soft_tau_update=soft_tau_update,
    num_critic_training_steps=num_critic_training_steps,
    num_pg_training_steps=num_pg_training_steps,
    policy_delay=policy_delay,
)

# Get the emitter
variation_fn = functools.partial(
    isoline_variation, iso_sigma=iso_sigma, line_sigma=line_sigma
)

pg_emitter = PGAMEEmitter(
    config=pga_emitter_config,
    policy_network=policy_network,
    env=env,
    variation_fn=variation_fn,
)

# Instantiate MAP Elites
map_elites = MAPElites(
    scoring_function=scoring_fn,
    emitter=pg_emitter,
    metrics_function=metrics_function,
)

# Compute the centroids
centroids, random_key = compute_cvt_centroids(
    num_descriptors=env.behavior_descriptor_length,
    num_init_cvt_samples=num_init_cvt_samples,
    num_centroids=num_centroids,
    minval=min_bd,
    maxval=max_bd,
    random_key=random_key,
)

# compute initial repertoire
repertoire, emitter_state, random_key = map_elites.init(
    init_variables, centroids, random_key
)

log_period = 10
num_loops = int(num_iterations / log_period)

csv_logger = CSVLogger(
    "pgame-logs.csv",
    header=["loop", "iteration", "qd_score", "max_fitness", "coverage", "time", "generation"]
)
all_metrics = {}

# main loop
map_elites_scan_update = map_elites.scan_update

# main iterations
from tqdm import tqdm
bar = tqdm(range(num_loops))
try:
    for i in bar:
        start_time = time.time()
        # main iterations
        (repertoire, emitter_state, random_key,), metrics = jax.lax.scan(
            map_elites_scan_update,
            (repertoire, emitter_state, random_key),
            (),
            length=log_period,
        )
        timelapse = time.time() - start_time

        # log metrics
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

                csv_logger.log(logged_metrics)
        if wandb_run:
            wandb_run.log(logged_metrics)

        # Update bar
        bar.set_description(f"Gen: {gen}, qd_score: {logged_metrics['qd_score']:.2f}, max_fitness: {logged_metrics['max_fitness']:.2f}, coverage: {logged_metrics['coverage']:.2f}, time: {timelapse:.2f}")


except KeyboardInterrupt:
    print("Interrupted by user")