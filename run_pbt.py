import argparse
import os

try:
    os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"]
except KeyError:
    os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "1"

import functools
import time
from typing import Dict

import jax

import jax.numpy as jnp
from tqdm import tqdm

from qdax import environments

from qdax.baselines.pbt import PBT
from qdax.baselines.td3_pbt import PBTTD3, PBTTD3Config

print("Device count:", jax.device_count(), jax.devices())
print("XLA memory", os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"])

jax.config.update("jax_platform_name", "cpu")
devices = jax.devices("gpu")
num_devices = len(devices)
print(f"Detected the following {num_devices} device(s): {devices}")

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
parser.add_argument(
    "--policy_hidden_layer_sizes",
    type=int,
    default=128,
    help="Policy network hidden layer sizes",
)
parser.add_argument(
    "--critic_hidden_layer_sizes",
    type=int,
    default=128,
    help="critic network hidden layer sizes",
)
parser.add_argument(
    "--deterministic", default=False, action="store_true", help="Fixed init state"
)

# PBT
parser.add_argument(
    "--pbt_pop", type=int, default=3, help="PBT population size per device"
)
parser.add_argument(
    "--pbt_batch", type=int, default=250, help="PBT population size per device"
)

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
parser.add_argument("--debug", default=False, action="store_true", help="Debug flag")
parser.add_argument(
    "--logall", default=False, action="store_true", help="Lot at each generation"
)

# parse arguments
args = parser.parse_args()

args.algo = "PBT-TD3"

args.config = f"PBT-TD3 {args.pbt_pop}"

print("Parsed arguments:", args)

import wandb

print("Imported modules")

entity = None
project = args.wandb
wandb_run = None
if project != "":
    if "/" in project:
        entity, project = project.split("/")
    wandb_run = wandb.init(project=project, entity=entity, config={**vars(args)})

    print("Initialized wandb")

env_name = args.env_name
seed = args.seed

env_batch_size = args.pbt_batch

population_size_per_device = args.pbt_pop
population_size = population_size_per_device * num_devices
num_steps = args.evals
warmup_steps = 0
buffer_size = 100000

# PBT Config
num_best_to_replace_from = 1
num_worse_to_replace = 1

# TD3 config
episode_length: int = 1000
batch_size: int = 256
policy_delay: int = 2
grad_updates_per_step: float = 1
soft_tau_update: float = 0.005

policy_hidden_layer_size = (
    args.policy_hidden_layer_sizes,
    args.policy_hidden_layer_sizes,
)
critic_hidden_layer_size = (
    args.critic_hidden_layer_sizes,
    args.critic_hidden_layer_sizes,
)

num_loops = 10
print_freq = 1

# Initialize environments
env = environments.create(
    env_name=env_name,
    batch_size=env_batch_size * population_size_per_device,
    episode_length=episode_length,
    auto_reset=True,
)

eval_env = environments.create(
    env_name=env_name,
    batch_size=env_batch_size * population_size_per_device,
    episode_length=episode_length,
    auto_reset=True,
)


@jax.jit
def init_environments(random_key):
    env_states = jax.jit(env.reset)(rng=random_key)
    eval_env_first_states = jax.jit(eval_env.reset)(rng=random_key)

    reshape_fn = jax.jit(
        lambda tree: jax.tree_map(
            lambda x: jnp.reshape(
                x,
                (
                    population_size_per_device,
                    env_batch_size,
                )
                + x.shape[1:],
            ),
            tree,
        ),
    )
    env_states = reshape_fn(env_states)
    eval_env_first_states = reshape_fn(eval_env_first_states)

    return env_states, eval_env_first_states


key = jax.random.PRNGKey(seed)
key, *keys = jax.random.split(key, num=1 + num_devices)
keys = jnp.stack(keys)
env_states, eval_env_first_states = jax.pmap(
    init_environments, axis_name="p", devices=devices
)(keys)

# get agent
config = PBTTD3Config(
    episode_length=episode_length,
    batch_size=batch_size,
    policy_delay=policy_delay,
    soft_tau_update=soft_tau_update,
    critic_hidden_layer_size=critic_hidden_layer_size,
    policy_hidden_layer_size=policy_hidden_layer_size,
)

agent = PBTTD3(config=config, action_size=env.action_size)

# get the initial training states and replay buffers
agent_init_fn = agent.get_init_fn(
    population_size=population_size_per_device,
    action_size=env.action_size,
    observation_size=env.observation_size,
    buffer_size=buffer_size,
)
keys, training_states, replay_buffers = jax.pmap(
    agent_init_fn, axis_name="p", devices=devices
)(keys)

eval_policy = jax.pmap(agent.get_eval_fn(eval_env), axis_name="p", devices=devices)

# eval policy before training
population_returns, _ = eval_policy(training_states, eval_env_first_states)
population_returns = jnp.reshape(population_returns, (population_size,))
print(
    f"Evaluation over {env_batch_size} episodes,"
    f" Population mean return: {jnp.mean(population_returns)},"
    f" max return: {jnp.max(population_returns)}"
)

# get training function
num_iterations = args.episode_length
num_loops = num_steps // env_batch_size // num_iterations

print(f"Training for {num_loops} loops with {num_iterations} iterations each")
print(f"Total training steps: {num_loops * num_iterations}")

train_fn = agent.get_train_fn(
    env=env,
    num_iterations=num_iterations,
    env_batch_size=env_batch_size,
    grad_updates_per_step=grad_updates_per_step,
)
train_fn = jax.pmap(train_fn, axis_name="p", devices=devices)

pbt = PBT(
    population_size=population_size,
    num_best_to_replace_from=num_best_to_replace_from // num_devices,
    num_worse_to_replace=num_worse_to_replace // num_devices,
)
select_fn = jax.pmap(pbt.update_states_and_buffer_pmap, axis_name="p", devices=devices)


@jax.jit
def unshard_fn(sharded_tree):
    tree = jax.tree_map(lambda x: jax.device_put(x, "cpu"), sharded_tree)
    tree = jax.tree_map(
        lambda x: jnp.reshape(x, (population_size,) + x.shape[2:]), tree
    )
    return tree


for i in tqdm(range(num_loops), total=num_loops):
    # Update for num_steps
    (training_states, env_states, replay_buffers), metrics = train_fn(
        training_states, env_states, replay_buffers
    )
    # print(metrics)
    evaluations = i * env_batch_size * population_size
    num_frames = evaluations * episode_length
    print(f"{num_frames} frames, {evaluations} episodes")

    # Eval policy after training
    population_returns, _ = eval_policy(training_states, eval_env_first_states)
    population_returns_flatten = jnp.reshape(population_returns, (population_size,))

    print(population_returns_flatten.shape)
    min_fit = jnp.min(population_returns_flatten)
    max_fit = jnp.max(population_returns_flatten)
    mean_fit = jnp.mean(population_returns_flatten)
    std_fit = jnp.std(population_returns_flatten)

    # wandb logging
    if wandb_run is not None:
        wandb.log(
            {
                "pbt_min": min_fit,
                "pbt_max": max_fit,
                "pbt_mean": mean_fit,
                "pbt_std": std_fit,
            }
        )

    if i % print_freq == 0:
        print(
            f"Evaluation over {env_batch_size} episodes,"
            f" Population mean return: {mean_fit},"
            f" max return: {max_fit}"
        )

    # PBT selection
    if i < (num_loops - 1):
        keys, training_states, replay_buffers = select_fn(
            keys, population_returns, training_states, replay_buffers
        )
