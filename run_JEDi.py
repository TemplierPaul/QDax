#@title Installs and Imports
# !pip install ipympl |tail -n 1
# %matplotlib widget
# from google.colab import output
# output.enable_custom_widget_manager()

import os

from IPython.display import clear_output
import functools
import time

import jax
import jax.numpy as jnp

try:
    import brax
except:
    # !pip install git+https://github.com/google/brax.git@v0.0.15 |tail -n 1
    import brax

try:
    import qdax
except:
    # !pip install --no-deps git+https://github.com/adaptive-intelligent-robotics/QDax@main |tail -n 1
    import qdax


from qdax.core.map_elites import MAPElites
from qdax.core.containers.mapelites_repertoire import compute_cvt_centroids, MapElitesRepertoire
from qdax import environments
from qdax.tasks.brax_envs import scoring_function_brax_envs as scoring_function
from qdax.tasks.brax_envs import reset_based_scoring_function_brax_envs

from qdax.core.neuroevolution.buffers.buffer import QDTransition
from qdax.core.neuroevolution.networks.networks import MLP
from qdax.core.emitters.mutation_operators import isoline_variation
from qdax.core.emitters.standard_emitters import MixingEmitter
from qdax.utils.plotting import plot_map_elites_results

from qdax.utils.metrics import CSVLogger, default_qd_metrics

from jax.flatten_util import ravel_pytree

from brax.io import html


if "COLAB_TPU_ADDR" in os.environ:
  from jax.tools import colab_tpu
  colab_tpu.setup_tpu()

# Argparse
import argparse
parser = argparse.ArgumentParser(description='JEDi')
parser.add_argument('--env', type=str, default="pointmaze", help='environment name', target='env_name')
parser.add_argument('--batch_size', type=int, default=100, help='batch size', target='batch_size')
parser.add_argument('--episode_length', type=int, default=1000, help='episode length', target='episode_length')
parser.add_argument('--num_iterations', type=int, default=10000, help='number of iterations', target='num_iterations')
parser.add_argument('--seed', type=int, default=42, help='seed', target='seed')
parser.add_argument('--policy_hidden_layer_sizes', type=int, nargs='+', default=(64, 64), help='policy hidden layer sizes', target='policy_hidden_layer_sizes')
parser.add_argument('--iso_sigma', type=float, default=0.005, help='isoline sigma', target='iso_sigma')
parser.add_argument('--line_sigma', type=float, default=0.05, help='line sigma', target='line_sigma')
parser.add_argument('--num_init_cvt_samples', type=int, default=50000, help='number of initial samples', target='num_init_cvt_samples')
parser.add_argument('--num_centroids', type=int, default=1024, help='number of centroids', target='num_centroids')
parser.add_argument('--min_bd', type=float, default=0.0, help='minimum bound', target='min_bd')
parser.add_argument('--max_bd', type=float, default=1.0, help='maximum bound', target='max_bd')

# robert bool: set true if using robert
parser.add_argument('--robert', default=False, action="store_true", help='Use Evosax networks')
parser.add_argument('--algo', type=str, default="mapelites", help='Algorithm')

args = parser.parse_args()

plot_maze = False
if args.env_name == "pointmaze":
    min_bd = -1.0
    episode_length = 100
    plot_maze = True

if args.env_name == "antmaze":
    min_bd, max_bd = [-5, 40]
    episode_length = 250
    plot_maze = False

if args.robert:
    policy_hidden_layer_sizes = (32,) * 4
    activation = jnp.tanh
else:
    policy_hidden_layer_sizes = (64, 64)
    activation = jax.nn.relu

print(policy_hidden_layer_sizes)