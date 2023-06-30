from qdax.utils.plotting import plot_map_elites_results
import matplotlib.pyplot as plt
import matplotlib

# @title QD Training Definitions Fields
# @markdown ---
batch_size = 100  # @param {type:"number"}
env_name = "pointmaze"  # @param['ant_uni', 'hopper_uni', 'walker2d_uni', 'halfcheetah_uni', 'humanoid_uni', 'ant_omni', 'humanoid_omni']
# env_name = 'pointmaze'#@param['ant_uni', 'hopper_uni', 'walker2d_uni', 'halfcheetah_uni', 'humanoid_uni', 'ant_omni', 'humanoid_omni']
episode_length = 1000  # @param {type:"integer"}
num_iterations = 100  # @param {type:"integer"}
seed = 42  # @param {type:"integer"}
policy_hidden_layer_sizes = (64, 64)  # @param {type:"raw"}
# iso_sigma = 0.005 #@param {type:"number"}
# line_sigma = 0.05 #@param {type:"number"}
num_init_cvt_samples = 50000  # @param {type:"integer"}
num_centroids = 1024  # @param {type:"integer"}
# num_centroids = 8192 #@param {type:"integer"}
min_bd = 0.0  # @param {type:"number"}
max_bd = 1.0  # @param {type:"number"}
# @markdown ---

plot_maze = False
robert = True

if env_name == "pointmaze":
    min_bd = -1.0
    episode_length = 100
    plot_maze = True

if env_name == "antmaze":
    min_bd, max_bd = [-5, 40]
    episode_length = 250
    plot_maze = False

if robert:
    policy_hidden_layer_sizes = (32,) * 4
    activation = "tanh"
else:
    policy_hidden_layer_sizes = (64, 64)
    activation = "relu"

print(policy_hidden_layer_sizes)

# As a dict
config = {
    "batch_size": batch_size,
    "env_name": env_name,
    "episode_length": episode_length,
    "num_iterations": num_iterations,
    "seed": seed,
    "policy_hidden_layer_sizes": policy_hidden_layer_sizes,
    "num_init_cvt_samples": num_init_cvt_samples,
    "num_centroids": num_centroids,
    "min_bd": min_bd,
    "max_bd": max_bd,
    "activation": activation,
}

import qdax.utils.algo_factory as factory_module

import importlib

importlib.reload(factory_module)

MEFactory = factory_module.MEFactory
factory = MEFactory(config)
# print(factory)

# print("CMAME")
# factory.get_mapelites()
factory.get_cmame()
print(factory)

repertoire, all_metrics = factory.run()

print("CMAME ran successfully!")

env_steps = all_metrics["env_steps"]
fig, axes = plot_map_elites_results(
    env_steps=env_steps,
    metrics=all_metrics,
    repertoire=repertoire,
    min_bd=min_bd,
    max_bd=max_bd,
)
plt.suptitle("MAP-Elites")

plt.show()
