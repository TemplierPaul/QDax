from qdax.utils.plotting import plot_map_elites_results
from qdax.jedi.plotting import plot_jedi_results
from qdax.jedi.plotting import plot_2d_count, scatter_count

import matplotlib.pyplot as plt

import importlib

# algo_name = "cmame"
# algo_name = "mapelites"
algo_name = "jedi"

env_name = "pointmaze"  # @param['ant_uni', 'hopper_uni', 'walker2d_uni', 'halfcheetah_uni', 'humanoid_uni', 'ant_omni', 'humanoid_omni']
env_name = "halfcheetah_uni"  # @param['ant_uni', 'hopper_uni', 'walker2d_uni', 'halfcheetah_uni', 'humanoid_uni', 'ant_omni', 'humanoid_omni']
episode_length = 1000  # @param {type:"integer"}
policy_hidden_layer_sizes = (64, 64)  # @param {type:"raw"}
seed = 42  # @param {type:"integer"}
num_centroids = 1024  # @param {type:"integer"}
# num_centroids = 8192 #@param {type:"integer"}
num_init_cvt_samples = 50000  # @param {type:"integer"}
min_bd = 0.0  # @param {type:"number"}
max_bd = 1.0  # @param {type:"number"}

batch_size = 100  # @param {type:"number"}
num_iterations = 1000  # @param {type:"integer"}

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
    "nb_bd": 2,
}

if algo_name == "jedi":
    ## JEDi
    import qdax.jedi.jedi_factory as factory_module

    importlib.reload(factory_module)

    JEDiFactory = factory_module.JEDiFactory
    factory = JEDiFactory(config)

    jedi_config = {
        "explore": True,
        "exploit": True,
        "crowding": True,
        "aria": True,
        "sample_number": 10,
        "sigma": 0.1,
        "es_gens": 100,
        "macro_loops": 10,
        "add_population": True,
    }

    factory.get_explore_exploit(jedi_config)
    # factory.get_aria(jedi_config)

    print(factory)

    ## Run
    repertoire, all_metrics = factory.run()

    print(f"{factory} ran successfully!")

    # QD stats
    title = f"JEDi - {env_name}"
    fig, ax = plot_jedi_results(
        repertoire, all_metrics, title, min_bd, max_bd, log_scale=True
    )
    file_name = f"plots/{env_name}_jedi.png"
    fig.savefig(file_name, bbox_inches="tight")

    # GP shape
    fig, ax = factory.plot_gp()
    file_name = f"plots/{env_name}_gp.png"
    fig.savefig(file_name, bbox_inches="tight")
    plt.suptitle(title, fontsize=16)

    # Cost distribution
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    axes[0] = plot_2d_count(repertoire, min_bd, max_bd, log_scale=True, ax=axes[0])
    axes[1] = scatter_count(repertoire, log_scale=True, ax=axes[1])
    plt.suptitle(title, fontsize=16)
    file_name = f"plots/{env_name}_jedi_count.png"
    fig.savefig(file_name, bbox_inches="tight")

else:
    # MAP-Elites
    import qdax.utils.algo_factory as factory_module

    importlib.reload(factory_module)

    MEFactory = factory_module.MEFactory
    factory = MEFactory(config)

    algos = {
        "mapelites": factory.get_mapelites,
        "cmame": factory.get_cmame,
        "pgame": factory.get_pgame,
    }

    algos[algo_name]()

    print(factory)

    ## Run
    repertoire, all_metrics = factory.run()

    print(f"{factory} ran successfully!")

    env_steps = all_metrics["env_steps"]
    fig, axes = plot_map_elites_results(
        env_steps=env_steps,
        metrics=all_metrics,
        repertoire=repertoire,
        min_bd=min_bd,
        max_bd=max_bd,
    )
    plt.suptitle(str(factory), fontsize=16)

    file_name = f"plots/{env_name}_{factory.me_type}.png"
    fig.savefig(file_name, bbox_inches="tight")

    # Cost distribution
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    axes[0] = plot_2d_count(repertoire, min_bd, max_bd, log_scale=True, ax=axes[0])
    axes[1] = scatter_count(repertoire, log_scale=True, ax=axes[1])
    title = f"{env_name} - {factory.me_type}"
    plt.suptitle(title, fontsize=16)
    file_name = f"plots/{env_name}_{factory.me_type}_count.png"
    fig.savefig(file_name, bbox_inches="tight")

plt.show()
