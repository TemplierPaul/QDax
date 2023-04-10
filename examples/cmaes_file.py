from jax.config import config
# config.update("jax_enable_x64", True)

import jax
import jax.numpy as jnp

import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

from qdax.core.cmaes import CMAES
from qdax.core.cmaes_sep import SepCMAES

import jax.profiler

import gc

import os
# os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"]="false"
# os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"]="0.9"
# os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"]="platform"

# ## Set the hyperparameters

#@title Hyperparameters
#@markdown ---
num_iterations = 100 #@param {type:"integer"}
num_dimensions = 100 #@param {type:"integer"}
batch_size = 36 #@param {type:"integer"}
num_best = 18 #@param {type:"integer"}
sigma_g = 0.5 # 0.5 #@param {type:"number"}
minval = -5.12 #@param {type:"number"}
optim_problem = "rastrigin" #@param["rastrigin", "sphere"]
#@markdown ---

# ## Define the fitness function - choose rastrigin or sphere

def rastrigin_scoring(x: jnp.ndarray):
    first_term = 10 * x.shape[-1]
    second_term = jnp.sum((x + minval * 0.4) ** 2 - 10 * jnp.cos(2 * jnp.pi * (x + minval * 0.4)))
    return -(first_term + second_term)

def sphere_scoring(x: jnp.ndarray):
    return -jnp.sum((x + minval * 0.4) * (x + minval * 0.4), axis=-1)

if optim_problem == "sphere":
    fitness_fn = sphere_scoring
elif optim_problem == "rastrigin":
    fitness_fn = jax.vmap(rastrigin_scoring)
else:
    raise Exception("Invalid opt function name given")

# ## Define a CMA-ES optimizer instance

cmaes = SepCMAES(
    population_size=batch_size,
    num_best=num_best,
    search_dim=num_dimensions,
    fitness_function=fitness_fn,
    mean_init=jnp.zeros((num_dimensions,)),
    init_sigma=sigma_g,
    delay_eigen_decomposition=True,
)

# ## Init the CMA-ES optimizer state

state = cmaes.init()
random_key = jax.random.PRNGKey(0)

# ## Run optimization iterations
means = [state.mean]

iteration_count = 0
from tqdm import tqdm
for it in tqdm(range(num_iterations)):
    iteration_count += 1
    
    print("Loop Mean shape", state.mean.shape)
    print("Cov:", state.cov_vector.shape)
    if len(state.mean.shape) > 1:
        raise Exception("Mean shape is too big")
    
    # Check nan in mean
    if jnp.any(jnp.isnan(state.mean)):
        raise Exception("Mean has nan")
    
    # sample
    samples, random_key = cmaes.sample(state, random_key)

    print("Loop Samples shape", samples.shape)
    
    # udpate
    state = cmaes.update(state, samples)
    print(state.mean)
    print("sigma", state.sigma)
    print("_c_s", cmaes._c_s)
    print("_d_s", cmaes._d_s)
    print("_chi", cmaes._chi)
    
    # check stop condition
    stop_condition = cmaes.stop_condition(state)

    if stop_condition:
        break

    # store data for plotting
    means.append(state.mean)
        
print("Num iterations before stop condition: ", iteration_count)

# ## Check final fitnesses and distribution mean

# checking final fitness values
fitnesses = fitness_fn(samples)

print("Min fitness in the final population: ", jnp.min(fitnesses))
print("Mean fitness in the final population: ", jnp.mean(fitnesses))
print("Max fitness in the final population: ", jnp.max(fitnesses))

# checking mean of the final distribution
print("Final mean of the distribution: \n", means[-1])
# print("Final covariance matrix of the distribution: ", covs[-1])

# ## Visualization of the optimization trajectory

# fig, ax = plt.subplots(figsize=(12, 6))

# # sample points to show fitness landscape
# random_key, subkey = jax.random.split(random_key)
# x = jax.random.uniform(subkey, minval=-4, maxval=8, shape=(100000, 2))
# f_x = fitness_fn(x)

# # plot fitness landscape
# points = ax.scatter(x[:, 0], x[:, 1], c=f_x, s=0.1)
# fig.colorbar(points)

# # plot cma-es trajectory
# traj_min = 0
# traj_max = iteration_count
# for mean, cov in zip(means[traj_min:traj_max], covs[traj_min:traj_max]):
#     ellipse = Ellipse((mean[0], mean[1]), cov[0, 0], cov[1, 1], fill=False, color='k', ls='--')
#     ax.add_patch(ellipse)
#     ax.plot(mean[0], mean[1], color='k', marker='x')
    
# ax.set_title(f"Optimization trajectory of CMA-ES between step {traj_min} and step {traj_max}")
# plt.show()



