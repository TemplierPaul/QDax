import jax

# parse args
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--num_runs", type=int, default=10)
parser.add_argument("--seed", type=int, default=10)
parser.add_argument("--num_steps", type=int, default=100)
args = parser.parse_args()

import os

print(f"{args.seed} - mem {os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION']}")

# Make big matrix operation
def big_matrix_op(x):
    return jax.numpy.matmul(x, x.T)

# matrix = jax.numpy.random.normal(size=(1000, 1000))
key = jax.random.PRNGKey(0)
matrix = jax.random.normal(key, shape=(1000, 1000))

for _ in range(args.num_steps):
    matrix = big_matrix_op(matrix)

print(f"{args.seed} - done")