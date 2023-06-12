import os

try:
    os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"]
except KeyError:
    os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "1"
print("XLA memory", os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"])

from qdax.core.rl_es_parts.es_setup import setup_es, get_es_parser

parser = get_es_parser()

# parse arguments
args = parser.parse_args()

if (
    args.carlies
    or args.testrl
    or args.surrogate
    or args.spearman
    or args.actor_injection
    or args.es == "multiactor"
):
    args.rl = True
    # args.actor_injection = False

if args.es_target and not args.rl:
    raise ValueError("ES target requires RL")

if args.injection_clip and not args.actor_injection:
    raise ValueError("Injection clip requires actor injection")

if args.debug:
    # Cheap ES to debug
    debug_values = {
        # 'env_name': 'walker2d_uni',
        "episode_length": 100,
        "pop": 10,
        "evals": 100,
        "policy_hidden_layer_sizes": 16,
        "critic_hidden_layer_sizes": 16,
        "output": "debug",
        "surrogate_batch": 10,
    }
    for k, v in debug_values.items():
        setattr(args, k, v)

log_period = args.log_period
args.num_gens = args.evals // args.pop
# num_loops = int(args.num_gens / log_period)

args.policy_hidden_layer_sizes = (
    args.policy_hidden_layer_sizes,
    args.policy_hidden_layer_sizes,
)
args.critic_hidden_layer_sizes = (
    args.critic_hidden_layer_sizes,
    args.critic_hidden_layer_sizes,
)

algos = {
    "open": "OpenAI",
    "openai": "OpenAI",
    "canonical": "Canonical",
    "cmaes": "CMAES",
    "random": "Random",
    "multiactor": "Multi-actorRL",
}
args.algo = algos[args.es]

suffix = ""
if args.rl:
    suffix = "-RL"
if args.carlies:
    suffix = "-CARLIES"
if args.testrl:
    suffix = "-TestRL"
if args.surrogate:
    suffix = "-Surrogate"
if args.spearman:
    suffix = "-Spearman"
args.algo += f"{suffix}"


if args.actor_injection:
    args.algo += "-AI"

# args.config = f"ES {args.pop} - \u03C3 {args.es_sigma} - \u03B1 {args.learning_rate}"
# if args.elastic_pull > 0:
#     args.config += f" - \u03B5 {args.elastic_pull}" # \u03B5 is epsilon
# if args.surrogate:
#     args.config += f" - \u03C9 {args.surrogate_omega} ({args.surrogate_batch})" # \u03C9 is omega

print("Parsed arguments:", args)


# Import after parsing arguments
import functools
import time
from typing import Dict

import jax

print("Device count:", jax.device_count(), jax.devices())
import jax.numpy as jnp

from qdax.utils.metrics import CSVLogger
from qdax.utils.plotting import plot_map_elites_results

from qdax.core.rl_es_parts.es_utils import (
    ESMetrics,
)


EM = setup_es(args)

es = EM.es
env = EM.env
policy_network = EM.policy_network
emitter = EM.emitter
emitter_state = EM.emitter_state
repertoire = EM.repertoire
random_key = EM.random_key
wandb_run = EM.wandb_run
# scoring_fn = EM.scoring_fn

#######
# Run #
if args.output != "":
    import os

    directory = args.output

    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Directory '{directory}' created successfully!")
    else:
        print(f"Directory '{directory}' already exists!")

    plot_file = args.output + "/plot.png"
    log_file = args.output + "/log.csv"

    import json

    with open(args.output + "/config.json", "w") as file:
        json.dump(args.__dict__, file, indent=4)

# get all the fields in ESMetrics
header = ESMetrics.__dataclass_fields__.keys()

csv_logger = CSVLogger(
    log_file,
    header=[
        "loop",
        "generation",
        "qd_score",
        "max_fitness",
        "coverage",
        "time",
        "frames",
    ]
    + list(header),
)
all_metrics: Dict[str, float] = {}

# main loop
es_scan_update = es.scan_update

# main iterations
from tqdm import tqdm

bar = tqdm(range(args.evals))
evaluations = 0
gen = 0
try:
    while evaluations < args.evals:
        start_time = time.time()
        (
            repertoire,
            emitter_state,
            random_key,
        ), metrics = jax.lax.scan(
            es_scan_update,
            (repertoire, emitter_state, random_key),
            (),
            length=log_period,
        )
        timelapse = time.time() - start_time

        # log metrics
        gen += 1
        logged_metrics = {
            "time": timelapse,
            # "loop": 1 + i,
            "generation": gen,
            "frames": gen * args.episode_length * args.pop,
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

        if args.logall and args.output != "":
            output = args.output + "/gen_" + str(gen)
            print("Saving to", output)
            emitter_state.save(output)

        # Update bar
        evaluations = logged_metrics["evaluations"]
        evaluations = int(evaluations)
        # Set bar progress
        bar.update(evaluations - bar.n)
        bar.set_description(
            f"Gen: {gen}, qd_score: {logged_metrics['qd_score']:.2f}, max_fitness: {logged_metrics['max_fitness']:.2f}, coverage: {logged_metrics['coverage']:.2f}, time: {timelapse:.2f}"
        )
except KeyboardInterrupt:
    print("Interrupted by user")
finally:
    # Save
    if args.output != "":
        output = args.output + "/gen_" + str(gen)
        print("Saving to", output)
        emitter_state.save(output)

# print(logged_metrics)
for k, v in logged_metrics.items():
    print(f"{k}: {v}")

#################
# Visualisation #

import wandb

if args.plot:
    # create the x-axis array
    env_steps = jnp.arange(logged_metrics["evaluations"]) * args.episode_length

    # Check the number of dimensions of the descriptors
    if len(repertoire.descriptors.shape) == 2:
        # create the plots and the grid
        try:
            fig, axes = plot_map_elites_results(
                env_steps=env_steps,
                metrics=all_metrics,
                repertoire=repertoire,
                min_bd=args.min_bd,
                max_bd=args.max_bd,
            )

            import matplotlib.pyplot as plt

            plt.savefig(plot_file)

        except ValueError:
            print("Error plotting results")

        # Log the repertoire plot
        if wandb_run:
            try:
                from qdax.utils.plotting import plot_2d_map_elites_repertoire

                fig, ax = plot_2d_map_elites_repertoire(
                    centroids=repertoire.centroids,
                    repertoire_fitnesses=repertoire.fitnesses,
                    minval=args.min_bd,
                    maxval=args.max_bd,
                    repertoire_descriptors=repertoire.descriptors,
                )
                wandb_run.log({"archive": wandb.Image(fig)})
            except Exception:
                print("Error plotting repertoire")

    try:
        html_content = repertoire.record_video(env, policy_network)
        video_file = plot_file.replace(".png", ".html")
        with open(video_file, "w") as file:
            file.write(html_content)
        # Log the plot
        if wandb_run:
            wandb_run.log({"best_agent": wandb.Html(html_content)})
            wandb.finish()

    except Exception:
        print("Error recording video")
