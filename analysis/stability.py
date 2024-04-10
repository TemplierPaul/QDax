import argparse
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import json


from qdax.core.rl_es_parts.es_setup import setup_es, fill_default

import argparse


# Make legent font size bigger
plt.rcParams.update({"font.size": 30})


def net_shape(net):
    return jax.tree_map(lambda x: x.shape, net)


def scores(fitnesses, descriptors) -> jnp.ndarray:
    return fitnesses


def make_compute_canonical_update(
    sigma, sample_number, scoring_fn, es_emitter, injection=False
):
    def no_injection(sample_noise, actor, parent):
        # Applying noise
        networks = jax.tree_map(
            lambda x: jnp.repeat(x, sample_number, axis=0),
            parent,
        )
        networks = jax.tree_map(
            lambda mean, noise: mean + sigma * noise,
            networks,
            sample_noise,
        )

        norm = -jnp.inf

        return sample_noise, networks, norm

    if injection:
        # Set the emitter config params
        es_emitter._config.sample_number = sample_number
        es_emitter._config.sample_sigma = sigma
        injection_fn = es_emitter._inject_actor
    else:
        injection_fn = no_injection
    flatten_fn = es_emitter.flatten

    @jax.jit
    def compute_canonical_update(parent, actor):
        results = {}
        # add 1dim to parent
        parent = jax.tree_map(lambda x: jnp.expand_dims(x, axis=0), parent)

        # print(f"Parent, {net_shape(parent)}")
        mu = sample_number // 2
        # sample points as gaussian noise
        random_key = jax.random.PRNGKey(42)
        random_key, subkey = jax.random.split(random_key)

        sample_noise = jax.tree_map(
            lambda x: jax.random.normal(
                key=subkey,
                shape=jnp.repeat(x, sample_number, axis=0).shape,
            ),
            parent,
        )

        gradient_noise, networks, norm_factor = injection_fn(
            sample_noise,
            actor,
            parent,
        )

        # print(f"networks, {net_shape(networks)}")

        # compute fitness
        random_key, subkey = jax.random.split(random_key)
        fitnesses, _, _, _ = scoring_fn(networks, subkey)
        results["fit_std"] = jnp.std(fitnesses)
        results["fit_min"] = jnp.min(fitnesses)
        results["fit_max"] = jnp.max(fitnesses)
        results["fit_median"] = jnp.median(fitnesses)

        # Parent fitness averaged
        random_key, subkey = jax.random.split(random_key)
        # reproduce the parent network
        multi_parent = jax.tree_map(
            lambda x: jnp.repeat(x, sample_number, axis=0),
            parent,
        )
        parent_fit, _, _, _ = scoring_fn(multi_parent, subkey)
        results["parent_mean"] = jnp.mean(parent_fit)
        results["parent_std"] = jnp.std(parent_fit)
        results["parent_min"] = jnp.min(parent_fit)
        results["parent_max"] = jnp.max(parent_fit)
        results["parent_median"] = jnp.median(parent_fit)

        # compute update
        scores = fitnesses

        ranking_indices = jnp.argsort(scores, axis=0)
        ranks = jnp.argsort(ranking_indices, axis=0)
        ranks = sample_number - ranks  # Inverting the ranks

        weights = jnp.where(ranks <= mu, jnp.log(mu + 0.5) - jnp.log(ranks), 0)
        weights /= jnp.sum(weights)  # Normalizing the weights

        # Reshaping rank to match shape of genotype_noise
        weights = jax.tree_map(
            lambda x: jnp.reshape(
                jnp.repeat(weights.ravel(), x[0].ravel().shape[0], axis=0), x.shape
            ),
            gradient_noise,
        )

        # Computing the update
        # Noise is multiplied by rank
        gradient = jax.tree_map(
            lambda noise, rank: jnp.multiply(noise, rank),
            gradient_noise,
            weights,
        )
        # Noise is summed over the sample dimension and multiplied by sigma
        gradient = jax.tree_map(
            lambda x: jnp.reshape(x, (sample_number, -1)),
            gradient,
        )
        gradient = jax.tree_map(
            lambda g, p: jnp.reshape(
                jnp.sum(g, axis=0) * sigma,
                p.shape,
            ),
            gradient,
            parent,
        )

        # remove one dimension
        gradient = jax.tree_map(lambda x: x.squeeze(0), gradient)

        # flatten
        gradient = flatten_fn(gradient)

        results["gradient"] = gradient
        results["norm_factor"] = norm_factor

        return results

    return compute_canonical_update


def project(update_vec, vector):
    update_vec = jnp.dot(update_vec, vector) / jnp.dot(vector, vector)
    # print(f"update_vec, {net_shape(update_vec)}")
    return update_vec


def stability_interpolate(
    offspring, actor, n_points, sample_size, sigma, EM, injection=False, batch_size=None
):
    es_emitter = EM.emitter.es_emitter
    unflatten_fn = jax.vmap(es_emitter.unflatten)
    flatten_fn = es_emitter.flatten
    scoring_fn = EM.scoring_fn

    actor_genes = flatten_fn(actor)
    offspring_genes = flatten_fn(offspring)

    vector = actor_genes - offspring_genes

    # Interpolate from -1 to 2
    alphas = jnp.linspace(-1, 2, n_points)
    interpolated = jnp.outer(alphas, vector) + offspring_genes

    # print first component of each network
    # print("Interpolated", interpolated[:, 0])

    # print(f"networks, {net_shape(networks)}")

    compute_canonical_update = make_compute_canonical_update(
        sigma, sample_size, scoring_fn, es_emitter, injection=injection
    )

    if batch_size is None:
        networks = unflatten_fn(interpolated)

        # vmap compute_canonical_update over networks
        results = jax.vmap(
            lambda network: compute_canonical_update(network, actor),
            in_axes=0,
        )(networks)

    else:
        from tqdm import tqdm

        # vmap compute_canonical_update over genomes in batches
        # Split networks into batches
        num_batches = int(jnp.ceil(n_points / batch_size))
        # print(f"num_batches: {num_batches}")
        results = {}
        bar = tqdm(range(num_batches))
        for i in bar:
            genomes = interpolated[i * batch_size : (i + 1) * batch_size]
            # print("genomes", genomes.shape)
            networks = unflatten_fn(genomes)
            new_res = jax.vmap(
                lambda network: compute_canonical_update(network, actor),
                in_axes=0,
            )(networks)
            # Append new results to results, create list if empty
            if len(results) == 0:
                results = new_res
            else:
                results = jax.tree_map(
                    lambda x, y: jnp.concatenate([x, y], axis=0),
                    results,
                    new_res,
                )
        # update = jnp.concatenate(results, axis=0)

    # for k, v in results.items():
    #     print(f"{k}: {v.shape}")
    update = results["gradient"]

    # Evaluate the networks to have a baseline
    networks = unflatten_fn(interpolated)
    random_key = jax.random.PRNGKey(42)
    fitnesses, _, _, _ = scoring_fn(networks, random_key)

    # print(f"update, {net_shape(update)}")

    # project onto offspring-actor vector
    update_comp = jax.vmap(project, in_axes=(0, None))(update, vector)

    results["update_comp"] = update_comp
    return alphas, results, fitnesses


def pop_spread(offspring, actor, sample_size, sigma, EM, injection=False):
    parent = jax.tree_map(lambda x: jnp.expand_dims(x, axis=0), offspring)
    # print(f"parent, {net_shape(parent)}")
    # print(f"actor, {net_shape(actor)}")

    es_emitter = EM.emitter.es_emitter

    def do_injection(sample_noise, actor, parent):
        x_actor = es_emitter.flatten(actor)
        # print("x_actor shape", x_actor.shape)
        x_parent = es_emitter.flatten(parent)
        # print("x_parent shape", x_parent.shape)
        y_actor = (x_actor - x_parent) / sigma

        norm = jnp.linalg.norm(y_actor)
        # alpha clip es_emitter.c_y / norm to 1
        # norm = jnp.minimum(1, es_emitter.c_y / norm)
        norm = 1
        normed_y_actor = norm * y_actor
        normed_y_net = es_emitter.unflatten(normed_y_actor)
        # Add 1 dimension
        normed_y_net = jax.tree_map(
            lambda x: x[None, ...],
            normed_y_net,
        )

        # Applying noise
        networks = jax.tree_map(
            lambda x: jnp.repeat(x, sample_size, axis=0),
            parent,
        )
        # print("networks shape", jax.tree_map(lambda x: x.shape, networks))
        networks = jax.tree_map(
            lambda mean, noise: mean + sigma * noise,
            networks,
            sample_noise,
        )
        # print("networks shape", jax.tree_map(lambda x: x.shape, networks))

        # Repeat actor
        actor = jax.tree_util.tree_map(
            lambda x: jnp.repeat(x[None, ...], 1, axis=0),
            actor,
        )
        # print("repeated actor shape", jax.tree_map(lambda x: x.shape, actor))

        # Replace the n last one, with n = 1
        networks = jax.tree_util.tree_map(
            lambda x, y: jnp.concatenate([x[:-1], y], axis=0),
            networks,
            actor,
        )

        # replace actor in sample_noise by scaled_actor
        sample_noise = jax.tree_util.tree_map(
            lambda x, y: jnp.concatenate([x[:-1], y], axis=0),
            sample_noise,
            normed_y_net,
        )

        return sample_noise, networks, norm

    def no_injection(sample_noise, actor, parent):
        # Applying noise
        networks = jax.tree_map(
            lambda x: jnp.repeat(x, sample_size, axis=0),
            parent,
        )
        networks = jax.tree_map(
            lambda mean, noise: mean + sigma * noise,
            networks,
            sample_noise,
        )

        norm = -jnp.inf

        return sample_noise, networks, norm

    if injection:
        # Set the emitter config params
        injection_fn = do_injection
    else:
        injection_fn = no_injection
    flatten_fn = es_emitter.flatten

    # sample points as gaussian noise
    random_key = jax.random.PRNGKey(42)
    random_key, subkey = jax.random.split(random_key)

    sample_noise = jax.tree_map(
        lambda x: jax.random.normal(
            key=subkey,
            shape=jnp.repeat(x, sample_size, axis=0).shape,
        ),
        parent,
    )

    gradient_noise, networks, norm_factor = injection_fn(
        sample_noise,
        actor,
        parent,
    )

    print(f"Norm factor {norm_factor}")

    # print(f"networks, {net_shape(networks)}")

    # compute fitness
    random_key, subkey = jax.random.split(random_key)
    fitnesses, _, _, _ = scoring_fn(networks, subkey)

    # get genomes
    centered = jax.tree_map(
        lambda x, y: x - y,
        networks,
        parent,
    )

    # print(f"centered nets, {net_shape(centered)}")

    centered_genomes = jax.vmap(flatten_fn)(centered)

    # print(f"genomes, {net_shape(centered_genomes)}")

    # project onto parent-actor vector
    vector = flatten_fn(actor) - flatten_fn(parent)
    alphas = jax.vmap(project, in_axes=(0, None))(centered_genomes, vector)

    normed_actor = flatten_fn(actor) - flatten_fn(parent)
    actor_alpha = project(normed_actor, vector)
    print(f"actor_alpha: {actor_alpha}")

    return alphas, fitnesses


def stability_plot(save_path, gen, sigma, injection):
    config = f"gen {gen} | \u03C3={sigma}"
    if injection:
        config += " | Injection"
    if args.deterministic:
        config += " | Deterministic"
    else:
        config += " | Stochastic"
    print(f"Running {config}")

    offspring_genes = jnp.load(f"{save_path}/gen_{gen}_offspring.npy")
    offspring = emitter.es_emitter.unflatten(offspring_genes)

    actor_genes = jnp.load(f"{save_path}/gen_{gen}_actor.npy")
    actor = emitter.es_emitter.unflatten(actor_genes)

    alphas, results, fitnesses = stability_interpolate(
        offspring,
        actor,
        n_points=100,
        sample_size=100,
        sigma=sigma,
        EM=EM,
        injection=injection,
        batch_size=10,
    )

    # for k, v in results.items():
    #     print(f"{k}: {v.shape}")
    n_plots = 5
    fig, axs = plt.subplots(n_plots, 1, figsize=(20, 10 * n_plots))

    # Update component plot
    update_comp = results["update_comp"]
    idx = 0
    axs[idx].plot(alphas, update_comp)
    axs[idx].scatter(alphas, update_comp, label="Update direction")
    axs[idx].axvline(x=0, color="red")
    axs[idx].axvline(x=1, color="green")
    axs[idx].axhline(y=0, color="black", linestyle="dotted", label="Stability")
    axs[idx].legend()
    axs[idx].set_xlabel("Alpha")
    axs[idx].set_ylabel("Update component")
    title = f"Update component along interpolation \n{config}"
    axs[idx].set_title(title)

    # Fitness plot
    # fitnesses = results["fitnesses"]
    idx = 1
    axs[idx].plot(alphas, fitnesses)
    axs[idx].scatter(alphas, fitnesses, label="Fitness")

    pop_alphas, pop_fitnesses = pop_spread(
        offspring, actor, sample_size=100, sigma=sigma, EM=EM, injection=injection
    )
    axs[idx].scatter(pop_alphas, pop_fitnesses, label="Population fitness")
    axs[idx].axvline(x=0, color="red")
    axs[idx].axvline(x=1, color="green")
    axs[idx].legend()
    axs[idx].set_xlabel("Alpha")
    axs[idx].set_ylabel("Fitness")
    title = f"Fitness landscape | Population distribution \n{config}"
    axs[idx].set_title(title)

    # Fitness plot
    idx = 2
    axs[idx].plot(alphas, results["parent_min"], c="red")
    axs[idx].scatter(alphas, results["parent_min"], c="red", label="Parent min fitness")
    axs[idx].plot(alphas, results["parent_median"], c="blue")
    axs[idx].scatter(
        alphas, results["parent_median"], c="blue", label="Parent median fitness"
    )
    axs[idx].plot(alphas, results["parent_max"], c="green")
    axs[idx].scatter(
        alphas, results["parent_max"], c="green", label="Parent max fitness"
    )
    axs[idx].axvline(x=0, color="red")
    axs[idx].axvline(x=1, color="green")
    axs[idx].legend()
    axs[idx].set_xlabel("Alpha")
    axs[idx].set_ylabel("Fitness")
    title = f"Parent fitness distribution \n{config}"
    axs[idx].set_title(title)

    # Noise
    idx = 3
    axs[idx].plot(alphas, results["parent_std"], c="red")
    axs[idx].scatter(alphas, results["parent_std"], c="red", label="Parent fitness std")
    axs[idx].axvline(x=0, color="red")
    axs[idx].axvline(x=1, color="green")
    axs[idx].legend()
    axs[idx].set_xlabel("Alpha")
    axs[idx].set_ylabel("Fitness")
    title = f"Fitness noise on center \n{config}"
    axs[idx].set_title(title)

    # Population fitness plot
    idx = 4
    axs[idx].plot(alphas, results["fit_min"], c="red")
    axs[idx].scatter(alphas, results["fit_min"], c="red", label="Pop min fitness")
    axs[idx].plot(alphas, results["fit_median"], c="blue")
    axs[idx].scatter(
        alphas, results["fit_median"], c="blue", label="Pop median fitness"
    )
    axs[idx].plot(alphas, results["fit_max"], c="green")
    axs[idx].scatter(alphas, results["fit_max"], c="green", label="Pop max fitness")
    axs[idx].axvline(x=0, color="red")
    axs[idx].axvline(x=1, color="green")
    axs[idx].legend()
    axs[idx].set_xlabel("Alpha")
    axs[idx].set_ylabel("Fitness")
    title = f"Population fitness std \n{config}"
    axs[idx].set_title(title)
    config_string = config.replace("| ", "_").replace(" ", "_")
    plt.savefig(f"{save_path}/stability_{config_string}.png")

    # plt.show()


def stability_subplots(save_path, gen, sigma, injection):
    config = f"gen {gen} | \u03C3={sigma}"
    if injection:
        config += " | Injection"
    if args.deterministic:
        config += " | Deterministic"
    else:
        config += " | Stochastic"
    print(f"Running {config}")

    offspring_genes = jnp.load(f"{save_path}/gen_{gen}_offspring.npy")
    offspring = emitter.es_emitter.unflatten(offspring_genes)

    actor_genes = jnp.load(f"{save_path}/gen_{gen}_actor.npy")
    actor = emitter.es_emitter.unflatten(actor_genes)

    alphas, results, fitnesses = stability_interpolate(
        offspring,
        actor,
        n_points=100,
        sample_size=100,
        sigma=sigma,
        EM=EM,
        injection=injection,
        batch_size=10,
    )
    # Make subplots in separate figures and save them

    # Update component plot
    update_comp = results["update_comp"]
    fig, ax = plt.subplots(1, 1, figsize=(20, 10))
    ax.plot(alphas, update_comp)
    ax.scatter(alphas, update_comp, label="Update direction")
    ax.axvline(x=0, color="red")
    ax.axvline(x=1, color="green")
    ax.axhline(y=0, color="black", linestyle="dotted", label="Stability")
    ax.legend()
    ax.set_xlabel("Alpha")
    ax.set_ylabel("Update component")
    title = f"Update component along interpolation"
    ax.set_title(title)
    # config_string = config.replace("| ", "_").replace(" ", "_") + "_update"
    config_string = f"stability_gen_{gen}"
    if injection:
        config_string += "_injection"
    config_string += "_update"
    plt.savefig(f"{save_path}/stability_{config_string}.png")

    # Fitness plot
    # fitnesses = results["fitnesses"]
    fig, ax = plt.subplots(1, 1, figsize=(20, 10))
    ax.plot(alphas, fitnesses)
    ax.scatter(alphas, fitnesses, label="Fitness")

    pop_alphas, pop_fitnesses = pop_spread(
        offspring, actor, sample_size=100, sigma=sigma, EM=EM, injection=injection
    )
    # ax.scatter(pop_alphas, pop_fitnesses, label="Population fitness")
    ax.axvline(x=0, color="red")
    ax.axvline(x=1, color="green")
    ax.legend()
    ax.set_xlabel("Alpha")
    ax.set_ylabel("Fitness")
    title = f"Fitness landscape | Population distribution"
    ax.set_title(title)
    # config_string = config.replace("| ", "_").replace(" ", "_") + "_fitness"
    config_string = f"stability_gen_{gen}"
    if injection:
        config_string += "_injection"
    config_string += "_fitness"
    plt.savefig(f"{save_path}/stability_{config_string}.png")

    # Fitness plot
    # fig, ax = plt.subplots(1, 1, figsize=(20, 10))
    # ax.plot(alphas, results["parent_min"], c="red")
    # ax.scatter(alphas, results["parent_min"], c="red", label="Parent min fitness")
    # ax.plot(alphas, results["parent_median"], c="blue")
    # ax.scatter(
    #     alphas, results["parent_median"], c="blue", label="Parent median fitness"
    # )
    # ax.plot(alphas, results["parent_max"], c="green")
    # ax.scatter(alphas, results["parent_max"], c="green", label="Parent max fitness")
    # ax.axvline(x=0, color="red")
    # ax.axvline(x=1, color="green")
    # ax.legend()
    # ax.set_xlabel("Alpha")
    # ax.set_ylabel("Fitness")
    # title = f"Parent fitness distribution"
    # ax.set_title(title)
    # config_string = config.replace("| ", "_").replace(" ", "_") + "_parent"
    # plt.savefig(f"{save_path}/stability_{config_string}.png")

    # Noise
    # fig, ax = plt.subplots(1, 1, figsize=(20, 10))
    # ax.plot(alphas, results["parent_std"], c="red")
    # ax.scatter(alphas, results["parent_std"], c="red", label="Parent fitness std")
    # ax.axvline(x=0, color="red")
    # ax.axvline(x=1, color="green")
    # ax.legend()
    # ax.set_xlabel("Alpha")
    # ax.set_ylabel("Fitness")
    # title = f"Fitness noise on center"
    # ax.set_title(title)
    # config_string = config.replace("| ", "_").replace(" ", "_") + "_noise"
    # plt.savefig(f"{save_path}/stability_{config_string}.png")

    # Population fitness plot
    fig, ax = plt.subplots(1, 1, figsize=(20, 10))
    ax.plot(alphas, results["fit_min"], c="red")
    ax.scatter(alphas, results["fit_min"], c="red", label="Pop min fitness")
    ax.plot(alphas, results["fit_median"], c="blue")
    ax.scatter(alphas, results["fit_median"], c="blue", label="Pop median fitness")
    ax.plot(alphas, results["fit_max"], c="green")
    ax.scatter(alphas, results["fit_max"], c="green", label="Pop max fitness")
    ax.axvline(x=0, color="red")
    ax.axvline(x=1, color="green")
    ax.legend()
    ax.set_xlabel("Alpha")
    ax.set_ylabel("Fitness")
    title = f"Population fitness std"
    ax.set_title(title)
    # config_string = config.replace("| ", "_").replace(" ", "_") + "_pop"
    config_string = f"stability_gen_{gen}"
    if injection:
        config_string += "_injection"
    config_string += "_pop"
    plt.savefig(f"{save_path}/stability_{config_string}.png")


if __name__ == "__main__":
    # parse first cli argument
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "save_path", type=str, help="Path to the folder containing the config.json"
    )
    # List of generations to plot
    parser.add_argument(
        "--gens", type=int, nargs="+", help="Generations to plot", default=None
    )
    args = parser.parse_args()
    save_path = args.save_path
    print(args)
    print(args.gens)

    # get all numbers in gen_nums that are 1 or a multiple of 100
    gens = args.gens
    if gens is None:
        # get number of generations
        import glob
        import re

        gen_files = glob.glob(save_path + "/gen_*.npy")
        # print(gen_files)
        gen_nums = [int(re.findall(r"\d+", f.split("/")[-1])[0]) for f in gen_files]
        # get unique
        gen_nums = list(set(gen_nums))
        gen_nums.sort()
        print(f"Found {len(gen_nums)} generations, max gen: {max(gen_nums)}")

        gens = [g for g in gen_nums if g == 1 or g % 100 == 0]

    print(f"Found {len(gens)} generations: {gens}")

    # Get last one
    gen = gens[-1]
    print(f"Plotting generation {gen}")

    # Load config
    print(save_path + "/config.json")
    with open(save_path + "/config.json", "r") as f:
        args = json.load(f)
        # Lists to tuples
        for k, v in args.items():
            if isinstance(v, list):
                args[k] = tuple(v)
        args = argparse.Namespace(**args)
        args.wandb = ""
        print(args)

    config_name = args.config
    if "TD3" not in config_name:
        # Exit
        print("Not a TD3 config, exiting")
        exit()

    default = {"surrogate_batch": 1024, "surrogate": False}
    # Replace missing values with default
    for k, v in default.items():
        if not hasattr(args, k):
            setattr(args, k, v)

    args = fill_default(args)
    EM = setup_es(args)

    es = EM.es
    env = EM.env
    policy_network = EM.policy_network
    emitter = EM.emitter
    emitter_state = EM.emitter_state
    repertoire = EM.repertoire
    random_key = EM.random_key
    wandb_run = EM.wandb_run
    scoring_fn = EM.scoring_fn

    sigma = args.es_sigma

    for injection in [True, False]:
        # stability_plot(save_path, gen, sigma, injection)
        stability_subplots(save_path, gen, sigma, injection)
