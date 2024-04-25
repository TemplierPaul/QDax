import argparse
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from qdax.core.rl_es_parts.es_setup import setup_es, fill_default
import glob
import re
from matplotlib.animation import FuncAnimation, PillowWriter
import jax.numpy as jnp
import json

FIT_NORM = {
    "halfcheetah_uni": (-2000, 5000),
    "walker2d_uni": (0, 4000),
}


def net_shape(net):
    return jax.tree_map(lambda x: x.shape, net)


# def multi_eval(nets, scoring_fn, n_eval):
#     fitnesses = []
#     key = jax.random.PRNGKey(0)
#     from tqdm import tqdm

#     for g in tqdm(range(n_eval)):
#         f, _, _, _ = scoring_fn(nets, key)
#         fitnesses.append(f)
#     return jnp.array(fitnesses).mean(axis=0)


class Path:
    def __init__(self, file_regex, name=None):
        self.file_regex = file_regex
        self.name = name
        self.files = glob.glob(file_regex)
        # Get last number in each file name
        self.genomes = {}
        self.gens = []
        for f in self.files:
            gen = int(re.findall(r"\d+", f.split("/")[-1])[-1])
            self.genomes[gen] = {"genome": jnp.load(f)}
            self.gens.append(gen)
        self.gens.sort()
        self.projected = None
        self.fitnesses = None

    def __repr__(self):
        return f"Path ({self.name}) with {len(self.genomes)} genomes"

    def __str__(self):
        return self.__repr__()

    def copy(self):
        new_path = Path(self.file_regex, self.name)
        return new_path

    def export_genomes(self):
        return jnp.array([self.genomes[g]["genome"] for g in self.gens])

    def transform(self, pca):
        genomes = self.export_genomes()
        self.projected = pca.transform(genomes)
        return self.projected

    def plot(self, ax, c, max_gen=None):
        if self.projected is None:
            raise ValueError("No projection computed")
        proj = self.projected
        if max_gen is not None:
            keep_gens = [g for g in self.gens if g <= max_gen]
            proj = proj[: len(keep_gens)]
        ax.plot(proj[:, 0], proj[:, 1], label=self.name, c=c)
        ax.scatter(proj[-1, 0], proj[-1, 1], marker="x", c=c)

    def evaluate(self, unflatten_fn, scoring_fn, key):
        genomes = self.export_genomes()
        nets = unflatten_fn(genomes)
        # key = jax.random.PRNGKey(0)
        self.fitnesses, descriptors, extra_scores, random_key = scoring_fn(nets, key)
        return self.fitnesses


from sklearn.decomposition import PCA
import plotly.graph_objs as go
import numpy as np
import plotly.offline as pyo

from sklearn.decomposition import PCA
import plotly.graph_objs as go
import numpy as np
import plotly.offline as pyo

colors = ["orange", "pink"]


class PathPCA:
    def __init__(self, paths, start_gen=0, end_gen=None, pca_dim=2):
        self.paths = paths
        self.genomes = jnp.concatenate([p.export_genomes() for p in paths])
        self.genomes = self.genomes[start_gen:]
        if end_gen is not None:
            self.genomes = self.genomes[:end_gen]
        self.pca = PCA(n_components=pca_dim)

        self.pca.fit(self.genomes)
        print("Explained variance", self.pca.explained_variance_ratio_)

        proj = [p.transform(self.pca) for p in paths]
        proj = jnp.concatenate(proj)
        self.dim = (
            proj[:, 0].min(),
            proj[:, 0].max(),
            proj[:, 1].min(),
            proj[:, 1].max(),
        )

        self.samples = None
        self.genomes = None
        self.true_fit = None

    def __repr__(self):
        return f"PCA of {' - '.join([p.name for p in self.paths])}"

    def __str__(self):
        return self.__repr__()

    def sample_grid(self, n_points=10, dx=0.1):
        # Interpolate as grid
        x_range = jnp.abs(self.dim[1] - self.dim[0])
        x_margin = x_range * dx
        y_range = jnp.abs(self.dim[3] - self.dim[2])
        y_margin = y_range * dx
        x, y = jnp.meshgrid(
            jnp.linspace(self.dim[0] - x_margin, self.dim[1] + x_margin, n_points),
            jnp.linspace(self.dim[2] - y_margin, self.dim[3] + y_margin, n_points),
        )

        # Flatten
        x = x.reshape(-1)
        y = y.reshape(-1)

        # Stack
        self.samples = jnp.stack([x, y], axis=-1)
        self.genomes = self.pca.inverse_transform(self.samples)
        return self.genomes

    def fitness_grid(self, unflatten_fn, scoring_fn, key):
        if self.samples is None:
            raise ValueError("No samples computed")
        # print(self.genomes.shape)
        nets = unflatten_fn(self.genomes)
        # key = jax.random.PRNGKey(0)
        self.true_fit, descriptors, extra_scores, random_key = scoring_fn(nets, key)
        # Compute fitness for each path
        for p in self.paths:
            p.evaluate(unflatten_fn, scoring_fn, key)
        return self.true_fit

    def plot(self, save=None, max_gen=None):
        fig, axes = plt.subplots(1, 2, figsize=(20, 10))
        if self.samples is not None:
            if self.true_fit is None:
                raise ValueError("No fitness computed")
            n_points = int(jnp.sqrt(self.true_fit.shape[0]))
            x = self.samples[:, 0]
            x_grid = x.reshape((n_points, n_points))
            y = self.samples[:, 1]
            y_grid = y.reshape((n_points, n_points))
            z_grid = self.true_fit.reshape((n_points, n_points))
            contour = axes[0].contourf(
                x_grid, y_grid, z_grid, 20, cmap="viridis", alpha=0.5
            )
            fig.colorbar(contour, location="right", use_gridspec=True)
        colors = ["r", "g", "y", "m", "c", "k"]
        colors = ["orange", "grey"]
        for p, c in zip(self.paths, colors):
            p.plot(axes[0], c, max_gen=max_gen)
            gens = [g for g in p.gens if g <= max_gen]
            fit = p.fitnesses[: len(gens)]
            axes[1].plot(gens, fit, label=p.name, c=c)[0]
            # x_axis max
            axes[1].set_xlim(0, max(p.gens))
            # y_axis max
            axes[1].set_ylim(0, 5500)
        # Start point
        axes[0].scatter(p.projected[0, 0], p.projected[0, 1], c="r", label="Start")
        axes[0].legend()
        axes[0].set_xlabel(f"PC1 - {self.pca.explained_variance_ratio_[0]:.2f}")
        axes[0].set_ylabel(f"PC2 - {self.pca.explained_variance_ratio_[1]:.2f}")

        axes[1].set_xlabel("Generation")
        axes[1].set_ylabel("Fitness")
        axes[1].set_title("Fitness")

        if save is not None:
            plt.savefig(save, dpi=300, bbox_inches="tight")

    def plot_3d(self, save=None):
        if self.samples is not None:
            if self.true_fit is None:
                raise ValueError("No fitness computed")
            n_points = int(jnp.sqrt(self.true_fit.shape[0]))
            x = self.samples[:, 0]
            x_grid = x.reshape((n_points, n_points))
            y = self.samples[:, 1]
            y_grid = y.reshape((n_points, n_points))
            z_grid = self.true_fit.reshape((n_points, n_points))

            x_size = x_grid.max() - x_grid.min()
            y_size = y_grid.max() - y_grid.min()
            x_size = float(x_size)
            y_size = float(y_size)

            y_size = y_size / x_size
            x_size = 1.0

            # Create a 3D surface plot
            fig = go.Figure(
                data=[go.Surface(x=x_grid, y=y_grid, z=z_grid, colorscale="Viridis")],
            )

            # Plot paths
            for p in self.paths:
                x = p.projected[:, 0]
                y = p.projected[:, 1]
                z = p.fitnesses
                trace = go.Scatter3d(
                    x=x,
                    y=y,
                    z=z,
                    mode="lines",
                    name=p.name,
                    # line=dict(
                    #     color='red',
                    #     width=4
                    # )
                )
                fig.add_trace(trace)

        # Set the axis labels and title
        fig.update_layout(
            scene=dict(
                xaxis_title="V1",
                yaxis_title="V2",
                zaxis_title="Fitness",
                aspectratio=dict(x=x_size, y=y_size, z=0.7),
                camera_eye=dict(x=1.2, y=1.2, z=0.6),
            )
        )
        # fig.show()
        if save is not None:
            pyo.plot(fig, filename=save, auto_open=False)

    def make_gif(self, save=None):
        fig, (ax1, ax2) = plt.subplots(
            1, 2, figsize=(20, 10), gridspec_kw={"width_ratios": [2, 1]}
        )

        n_points = int(jnp.sqrt(self.true_fit.shape[0]))
        x = self.samples[:, 0]
        x_grid = x.reshape((n_points, n_points))
        y = self.samples[:, 1]
        y_grid = y.reshape((n_points, n_points))
        z_grid = self.true_fit.reshape((n_points, n_points))
        contour = ax1.contourf(x_grid, y_grid, z_grid, 20, cmap="viridis", zorder=1)

        # colorbar
        if self.env_name is not None and self.env_name in FIT_NORM:
            contour.set_clim(*FIT_NORM[self.env_name])
        else:
            f_min, f_max = self.true_fit.min(), self.true_fit.max()
            contour.set_clim(f_min, f_max)
        fig.colorbar(contour, location="left", use_gridspec=True)

        ax1.set_title(" + ".join([p.name for p in self.paths]))
        ax1.set_xlabel(f"PC1 - {self.pca.explained_variance_ratio_[0]:.2f}")
        ax1.set_ylabel(f"PC2 - {self.pca.explained_variance_ratio_[1]:.2f}")

        ax2.set_xlabel("Generation")
        ax2.set_ylabel("Fitness")
        ax2.set_title("Fitness")

        plt.suptitle(f"{args.jobid} - {args.config}")

        for p in self.paths:
            p.line = ax1.plot(
                p.projected[:, 0], p.projected[:, 1], label=p.name, zorder=2
            )[0]
            p.fit_plot = ax2.plot(p.gens, p.fitnesses, label=p.name)[0]
        ax1.legend()

        def update(frame):
            # Update the data
            print(f"GIF {frame}", end="\r")
            for p in self.paths:
                p.line.set_data(p.projected[:frame, 0], p.projected[:frame, 1])
                p.fit_plot.set_data(p.gens[:frame], p.fitnesses[:frame])
            # update title with generation
            plt.title(
                " + ".join([p.name for p in self.paths]) + f" (gen {p.gens[frame]})"
            )
            return None

        # Create the animation
        frames = p.projected.shape[0]
        # frames = 200
        ani = FuncAnimation(fig, update, frames=frames, interval=1)

        # Save the animation as a GIF
        if save is not None:
            ani.save(save, writer=PillowWriter(fps=20))

        # show gif
        plt.close()


from sklearn.decomposition import PCA


def project(v, d):
    """Project vector v onto d"""
    return jnp.dot(v, d) / jnp.dot(d, d)


def project_2d(v, d):
    """Project vector v onto d"""
    return [project(v, dd) for dd in d]


def frobenius_norm(A):
    return jnp.sqrt(jnp.sum(A**2))


def pca_ratio(W, X):
    return frobenius_norm(X @ W) ** 2 / frobenius_norm(X) ** 2


class MyPCA:
    def __init__(self, dim=2, all_dim=2):
        self.dim = dim
        self.all_dim = all_dim
        self.pca = PCA(n_components=all_dim)
        self.pc = []
        self.center = None
        self.explained_variance_ratio_ = None

    def fit(self, genomes, center):
        genomes = genomes - center
        self.center = center
        self.pca.fit(genomes)
        self.explained_variance_ratio_ = self.pca.explained_variance_ratio_
        self._pc = jnp.array(
            [jnp.array(self.pca.components_[i]) for i in range(self.all_dim)]
        )
        self.pc = self._pc[: self.dim]
        # print("PC shapes", self.pc.shape, self._pc.shape)

    def transform(self, genomes):
        genomes = genomes - self.center
        return jnp.array([project_2d(g, self.pc) for g in genomes])

    def inverse_transform(self, proj):
        """Reconstruct genomes from projection"""
        return jnp.dot(proj, self.pc) + self.center

    def pca_ratio(self, X):
        return pca_ratio(self.pc.T, X)

    def all_pca_ratios(self, X, dims=None):
        ratios = []
        if dims is None:
            dims = range(1, self.all_dim + 1)
        for i in dims:
            pc = self._pc[:i]
            new_ratio = pca_ratio(pc.T, X)
            ratios.append(new_ratio)
        return jnp.array(ratios)


class StepPathPCA(PathPCA):
    def __init__(self, paths, env_name=None, max_gens=None, dims=None):
        self.paths = paths
        self.env_name = env_name

        if max_gens is not None:
            self.genomes = jnp.concatenate(
                [p.export_genomes()[:max_gens] for p in paths]
            )
        else:
            self.genomes = jnp.concatenate([p.export_genomes() for p in paths])
        self.center = self.paths[0].export_genomes()[:max_gens][-1]

        all_dims = max(dims) if dims is not None else 10
        # print("Dims", dims, "All dims", all_dims)
        self.pca = MyPCA(2, all_dim=all_dims)

        # print("Genomes shape", self.genomes.shape)

        self.pca.fit(self.genomes, self.center)
        # print("Sklearn variance", jnp.sum(self.pca.explained_variance_ratio_))
        # print("PC shape", self.pca.pc.shape)

        self.backward_variance = self.pca.all_pca_ratios(self.genomes, dims=dims)
        # print("Backward variance", self.backward_variance)

        forward_genomes = np.concatenate([p.export_genomes()[max_gens:] for p in paths])
        self.forward_variance = self.pca.all_pca_ratios(forward_genomes, dims=dims)
        # print("Forward variance", self.forward_variance)

        close_genomes = np.concatenate(
            [p.export_genomes()[max_gens : max_gens + 100] for p in paths]
        )
        self.close_variance = self.pca.all_pca_ratios(close_genomes, dims=dims)

        self.genomes = jnp.concatenate([p.export_genomes() for p in paths])
        self.total_variance = self.pca.all_pca_ratios(self.genomes, dims=dims)

        # self.sliding_variance = None
        # if max_gens > all_dims:
        #     sliding_gens = min(0, max_gens-100)
        #     sliding_genomes = np.concatenate(
        #         [p.export_genomes()[sliding_gens : max_gens] for p in paths]
        #     )
        #     sliding_pca = MyPCA(2, all_dim=all_dims)
        #     sliding_pca.fit(sliding_genomes, self.center)
        #     self.sliding_variance = sliding_pca.all_pca_ratios(close_genomes, dims=dims)

        proj = [p.transform(self.pca) for p in paths]
        proj = jnp.concatenate(proj)
        self.dim = (
            proj[:, 0].min(),
            proj[:, 0].max(),
            proj[:, 1].min(),
            proj[:, 1].max(),
        )

        self.samples = None
        self.genomes = None
        self.true_fit = None

    def __repr__(self):
        return f"StepPCA of {' - '.join([p.name for p in self.paths])} - ref {self.reference}"

    def sample_grid(self, n_points=10, dx=0.1):
        # Interpolate as grid
        x_range = jnp.abs(self.dim[1] - self.dim[0])
        x_margin = x_range * dx
        y_range = jnp.abs(self.dim[3] - self.dim[2])
        y_margin = y_range * dx
        x, y = jnp.meshgrid(
            jnp.linspace(self.dim[0] - x_margin, self.dim[1] + x_margin, n_points),
            jnp.linspace(self.dim[2] - y_margin, self.dim[3] + y_margin, n_points),
        )

        # Flatten
        x = x.reshape(-1)
        y = y.reshape(-1)

        # Stack
        self.samples = jnp.stack([x, y], axis=-1)
        self.genomes = self.pca.inverse_transform(self.samples)
        return self.genomes


if __name__ == "__main__":
    import matplotlib as mpl

    font_size = 20
    mpl_params = {
        "axes.labelsize": font_size,
        "axes.titlesize": font_size,
        "legend.fontsize": font_size,
        "xtick.labelsize": font_size,
        "ytick.labelsize": font_size,
        "font.size": font_size,
        "text.usetex": False,
        "axes.titlepad": 10,
    }
    mpl.rcParams.update(mpl_params)


    # parse first cli argument
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "save_path", type=str, help="Path to the folder containing the config.json"
    )
    # parser.add_argument("--max_gens", type=int, default=1000)
    # make max_gens a list
    parser.add_argument("--max_gens", nargs="+", type=int, default=[1000])
    # arg dx: float = 0.5
    parser.add_argument("--dx", type=float, default=0.5, help="dx")
    plot_args = parser.parse_args()
    save_path = plot_args.save_path
    print(plot_args)

    if 0 in plot_args.max_gens:
        raise ValueError("max_gens should be > 0")

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

    default = {
        "surrogate_batch": 1024,
        "surrogate": False,
        # "deterministic": True,
        # "es": "cmaes"
    }
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

    def scores(fitnesses, descriptors) -> jnp.ndarray:
        return fitnesses

    # Check if there is a RL part
    try:
        emitter.es_emitter
        rl = True
        print("ES + RL")
    except AttributeError:
        rl = False
        print("Only ES")

    n_points = 100
    path_names = {
        "ES": save_path + "/gen_*_offspring.npy",
        "Actor": save_path + "/gen_*_actor.npy",
    }

    paths = [["ES"]]

    if rl:
        paths = [
            # ["Actor"],
            ["ES", "Actor"],
        ]
        unflatten_fn = jax.vmap(emitter.es_emitter.unflatten)
    else:
        unflatten_fn = jax.vmap(emitter.unflatten)

    for path in paths:
        print(path)
        pca = StepPathPCA(
            [Path(path_names[p], name=p) for p in path],
            env_name=args.env_name,
            max_gens=1000,
            # max_gens=plot_args.max_gens,
        )

        print("_".join([p.name for p in pca.paths]))
        pca.sample_grid(n_points, dx=plot_args.dx)
        # Evaluate

        scoring_fn = EM.scoring_fn
        pca.fitness_grid(unflatten_fn, scoring_fn, random_key)

        # 3D
        # save = save_path + f"/3dpath_{'_'.join([p.name for p in pca.paths])}.html"
        # pca.plot_3d(save=save)
        for max_gens in plot_args.max_gens:
            print(f"Plotting {max_gens}")
            # 2D
            save = (
                save_path
                + f"/2dpath_{'_'.join([p.name for p in pca.paths])}_{max_gens}.png"
            )
            pca.plot(save=save, max_gen=max_gens)

        # GIF
        # save = save_path + f"/animated_path_{'_'.join([p.name for p in pca.paths])}.gif"
        # pca.make_gif(save=save)
