import argparse
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from qdax.core.rl_es_parts.es_setup import setup_es
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

def multi_eval(nets, scoring_fn, n_eval):
    fitnesses = []
    key = jax.random.PRNGKey(0)
    from tqdm import tqdm
    for g in tqdm(range(n_eval)):
        f, _, _, _ = scoring_fn(nets, key)
        fitnesses.append(f)
    return jnp.array(fitnesses).mean(axis=0)

class Path:
    def __init__(self, file_regex, name=None):
        self.file_regex = file_regex
        self.name = name
        self.files = glob.glob(file_regex)
        # Get last number in each file name
        self.genomes = {}
        self.gens = []
        for f in self.files:
            gen = int(re.findall(r'\d+', f.split("/")[-1])[-1])
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

    def plot(self, ax, c):
        if self.projected is None:
            raise ValueError("No projection computed")
        ax.plot(self.projected[:, 0], self.projected[:, 1], label=self.name, c=c)
        ax.scatter(self.projected[-1, 0], self.projected[-1, 1], marker="x", c=c)

    def evaluate(self, unflatten_fn, scoring_fn):
        genomes = self.export_genomes()
        nets = unflatten_fn(genomes)
        key = jax.random.PRNGKey(0)
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
        self.dim = (proj[:, 0].min(), proj[:, 0].max(), proj[:, 1].min(), proj[:, 1].max())

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

    def fitness_grid(self, unflatten_fn, scoring_fn):
        if self.samples is None:
            raise ValueError("No samples computed")
        # print(self.genomes.shape)
        nets = unflatten_fn(self.genomes)
        key = jax.random.PRNGKey(0)
        self.true_fit, descriptors, extra_scores, random_key = scoring_fn(nets, key)
        # Compute fitness for each path
        for p in self.paths:
            p.evaluate(unflatten_fn, scoring_fn)
        return self.true_fit

    def plot(self, save=None):
        plt.figure(figsize=(20, 10))
        if self.samples is not None:
            if self.true_fit is None:
                raise ValueError("No fitness computed")
            n_points = int(jnp.sqrt(self.true_fit.shape[0]))
            x = self.samples[:, 0]
            x_grid = x.reshape((n_points, n_points))
            y = self.samples[:, 1]
            y_grid = y.reshape((n_points, n_points))
            z_grid = self.true_fit.reshape((n_points, n_points))
            plt.contourf(x_grid, y_grid, z_grid, 20, cmap="viridis")
            plt.colorbar()
        colors = ["r", "g", "y", "m", "c", "k"]
        for p, c in zip(self.paths, colors):
            p.plot(plt.gca(), c)
        plt.legend()
        plt.xlabel(f"PC1 - {self.pca.explained_variance_ratio_[0]:.2f}")
        plt.ylabel(f"PC2 - {self.pca.explained_variance_ratio_[1]:.2f}")
        if save is not None:
            plt.savefig(save)

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
            fig = go.Figure(data=[go.Surface(
                x=x_grid,
                y=y_grid,
                z=z_grid,
                colorscale='Viridis'
            )],
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
                    mode='lines',
                    name=p.name,
                    # line=dict(
                    #     color='red',
                    #     width=4
                    # )
                )
                fig.add_trace(trace)

        
        # Set the axis labels and title
        fig.update_layout(scene=dict(
            xaxis_title='V1',
            yaxis_title='V2',
            zaxis_title='Fitness',
            aspectratio=dict(x=x_size, y=y_size, z=0.7),
            camera_eye=dict(x=1.2, y=1.2, z=0.6)
        ))
        # fig.show()
        if save is not None:
            pyo.plot(fig, filename=save, auto_open=False)

    def make_gif(self, save=None):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10), gridspec_kw={'width_ratios': [2, 1]})

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
            p.line = ax1.plot(p.projected[:, 0], p.projected[:, 1], label=p.name, zorder = 2)[0]
            p.fit_plot = ax2.plot(p.gens, p.fitnesses, label=p.name)[0]
        ax1.legend()

        def update(frame):
            # Update the data
            print(f"GIF {frame}", end="\r")
            for p in self.paths:
                p.line.set_data(p.projected[:frame, 0], p.projected[:frame, 1])
                p.fit_plot.set_data(p.gens[:frame], p.fitnesses[:frame])
            # update title with generation
            plt.title(" + ".join([p.name for p in self.paths]) + f" (gen {p.gens[frame]})")
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

class MyPCA:
    def __init__(self, dim=2):
        self.dim = dim
        self.pca = PCA(n_components=dim)
        self.pc = []
        self.center = None
        self.explained_variance_ratio_ = None

    def fit(self, genomes, center):
        genomes = genomes - center
        self.center = center
        self.pca.fit(genomes)
        self.explained_variance_ratio_ = self.pca.explained_variance_ratio_
        self.pc = jnp.array([jnp.array(self.pca.components_[i]) for i in range(self.dim)])

    def transform(self, genomes):
        genomes = genomes - self.center
        return jnp.array([project_2d(g, self.pc) for g in genomes])
    
    def inverse_transform(self, proj):
        """Reconstruct genomes from projection"""
        return jnp.dot(proj, self.pc) + self.center

class StepPathPCA(PathPCA):
    def __init__(self, paths, env_name=None):
        self.paths = paths
        self.env_name = env_name

        self.center = self.paths[0].export_genomes()[-1]
        
        self.genomes = jnp.concatenate([p.export_genomes() for p in paths])
        
        self.pca = MyPCA(2)

        self.pca.fit(self.genomes, self.center)
        print("Explained variance", self.pca.explained_variance_ratio_)

        self.genomes = jnp.concatenate([p.export_genomes() for p in paths])

        proj = [p.transform(self.pca) for p in paths]
        proj = jnp.concatenate(proj)
        self.dim = (proj[:, 0].min(), proj[:, 0].max(), proj[:, 1].min(), proj[:, 1].max())

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
    # parse first cli argument
    parser = argparse.ArgumentParser()
    parser.add_argument("save_path", type=str, help="Path to the folder containing the config.json")
    # arg dx: float = 0.5
    parser.add_argument("--dx", type=float, default=0.5, help="dx")
    plot_args = parser.parse_args()
    save_path = plot_args.save_path
    print(plot_args)

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

    EM = setup_es(args)

    es = EM.es
    env = EM.env
    policy_network = EM.policy_network
    emitter = EM.emitter
    emitter_state = EM.emitter_state
    repertoire = EM.repertoire
    random_key = EM.random_key
    wandb_run  = EM. wandb_run
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
            ["Actor"],
            ["ES", "Actor"],
        ]
        unflatten_fn = jax.vmap(emitter.es_emitter.unflatten)
    else:
        unflatten_fn = jax.vmap(emitter.unflatten)

    for path in paths:
        print(path)
        pca = StepPathPCA([Path(path_names[p], name=p) for p in path], env_name=args.env_name)

        print('_'.join([p.name for p in pca.paths]))
        pca.sample_grid(n_points, dx=plot_args.dx)
        # Evaluate 
        
        scoring_fn = EM.scoring_fn
        pca.fitness_grid(unflatten_fn, scoring_fn)

        # 3D
        save=save_path + f"/3dpath_{'_'.join([p.name for p in pca.paths])}.html"
        pca.plot_3d(save=save)

        # 2D
        save=save_path + f"/2dpath_{'_'.join([p.name for p in pca.paths])}.png"
        pca.plot(save=save)

        # GIF
        save=save_path + f"/animated_path_{'_'.join([p.name for p in pca.paths])}.gif"
        pca.make_gif(save=save)