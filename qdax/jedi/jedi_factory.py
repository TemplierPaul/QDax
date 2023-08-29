from qdax.utils.algo_factory import MEFactory
from qdax.jedi.gaussian_process import fit_GP, plot_GP
from qdax.jedi.pareto import (
    centroids_pareto,
    archive_pareto,
    crowded,
)
from qdax.jedi.ucb import gp_ucb
from qdax.jedi.evolution_strategies.base_es import BD_ES, Fitness_ES
from qdax.jedi.evolution_strategies.aria_es import ARIA_ES
from qdax.jedi.jedi_loop import aim_for
from time import time
import jax.numpy as jnp
import jax


class JEDi:
    # Selection
    centroids_pareto = centroids_pareto
    archive_pareto = archive_pareto
    crowded = crowded
    gp_ucb = gp_ucb

    # Evolution Strategies
    BD_ES = BD_ES
    Fitness_ES = Fitness_ES
    ARIA_ES = ARIA_ES

    @classmethod
    def weighted_ucb(cls):
        raise NotImplementedError("TODO")

    @classmethod
    def add_centers(cls, metrics):
        """
        Add the ES centers to the archive
        """
        # reshape metrics (-1, 2)*
        fitnesses = metrics["fitness"].reshape(-1)
        descriptors = metrics["descriptor"].reshape(-1, 2)
        genotypes = metrics["genotype"]

        # Concatenate on first 2 dimensions
        genotypes = jax.tree_map(lambda x: x.reshape(-1, *x.shape[2:]), genotypes)

        return genotypes, descriptors, fitnesses

    @classmethod
    def add_population(cls, metrics):
        """
        Add the whole ES population to the archive
        """
        fitnesses = metrics["population_fitness"].reshape(-1)
        descriptors = metrics["population_descriptors"].reshape(-1, 2)
        genotypes = metrics["population_networks"]
        # Concatenate on first 2 dimensions
        genotypes = jax.tree_map(lambda x: x.reshape(-1, *x.shape[3:]), genotypes)

        return genotypes, descriptors, fitnesses


class JEDiStep:
    def __init__(
        self,
        selection_fn,
        es_type,
        addition_fn,
    ) -> None:
        self.selection_fn = selection_fn
        self.es_type = es_type
        self.addition_fn = addition_fn

        self.base_state = None


class JEDiFactory(MEFactory):
    def __init__(self, config):
        super().__init__(config)
        self.me_type = "JEDi"
        self.jedi_steps = []
        self.es = []
        self.jedi_config = None

        self.opt_posterior = None
        self.train_data = None

    def __repr__(self) -> str:
        return f"JEDiFactory({self.me_type})"

    def get_explore_exploit(
        self,
        jedi_config,
    ):
        # self._get_default()
        self.get_mapelites()

        default_jedi_config = {
            "env_name": self.config["env_name"],
            "episode_length": self.config["episode_length"],
            "explore": True,
            "exploit": True,
            "crowding": True,
            "aria": False,
            "sample_number": 10,
            "sigma": 0.1,
            "es_gens": 1000,
            "macro_loops": 10,
            "target_nb": 10,
            "add_population": True,
        }

        # merge dicts, jedi_config takes precedence
        self.jedi_config = {**default_jedi_config, **jedi_config}
        explore = self.jedi_config["explore"]
        exploit = self.jedi_config["exploit"]
        crowding = self.jedi_config["crowding"]

        addition_fn = (
            JEDi.add_population
            if self.jedi_config["add_population"]
            else JEDi.add_centers
        )

        if not explore and not exploit:
            raise ValueError("At least one of explore or exploit must be True")

        steps = []

        def f(x):
            return crowded(x) if crowding else x

        if explore:
            if self.jedi_config["aria"]:
                explore_step = JEDiStep(
                    selection_fn=f(JEDi.centroids_pareto),
                    es_type=JEDi.ARIA_ES,
                    addition_fn=addition_fn,
                )
            else:
                explore_step = JEDiStep(
                    selection_fn=f(JEDi.centroids_pareto),
                    es_type=JEDi.BD_ES,
                    addition_fn=addition_fn,
                )
            steps.append(explore_step)

        if exploit:
            exploit_step = JEDiStep(
                selection_fn=f(JEDi.archive_pareto),
                es_type=JEDi.Fitness_ES,
                addition_fn=addition_fn,
            )
            steps.append(exploit_step)

        self.jedi_steps = steps

    def get_aria(
        self,
        jedi_config,
    ):
        # self._get_default()
        self.get_mapelites()

        default_jedi_config = {
            "env_name": self.config["env_name"],
            "episode_length": self.config["episode_length"],
            "explore": True,
            "exploit": True,
            "crowding": True,
            "aria": False,
            "sample_number": 10,
            "sigma": 0.1,
            "es_gens": 1000,
            "macro_loops": 10,
            "target_nb": 10,
            "add_population": True,
        }

        # merge dicts, jedi_config takes precedence
        self.jedi_config = {**default_jedi_config, **jedi_config}

        crowding = self.jedi_config["crowding"]

        addition_fn = (
            JEDi.add_population
            if self.jedi_config["add_population"]
            else JEDi.add_centers
        )

        def f(x):
            return crowded(x) if crowding else x

        step = JEDiStep(
            selection_fn=f(JEDi.centroids_pareto),
            es_type=JEDi.ARIA_ES,
            addition_fn=addition_fn,
        )

        self.jedi_steps = [step]

    def custom_steps(self, steps):
        self._get_default()

        self.jedi_steps = steps

    def run(self):
        assert self.jedi_steps, "No JEDi steps defined"

        random_key = self.random_key
        bounds = [[self.config["min_bd"], self.config["max_bd"]]] * self.config["nb_bd"]
        qd_offset = self.qd_offset
        target_nb = self.jedi_config["target_nb"]

        # Initialise
        for step in self.jedi_steps:
            es = step.es_type(
                self.env,
                self.jedi_config,
            )
            step.base_state, _ = es.init(self.policy_network)
            self.es.append(es)

        # Logs
        repertoire = self.init_repertoire
        print("Logging initial archive")
        loop = 0
        logs = {
            "steps": [],
            "evaluations": [],
            "frames": [],
            "max_fitness": [],
            "coverage": [],
            "qd_score": [],
        }
        repertoire_empty = repertoire.fitnesses == -jnp.inf
        counts = repertoire.count[~repertoire_empty]
        initial_evals = sum(counts)

        logs["steps"].append(loop)
        evals = (
            loop
            * target_nb
            * self.jedi_config["es_gens"]
            * (self.jedi_config["sample_number"])
            + initial_evals
        )
        logs["evaluations"].append(evals)
        logs["frames"].append(evals * self.config["episode_length"])

        qd_score = jnp.sum(repertoire.fitnesses, where=~repertoire_empty)
        logs["qd_score"].append(
            qd_score + self.reward_offset * jnp.sum(1.0 - repertoire_empty)
        )
        logs["coverage"].append(100 * jnp.mean(1.0 - repertoire_empty))
        logs["max_fitness"].append(jnp.max(repertoire.fitnesses))

        try:
            # Run
            for loop in range(self.jedi_config["macro_loops"]):
                max_fitness = jnp.max(repertoire.fitnesses)
                print(f"Loop {loop}, max fitness: {max_fitness}")
                # Fit GP to repertoire
                print("Fitting GP...")
                opt_posterior, train_data = fit_GP(repertoire)
                self.opt_posterior, self.train_data = opt_posterior, train_data
                # print("Done.")

                # Get corresponding step
                step_index = loop % len(self.jedi_steps)
                es = self.es[step_index]
                base_es_emitter_state = self.jedi_steps[step_index].base_state
                selection_fn = self.jedi_steps[step_index].selection_fn
                print(f"Starting {es.name}")
                target_bd = selection_fn(
                    repertoire,
                    opt_posterior,
                    train_data,
                    bounds,
                    n_points=target_nb,
                    plot=True,
                )

                # vmap on targets
                t0 = time()
                metrics, es_emitter_state, random_key = jax.vmap(
                    lambda target: aim_for(
                        target,
                        repertoire,
                        es,
                        self.jedi_config,
                        base_es_emitter_state,
                        random_key,
                    )
                )(target_bd)

                random_key = random_key[0]

                print(f"ES time: {time() - t0:.2f}s")

                genotypes, descriptors, fitnesses = step.addition_fn(metrics)
                repertoire = repertoire.add(genotypes, descriptors, fitnesses)

                logs["steps"].append(loop + 1)
                evals = (
                    (loop + 1)
                    * target_nb
                    * self.jedi_config["es_gens"]
                    * (self.jedi_config["sample_number"])
                )
                logs["evaluations"].append(evals)
                logs["frames"].append(evals * self.config["episode_length"])

                repertoire_empty = repertoire.fitnesses == -jnp.inf
                qd_score = jnp.sum(repertoire.fitnesses, where=~repertoire_empty)
                logs["qd_score"].append(
                    qd_score + qd_offset * jnp.sum(1.0 - repertoire_empty)
                )
                logs["coverage"].append(100 * jnp.mean(1.0 - repertoire_empty))
                logs["max_fitness"].append(jnp.max(repertoire.fitnesses))

        except KeyboardInterrupt:
            print("Interrupted by user")

        print("Fitting final GP")
        self.opt_posterior, self.train_data = fit_GP(repertoire)
        max_fitness = jnp.max(repertoire.fitnesses)
        print(f"Final loop, max fitness: {max_fitness}")

        self.end_repertoire = repertoire
        return repertoire, logs

    def plot_gp(self):
        return plot_GP(
            self.end_repertoire,
            self.opt_posterior,
            grid_size=30,
            min_bd=self.config["min_bd"],
            max_bd=self.config["max_bd"],
        )
