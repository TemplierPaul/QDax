import os

import functools
import time

import jax
import jax.numpy as jnp

import brax
import qdax


from qdax.core.map_elites import MAPElites
from qdax.core.containers.mapelites_repertoire import compute_cvt_centroids
from qdax import environments
from qdax.tasks.brax_envs import scoring_function_brax_envs as scoring_function
from qdax.core.neuroevolution.buffers.buffer import QDTransition
from qdax.core.neuroevolution.networks.networks import MLP
from qdax.core.emitters.mutation_operators import isoline_variation
from qdax.utils.plotting import plot_map_elites_results, plot_map_elites_age

from qdax.core.emitters.pga_me_emitter import PGAMEConfig, PGAMEEmitter
from qdax.utils.metrics import CSVLogger, default_qd_metrics

from qdax.core.emitters.inj_me_emitter import InjectionMEEmitter
from qdax.core.containers.age_repertoire import AgeMapElitesRepertoire, Age, age_qd_metrics
from qdax.core.age_mapelites import AgeMAPElites

from dataclasses import dataclass

@dataclass
class PGAMaker:
    es = None
    env = None
    emitter = None
    emitter_state = None
    repertoire = None
    random_key = None
    wandb_run = None
    policy_network = None
    rollout_fn = None
    eval_fn = None

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)
