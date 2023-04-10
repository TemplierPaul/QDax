from dataclasses import dataclass
from functools import partial
from typing import Any, Optional, Tuple
from typing import Callable, Tuple


import flax.linen as nn
import jax
import optax
from jax import numpy as jnp
from jax.tree_util import tree_flatten, tree_unflatten, tree_map

from qdax.core.containers.repertoire import Repertoire
from qdax.core.emitters.emitter import Emitter, EmitterState
from qdax.core.neuroevolution.buffers.buffer import QDTransition, ReplayBuffer
from qdax.core.neuroevolution.losses.td3_loss import make_td3_loss_fn
from qdax.core.neuroevolution.networks.networks import QModule
from qdax.environments.base_wrappers import QDEnv
from qdax.types import Descriptor, ExtraScores, Fitness, Genotype, Params, RNGKey
from qdax.core.emitters.vanilla_es_emitter import VanillaESConfig, VanillaESEmitterState, VanillaESEmitter, NoveltyArchive
from qdax.core.emitters.qpg_emitter import QualityPGConfig, QualityPGEmitterState, QualityPGEmitter
from qdax.core.rl_es_parts.es_utils import ESRepertoire, ESMetrics
from qdax.core.cmaes import CMAESState
from qdax.core.emitters.esrl_emitter import ESRLConfig, ESRLEmitterState, ESRLEmitter
 
def flatten(network):
    flat_variables, _ = tree_flatten(network)
    vect = jnp.concatenate([jnp.ravel(x) for x in flat_variables])
    return vect

class TestGradientsEmitter(ESRLEmitter):
    partial(
        jax.jit,
        static_argnames=("self",),
    )
    def state_update(
        self,
        emitter_state: ESRLEmitterState,
        repertoire: ESRepertoire,
        genotypes: Genotype,
        fitnesses: Fitness,
        descriptors: Descriptor,
        extra_scores: ExtraScores,
    ) -> ESRLEmitterState:
        """Generate the gradient offspring for the next emitter call. Also
        update the novelty archive and generation count from current call.

        Chooses between ES and RL update.

        Params:
            emitter_state
            repertoire: unused
            genotypes: the genotypes of the offspring
            fitnesses: the fitnesses of the offspring
            descriptors: the descriptors of the offspring
            extra_scores: the extra scores of the offspring

        Returns:
            the updated emitter state
        """

        key, emitter_state = emitter_state.get_key()
        
        # Store es center
        old_offspring = emitter_state.es_state.offspring
        # To vector
        old_center = emitter_state.es_state.offspring
        old_center = flatten(old_center)

        # Do ES update
        emitter_state = self.es_state_update(
            emitter_state,
            repertoire,
            genotypes,
            fitnesses,
            descriptors,
            extra_scores
        )

        # Compute ES step
        # es_step = emitter_state.es_state.offspring - old_center
        new_center = emitter_state.es_state.offspring
        new_center = flatten(new_center)

        es_step = new_center - old_center

        key, emitter_state = emitter_state.get_key()
        
        # Update random key
        rl_state = emitter_state.rl_state.replace(
            random_key=key,
        )

        # Compute RL update on old center
        rl_center = self.rl_emitter.emit_pg(
            emitter_state = rl_state,
            parents=old_offspring,
        )
        rl_center = flatten(rl_center)

        # Compute RL step
        rl_step = rl_center - old_center

        # Compute step norms
        es_step_norm = jnp.linalg.norm(es_step)
        rl_step_norm = jnp.linalg.norm(rl_step)
        # Compute cosine similarity
        cos_sim = jnp.dot(es_step, rl_step) / (es_step_norm * rl_step_norm)

        # Compute the % of dimensions where the sign of the step is the same
        es_sign = jnp.sign(es_step)
        rl_sign = jnp.sign(rl_step)
        same_sign = jnp.sum(es_sign == rl_sign) / len(es_sign)

        # Log
        actor_genome = emitter_state.rl_state.actor_params
        actor_fitness, _ = self.multi_eval(actor_genome, key)

        actor_genome = flatten(actor_genome)
        # Compute center - actor distance
        actor_dist = jnp.linalg.norm(actor_genome - new_center)

        offspring = emitter_state.es_state.offspring

        metrics = self.get_metrics(
            emitter_state,
            offspring,
            extra_scores,
            fitnesses,
            evaluations=emitter_state.metrics.evaluations + self.es_emitter._config.sample_number,
            random_key=key,
        )

        metrics = metrics.replace(
            actor_fitness=actor_fitness,
            es_step_norm = es_step_norm,
            rl_step_norm = rl_step_norm,
            cosine_similarity = cos_sim,
            actor_center_dist = actor_dist,
            shared_directions = same_sign,
        )

        emitter_state = emitter_state.set_metrics(metrics)

        return emitter_state