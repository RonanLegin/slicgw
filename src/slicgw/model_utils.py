import numpy as np

import jax
import jax.numpy as jnp
from jax import grad, vmap, jacfwd, jit, jacrev, vjp
import orbax

from scoregen.models import State, init_model, get_config, get_score_fn
from scoregen.sampling import get_sampler as get_sampler_noise
from scoregen import VESDE

import os


def load_score_model(config_path, checkpoint_path, rng=None):
    
    # Get SLIC model config
    config = get_config(config_path)
    
    if rng is None:
        rng = jax.random.PRNGKey(config.seed)
        
    rng, step_rng = jax.random.split(rng)
    score_model, init_model_state, initial_params = init_model(step_rng, config)
    
    # Initialize State
    state = State(step=0, opt_state=None,
                       lr=None,
                       model_state=init_model_state,
                       params=initial_params,
                       ema_rate=config.model.ema_rate,
                       params_ema=initial_params,
                       rng=rng) 

    # Load checkpoint
    if checkpoint_path is not None:
        ckptr = orbax.checkpoint.Checkpointer(orbax.checkpoint.PyTreeCheckpointHandler()) 
        state = ckptr.restore(checkpoint_path, item=state)

    # Setup SDE
    sde = VESDE(sigma_min=config.model.sigma_min, sigma_max=config.model.sigma_max, N=config.model.num_scales)
    
    return score_model, state, sde, rng, config



def get_latest_checkpoint(checkpoint_dir):
    # Get list of all checkpoints
    checkpoints = os.listdir(checkpoint_dir)

    # Remove non-integer named checkpoints if any
    checkpoints = [ckpt for ckpt in checkpoints if ckpt.isdigit()]

    # If there are no valid checkpoints, return None
    if not checkpoints:
        return None

    # Convert to integer and find the maximum (latest)
    checkpoints = [int(ckpt) for ckpt in checkpoints]
    latest_checkpoint = max(checkpoints)

    return latest_checkpoint
