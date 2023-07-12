import numpy as np
import os

import jax
import jax.numpy as jnp
from jax import grad, vmap, jacfwd, jit, jacrev, vjp

jax.config.update("jax_enable_x64", False)

from scoregen.models import State, init_model, get_config, get_score_fn, get_model_fn
from scoregen import VESDE

from slicgw.likelihood.slic_likelihood import get_score_likelihood
from slicgw.likelihood.simulation import model1 as gw_model
from slicgw.sampling.mala import mala_step as step_fn
from slicgw.sampling.ula import ula_step as step_burn_fn
from slicgw.sampling.utils import get_sample_fn
from slicgw.utils import read_data, load_score_model, get_sampler_noise, get_score_fn, plot_results, compute_precond_matrix
from slicgw.constants import *

from ripple import ms_to_Mc_eta
from ripple.waveforms import IMRPhenomD

import pandas as pd
import h5py
import scipy.signal as sl
from scipy.signal.windows import tukey

config = get_config('../data/config.json')

dir_name = 'test1/'
os.makedirs(dir_name, exist_ok=True)

N_pp_samples = 100

score_model, state, sde, rng, config = load_score_model(config_path='../data/config.json', checkpoint_path='../data/checkpoints/787/default/')

sampling_noise_fn = get_sampler_noise(sde, score_model, shape=(1, config.data.data_size, 2), eps=1e-5)

score_noise_fn = get_score_fn(sde, score_model, state.params_ema, state.model_state, train=False, return_state=False)

rng = jax.random.PRNGKey(config.seed)

theta_labels = ["Mc","eta"]

theta_pp = []
samples_pp = []
for i in range(N_pp_samples):
    rng, step_rng = jax.random.split(rng)
    noise_sample = sampling_noise_fn(rng, state)
    noise_sample = jnp.asarray(noise_sample, dtype=jax_dtype)

    rng, step_rng = jax.random.split(rng)
    m1 = jax.random.uniform(rng, dtype=jax_dtype, minval=25., maxval=35.)
    rng, step_rng = jax.random.split(rng)
    m2 = jax.random.uniform(rng, dtype=jax_dtype, minval=25., maxval=35.)

    Mc, eta = ms_to_Mc_eta(jnp.array([m1, m2], dtype=jax_dtype))
    theta_pp.append(np.array([Mc, eta]))
    np.save('theta_pp.npy',np.array(theta_pp))
    
    """ Create simulated data """

    inclination = np.pi
    chi1 = 0.
    chi2 = 0.
    tc = 2.0
    phic = 0.0
    dist_mpc = 400

    args = {
        'f': jnp.array(f[81:593], dtype=jax_dtype),
        'f_ref': f_ref,
        'strain_scale': strain_scale,
        'Fp': Fp,
        'Fc': Fc
    }

    
    theta_true = jnp.array([Mc, eta, chi1, chi2, dist_mpc, tc, phic, inclination], dtype=jax_dtype)
    
    signal = gw_model(theta_true, args)[None]
    data = signal + noise_sample
    

    """ Sample """
    
    num_steps = 500
    step_size = 0.05
    batch_size = 32
    
    
    
    def wrap_forward_model(theta_input, args, gw_model):
        theta_fixed = theta_input[2:]  # Fixed part of the theta_input
    
        def forward_model(theta):
            theta_combined = jnp.concatenate((theta, theta_fixed))
            return gw_model(theta_combined, args)#[81:593]

        return forward_model
    


    # Use vmap to create a batched version of forward_model
    forward_fn = wrap_forward_model(theta_true, args, gw_model)
    jacobian_fn = jacfwd(forward_fn, argnums=0)
    
    theta_initial = jnp.array([Mc, eta], dtype=jax_dtype).reshape(1,2)
    theta_initial = jnp.tile(theta_initial, (batch_size,1))

    score_likelihood_fn = get_score_likelihood(vmap(forward_fn,in_axes=(0,)), 
                                               vmap(jacobian_fn,in_axes=(0,)), 
                                               score_noise_fn
                                              )
    
    def wrap_score_fn(data):
        def score_fn(theta):
            score = score_likelihood_fn(theta, jnp.tile(data, (theta.shape[0],1,1)))
            return score
        return score_fn

    score_fn = wrap_score_fn(data)
    hessian_fn = jax.jacfwd(score_fn) 
    hessian = hessian_fn(theta_true[None,:2]).reshape(2,2)
    mass_matrix_cov = jnp.linalg.inv(hessian)

    mass_matrix = jnp.linalg.cholesky(jnp.abs(mass_matrix_cov))
    
    step_burn_fn_jit = jax.jit(step_burn_fn, static_argnums=(2,3))
    step_fn_jit = jax.jit(step_fn, static_argnums=(2,3))
    sample_fn = get_sample_fn(step_fn_jit, step_burn_fn_jit, score_fn, mass_matrix)

    # Define starting walkers
    rng, step_rng = jax.random.split(rng)
    m1 = jax.random.uniform(rng, shape=(batch_size,1), dtype=jax_dtype, minval=25., maxval=35.)
    rng, step_rng = jax.random.split(rng)
    m2 = jax.random.uniform(rng, shape=(batch_size,1), dtype=jax_dtype, minval=25., maxval=35.)
    
    Mc, eta = vmap(ms_to_Mc_eta)(jnp.concatenate([m1, m2], axis=1, dtype=jax_dtype))

    rng, step_rng = jax.random.split(rng)

    print('Sampling {}.'.format(i + 1))
    chain, chain_score = sample_fn(rng, theta_initial, step_size, num_steps, burn_in=0)
    
    samples_pp.append(np.array(chain))
    np.save('samples_pp.npy',np.array(samples_pp))
    #plot_results(chain, chain_score, theta_true[:2], theta_labels, dir_name, i)
    