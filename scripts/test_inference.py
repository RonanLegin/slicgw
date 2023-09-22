import numpy as np
import os
import argparse

import jax
import jax.numpy as jnp
from jax import grad, vmap, jacfwd, jit, jacrev, vjp

jax.config.update("jax_enable_x64", False)

from scoregen.models import State, init_model, get_config, get_score_fn, get_model_fn
from scoregen import VESDE

from slicgw.likelihood.slic_likelihood import get_score_likelihood
from slicgw.likelihood.simulation import model_td_whitened as gw_model
from slicgw.sampling.mala import mala_step as step_fn
from slicgw.sampling.ula import ula_step as step_burn_fn
from slicgw.sampling.utils import get_sample_fn
from slicgw.utils import read_data, get_latest_checkpoint, load_score_model, get_sampler_noise, get_score_fn, plot_results, compute_precond_matrix
from slicgw.constants import *

from ripple import ms_to_Mc_eta
from ripple.waveforms import IMRPhenomD

import pandas as pd
import h5py
import scipy.signal as sl
from scipy.signal.windows import tukey

import matplotlib.pyplot as plt
import psutil
import gc
from tqdm import tqdm



process = psutil.Process(os.getpid())

# Load average power spectral density of training data to use to whiten forward simulations
psd_avg = np.load('/mnt/home/rlegin/ceph/gw/psd_avg.npy')
psd_freq = np.load('/mnt/home/rlegin/ceph/gw/psd_freq.npy')
psd_avg_sqrt = np.sqrt(psd_avg)
psd_avg_sqrt = jnp.asarray(psd_avg_sqrt, dtype=jax_dtype)
psd_freq = jnp.asarray(psd_freq, dtype=jax_dtype)
    
# Define sampling parameters
num_sims = 500
num_steps = 400
step_size = 0.1
batch_size = 16

# Define fixed simulation parameters
theta_labels = ["dist_mpc"]
inclination = np.pi
m1_msun = 35
m2_msun = 32
chi1 = 0.
chi2 = 0.
tc = 2.0
phic = 0.0
Mc, eta = ms_to_Mc_eta(np.array([m1_msun, m2_msun]))

sim_args = {
    'f': jnp.array(f, dtype=jax_dtype),
    'f_ref': f_ref,
    'strain_scale': strain_scale,
    'Fp': Fp,
    'Fc': Fc,
    'psd_sqrt': psd_avg_sqrt
}

# Define wrapper for forward model. This is necessary if we want to keep certain simulation parameters fixed
def wrap_forward_model(theta_input, args, gw_model):
    theta_fixed = theta_input[jnp.array([0,1,2,3,5,6,7])]
    # The forward model combines the fixed thetas and the variable thetas at the correct index position (here, 4)
    def forward_model(theta):
        theta_combined = jnp.insert(theta_fixed, 4, theta)
        model = gw_model(theta_combined, args)
        return jnp.expand_dims(model, axis=-1)

    return forward_model

# Make a wrapper for the score of the likelihood function which accepts a realization of simulated data
def wrap_score_likelihood_fn(data, score_likelihood_fn):
    def score_fn(theta):
        score = score_likelihood_fn(theta, jnp.tile(data, (theta.shape[0],1,1)))
        return score
    return score_fn
        
      
    
def main(config_path, checkpoint_path, dir_name):
    
    config = get_config(config_path) # Get slic model config 
    
    rng = jax.random.PRNGKey(config.seed)
    os.makedirs(dir_name, exist_ok=True) # Create directory to save samples and ground truths 
    
    # Load latest slic checkpoint and get sampling function and slic score function
    latest_checkpoint = get_latest_checkpoint(checkpoint_path)
    model, state, sde, rng, config = load_score_model(config_path, os.path.join(checkpoint_path, str(latest_checkpoint), 'default'))
    sampling_noise_fn = get_sampler_noise(sde, model, shape=(1, config.data.data_size, 1), eps=1e-5)
    score_noise_fn = get_score_fn(sde, model, state.params_ema, state.model_state, train=False, return_state=False)
    
    sampling_noise_fn_jit = jax.jit(sampling_noise_fn)
    gw_model_jit = jax.jit(gw_model)
    score_noise_fn_jit = jax.jit(score_noise_fn)
    
    base_memory_usage = process.memory_info().rss

    thetas = []
    chains = []
    
    for i in range(num_sims):
        
        memory_usage = process.memory_info().rss
        loop_memory_usage = memory_usage - base_memory_usage
        print("memory usage",loop_memory_usage)
        
        
        """ Sample noise """
        rng, step_rng = jax.random.split(rng)
        noise_sample = sampling_noise_fn_jit(rng, state)
        noise_sample = jnp.asarray(noise_sample, dtype=jax_dtype)
        
        
        """ Sample ground truth parameters """
        rng, step_rng = jax.random.split(rng)
        dist_mpc = jax.random.uniform(rng, dtype=jax_dtype, minval=300., maxval=600.)
        thetas.append(np.array([dist_mpc]))
        
        
        """ Simulate data """
        theta_true = jnp.array([Mc, eta, chi1, chi2, dist_mpc, tc, phic, inclination], dtype=jax_dtype)
        signal = gw_model_jit(theta_true, sim_args)
        data = signal + noise_sample.reshape(-1,)
        data = jnp.expand_dims(data, axis=(0,2)) # Add batch and channel dimension to make it compatible with slic network

        
        """ Get batchable score function """
        forward_fn = wrap_forward_model(theta_true, sim_args, gw_model_jit)
        jacobian_fn = jacfwd(forward_fn, argnums=0)
        score_likelihood_fn = get_score_likelihood(vmap(forward_fn,in_axes=(0,)),
                                                   vmap(jacobian_fn,in_axes=(0,)),
                                                   score_noise_fn_jit,
                                                   time_slic=0.33333 # Temperature time 't' to evaluate slic model
                                                  )
        score_fn = wrap_score_likelihood_fn(data, score_likelihood_fn)

        
        """ Get mass matrix """
        hessian_fn = jax.jacfwd(score_fn)
        hessian = hessian_fn(theta_true[None,4:5]).reshape(1,1)
        mass_matrix_cov = jnp.linalg.inv(hessian)
        mass_matrix = jnp.linalg.cholesky(jnp.abs(mass_matrix_cov))
        
        
        """ Get initial walkers """
        theta_initial = jnp.array([dist_mpc], dtype=jax_dtype).reshape(1,1)
        theta_initial = jnp.tile(theta_initial, (batch_size,1))
        #rng, step_rng = jax.random.split(rng)
        #theta_initial = jax.random.uniform(rng, shape=(batch_size,1), dtype=jax_dtype, minval=300., maxval=600.)

        
        print('Sampling {}.'.format(i + 1))
        rng, step_rng = jax.random.split(rng)
        sample_fn = get_sample_fn(step_fn, step_burn_fn, score_fn, mass_matrix)
        chain, chain_score = sample_fn(rng, theta_initial, step_size, num_steps, burn_in=0)
        chains.append(np.array(chain))
        np.save(dir_name + 'chains.npy',np.array(chains))
        np.save(dir_name + 'thetas.npy',np.array(thetas))
        #plot_results(chain, chain_score, theta_true[4:5], theta_labels, dir_name, tc_trigger)
         

        
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Your script description")
    parser.add_argument("--config_path", required=True, help="Path to the configuration file.")
    parser.add_argument("--checkpoint_path", required=True, help="Path to the checkpoint file.")
    parser.add_argument("--dir_name", default=".", help="Directory to use. Defaults to the current directory.")

    main_args = parser.parse_args()
    main(main_args.config_path, main_args.checkpoint_path, main_args.dir_name)

