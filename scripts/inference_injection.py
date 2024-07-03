import numpy as np
import os
import argparse

import jax
import jax.numpy as jnp
from jax import grad, vmap, jacfwd, jit, jacrev, vjp

jax.config.update("jax_enable_x64", True)
from jax.lib import xla_bridge
print('device used:', xla_bridge.get_backend().platform)


from slicgw.simulation import model_fd_4d, model_td_3d, model_td_4d, model_td_5d, model_td_7d, model_dummy, sample_uniform_prior
from slicgw.sampling.mala import mala_step as step_fn
from slicgw.sampling.mala import mala_step as step_burn_fn # Using MALA instead of ULA for warmup phase
from slicgw.sampling.utils import get_sample_fn
from slicgw.utils import read_data, plot_results
from slicgw.constants import *
from slicgw.model_utils import get_latest_checkpoint, load_score_model

from scoregen.models import State, init_model, get_config, get_score_fn
from scoregen.sampling import get_sampler as get_sampler_noise
from scoregen import VESDE

import pandas as pd
import h5py
import scipy.signal as sl
from scipy.signal.windows import tukey

import matplotlib.pyplot as plt
from tqdm import tqdm

import json


# Note: this script assumes that the parameters to be inferred all have uniform priors.
# Note: This uniform prior is enforced as if conditions in the sampling code located in the src folder.


def main(save_dir, args):
       
    t_min = args.t_min
    
    config_path = os.path.join(args.workdir, 'config.json')
    checkpoint_path = os.path.join(args.workdir, 'checkpoints')
    
    # Load latest slic checkpoint and get sampling function and slic score function
    latest_checkpoint = get_latest_checkpoint(checkpoint_path)
    model, state, sde, rng, config = load_score_model(config_path, os.path.join(checkpoint_path, str(latest_checkpoint), 'default'))
    sampling_noise_fn = get_sampler_noise(sde, model, shape=(1, seglen*srate, 1), t_min=1e-5)
    sampling_noise_fn_jit = jax.jit(sampling_noise_fn)
    
    with open(os.path.join(config.data.data_test_directory, 'params_{}.json'.format(args.ifo)), 'r') as file:
        config_data = json.load(file)

    seglen_upfactor = config_data.get('seglen_upfactor')

    if seglen_upfactor is not None:
        print("Segment length upfactor:", seglen_upfactor)
    else:
        raise ValueError("Parameter 'seglen_upfactor' not found in the JSON file.")

    noise_samples = np.load(os.path.join(config.data.data_test_directory, 'noise_{}.npy'.format(args.ifo)))
    fourier_mean_avg = np.load(os.path.join(config.data.data_test_directory, 'fourier_mean_{}.npy'.format(args.ifo)))
    fourier_sigma_avg = np.load(os.path.join(config.data.data_test_directory, 'fourier_sigma_{}.npy'.format(args.ifo)))
    
    if args.score_type == 'gaussian':
        white_noise_std = np.sqrt(np.diag(np.load(os.path.join(config.data.data_test_directory, 'covariance_{}.npy'.format(args.ifo)))))
        white_noise_std = jnp.array(np.sqrt(white_noise_std**2 + sde.marginal_prob(None, t_min)[1]**2))

    tc_center = (2 * f.shape[0] / srate) / 2
    
    Fp, Fc = antenna_patterns[args.ifo]
    
    sim_args['fourier_mean_avg'] = jnp.array(fourier_mean_avg)[f >= f_ref]
    sim_args['fourier_sigma_avg'] =  jnp.array(fourier_sigma_avg)[f >= f_ref]
    sim_args['Fp'] = Fp
    sim_args['Fc'] = Fc
    sim_args['f'] = f
    sim_args['f_filtered'] = f[f >= f_ref]
    sim_args['tc'] = tc_center # Default time position is centered in the middle of the cropped segment. If 'tc' is a parameter to be inferred, then the value in sim_args['tc'] is ignored.
    sim_args['window'] = tukey(seglen_upfactor*seglen*srate, tukey_alpha)
    
    tc_range = 3/8*seglen # 3/8 factor to avoid having tc sampled at the boundary
    if args.sim_type == 'dummy':
        gw_model = model_dummy
        param_bounds = [(-1.0, 1.0),(0.2, 1.2)] # [mu, sigma]
        theta_labels = ['mu', 'sigma']
    elif args.sim_type == 'td_3d':
        gw_model = model_td_3d
        param_bounds = [(20., 40.),(200., 900.),(tc_center - tc_range, tc_center + tc_range)] # [Mc, dist_mpc, tc]
        theta_labels = ['Mc', 'dist_mpc', 'tc']
        
        mass_matrix = jnp.array([[ 3.09e-02,  0.00,  0.00],
                                [ 4.30e-01,  1.59e+01,  0.00],
                                [-5.81e-05, -1.32e-08,  3.56e-05]])
    elif args.sim_type == 'td_5d':
        gw_model = model_td_5d
        param_bounds = [(20., 40.),(0.15, 0.25),(200., 900.),(tc_center - tc_range, tc_center + tc_range),(0., 1.0)] # [Mc, eta, dist_mpc, tc, phic]
        theta_labels = ['Mc', 'eta', 'dist_mpc', 'tc', 'phic']
        
        mass_matrix = jnp.diag([2.24e-01, 2.51e-03, 1.57e+01, 1.42e-03, 4.64e-02])
        
    elif args.sim_type == 'td_7d':
        gw_model = model_td_7d
        param_bounds = [(20., 40.),(0.15, 0.25),(-1.0,1.0),(-1.0,1.0),(200., 900.),(tc_center - tc_range, tc_center + tc_range),(0., 1.0)] # [Mc, eta, chi1, chi2, dist_mpc, tc, phic] 
        theta_labels = ['Mc', 'eta', 'chi1', 'chi2', 'dist_mpc', 'tc', 'phic']
        #mass_matrix = jnp.diag(jnp.array([2.24e-01, 2.51e-03, 0.04, 0.04, 1.57e+01, 1.42e-03, 4.64e-02]))
        mass_matrix = jnp.array([[ 5.09e-01,  0.00e+00,  0.00e+00,  0.00e+00,  0.00e+00,  0.00e+00,  0.00e+00],
                                 [ 2.60e-03,  6.02e-03,  0.00e+00,  0.00e+00,  0.00e+00,  0.00e+00,  0.00e+00],
                                 [ 6.84e-02, -3.63e-02,  1.55e-01,  0.00e+00,  0.00e+00,  0.00e+00,  0.00e+00],
                                 [-6.68e-03,  3.23e-02, -2.54e-01,  3.24e-02,  0.00e+00,  0.00e+00,  0.00e+00],
                                 [ 1.70e+01, -5.01e-01, -2.77e+00, -9.58e-01,  1.99e+01,  0.00e+00,  0.00e+00],
                                 [ 5.06e-04, -5.11e-04,  2.08e-03, -2.63e-04,  2.10e-05,  3.61e-04,  0.00e+00],
                                 [ 1.53e-01, -2.67e-02,  8.02e-02, -6.94e-02, -2.20e-03, -3.72e-02,  3.68e-02]])
    else:
        raise ValueError(f"cannot find simulation with name {args.sim_type}")
    jac_gw_model = jacfwd(gw_model, argnums=0)
    
    
    if args.score_type == 'gaussian':
        
        sim_args['white_noise_std'] = white_noise_std
        
        def score_fn_whitenoise(x, args):
            sigma = args['white_noise_std']
            return -x/sigma**2

        def score_likelihood(theta, data, args):
            x = data - gw_model(theta, args)[seglen_upfactor*seglen*srate//2 - seglen*srate//2:seglen_upfactor*seglen*srate//2 + seglen*srate//2]
            jacobian = jac_gw_model(theta, args)[seglen_upfactor*seglen*srate//2 - seglen*srate//2:seglen_upfactor*seglen*srate//2 + seglen*srate//2]
            score_n = score_fn_whitenoise(x, args)
            score_ll = -jnp.dot(score_n, jacobian)
            return score_ll
        
    elif args.score_type == 'slic':
        
        score_fn_slic = get_score_fn(sde, model, state.params_ema, state.model_state, train=False, return_state=False)
            
        def score_likelihood(theta, data, args):
            x = data - gw_model(theta, args)[seglen_upfactor*seglen*srate//2 - seglen*srate//2:seglen_upfactor*seglen*srate//2 + seglen*srate//2]
            jacobian = jac_gw_model(theta, args)[seglen_upfactor*seglen*srate//2 - seglen*srate//2:seglen_upfactor*seglen*srate//2 + seglen*srate//2]
            
            x = x.astype(jnp.float32)
            score_n = score_fn_slic(x[None, ...,None], t_min*jnp.ones((1,), dtype=jnp.float32))[0,...,0]
            #score_n = score_fn_slic(x[None, ...,None], t_min*jnp.ones((1,), dtype=jnp.float64))[0,...,0]
            score_ll = -jnp.dot(score_n, jacobian)
            return score_ll
    else:
        raise ValueError(f"cannot find score function option with name {args.score_type}")
    score_likelihood_vmap = jax.jit(vmap(score_likelihood, in_axes=(0,0, None)))
    

    for i in args.index_array:
        
        rng = jax.random.PRNGKey(i + 9872)
        rng, sample_rng = jax.random.split(rng)
        
        # Sample true theta
        _, theta_true = sample_uniform_prior(sample_rng, param_bounds)
        ndim = theta_true.shape[0]

        
        
        # Generate noise realization
        rng, step_rng = jax.random.split(rng)
        if args.noise_type == 'white':
            noise = 1.0 * jax.random.normal(step_rng, shape=(seglen*srate,))
        elif args.noise_type == 'slic':
            noise = sampling_noise_fn_jit(step_rng, state).reshape(-1, seglen*srate)[0]
        elif args.noise_type == 'real':
            sample_index = jax.random.randint(step_rng, shape=(), minval=0, maxval=noise_samples.shape[0])
            noise = noise_samples[sample_index]
        else:
            raise ValueError(f"cannot find noise type with name {args.noise_type}")
        
        
        # Simulate signal
        signal = gw_model(theta_true, sim_args)[seglen_upfactor*seglen*srate//2 - seglen*srate//2:seglen_upfactor*seglen*srate//2 + seglen*srate//2]
        
        if args.add_diffusion_noise:
            rng, step_rng = jax.random.split(rng)
            diffusion_noise = sde.marginal_prob(None, t_min)[1] * jax.random.normal(step_rng, shape=(noise.shape[0],))
            data = signal + noise + diffusion_noise
        else:
            data = signal + noise
        
        
        plt.figure()
        plt.plot(np.linspace(0., seglen, data.shape[0]), np.clip(data, -60, 60), label='Mock data') # Clipping range in case of massive glitch
        plt.plot(np.linspace(0., seglen, data.shape[0]), signal, label='True signal')
        plt.ylabel('Amplitude')
        plt.xlabel('Time (seconds)')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'data{}.png'.format(i)), dpi=175)
        plt.close() 
        
        print('Initial mass matrix. ', mass_matrix)
        
        print('Sampling {}.'.format(i))
        
        rng, step_rng = jax.random.split(rng)
        theta_initial = jax.random.multivariate_normal(step_rng, theta_true, 0.001**2 * jnp.diag(jnp.diag(mass_matrix @ mass_matrix.T)), shape=(args.num_walkers,))
        theta_initial = jnp.clip(theta_initial, a_min=jnp.array([b[0] for b in param_bounds]), a_max=jnp.array([b[1] for b in param_bounds])) # Clip within prior bounds
        print('initial walker points:', theta_initial)
        
        rng, step_rng = jax.random.split(rng)
        step_burn_fn_jit = jax.jit(step_burn_fn, static_argnums=(3,))
        step_fn_jit = jax.jit(step_fn, static_argnums=(3,))
        
        sample_fn = get_sample_fn(step_fn_jit, step_burn_fn_jit, score_likelihood_vmap, args.num_steps, args.num_burn_steps, mass_matrix, param_bounds, adapt_stepsize=True)
        chain, chain_score = sample_fn(rng, theta_initial, data, sim_args, args.step_size)
        
        ##### EXCEPTION BURN IN PHASE
        #chain = chain[args.num_burn_steps:]
        #chain_score = chain_score[args.num_burn_steps:]

        try:
            np.save(os.path.join(save_dir, 'chains{}.npy').format(i), np.array(chain))
            np.save(os.path.join(save_dir, 'theta_true{}.npy').format(i), np.array(theta_true))
            plot_results(np.array(chain)[args.num_burn_steps:], np.array(chain_score)[args.num_burn_steps:], np.asarray(theta_true), theta_labels, save_dir, i)
        except Exception as e:
            pass                


        
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Perform inference with injected simulated signal.")
    
    parser.add_argument("--sim_type", type=str, choices=["dummy", "td_3d", "td_4d", "td_5d", "td_7d"], help="Choose the type of simulation (dummy, td_4d, td_7d)")
    parser.add_argument("--noise_type", type=str, choices=["white", "slic", "real", "real_glitch"], help="Choose the type of noise (white, slic, real)")
    parser.add_argument("--score_type", type=str, choices=["gaussian", "slic"], help="Choose the type of score function for the noise distribution(gaussian, slic)")
    parser.add_argument("--workdir", type=str, default="./run_slic/", help="Path to the working directory of the model.")
    parser.add_argument("--ifo", type=str, choices=["H1", "L1"], default="H1", help="Instrument ('H1', 'L1').")
    parser.add_argument("--output_folder_name", type=str, default="inference/", help="Output folder to save files in.")
    parser.add_argument("--t_min", type=float, default=0.1, help="Minimum time to fix slic model at.")
    parser.add_argument("--step_size", type=float, default=0.1, help="Step size of MALA sampler.")
    parser.add_argument("--warmup_step_size", type=float, default=0.001, help="Step size of warmup stage of MCMC sampler.")
    parser.add_argument("--num_walkers", type=int, default=16, help="Number of MALA chains to run in parallel.")
    parser.add_argument("--num_steps", type=int, default=2000, help="Number of MALA steps.")
    parser.add_argument("--num_burn_steps", type=int, default=0, help="Number of burn-in MALA steps.")
    parser.add_argument('--add_diffusion_noise', action='store_true', help='Add diffusion noise to data noise.')
    parser.add_argument("--index_array", nargs="+", type=int, help="List of indices to process.")


    args = parser.parse_args()

    args_dict = vars(args)
    
    save_dir = os.path.join(args.workdir, 'inference/', args.output_folder_name)
    os.makedirs(save_dir, exist_ok=True)
    
    # Save parameters used for inference.
    with open(os.path.join(save_dir, "inference.json"), 'w') as file:
        json.dump(args_dict, file)
    
    print(f"Inference parameters saved.")

    main(save_dir, args)
