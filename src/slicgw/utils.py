import numpy as np
import matplotlib.pyplot as plt
import corner

import jax
import jax.numpy as jnp
from jax import grad, vmap, jacfwd, jit, jacrev, vjp
import orbax

from scoregen.models import State, init_model, get_config, get_score_fn
from scoregen.sampling import get_sampler as get_sampler_noise
from scoregen import VESDE


def read_data(path, **kws):
    with h5py.File(path, 'r') as f:
        t0 = f['meta/GPSstart'][()]
        T = f['meta/Duration'][()]
        h = f['strain/Strain'][:]
        dt = T/len(h)
        time = t0 + dt*np.arange(len(h))
        return pd.Series(h, index=time, **kws)


def load_score_model(config_path, checkpoint_path):
    
    # Get SLIC model config
    config = get_config(config_path)

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
    ckptr = orbax.checkpoint.Checkpointer(orbax.checkpoint.PyTreeCheckpointHandler()) 
    state = ckptr.restore(checkpoint_path, item=state)

    # Setup SDE
    sde = VESDE(sigma_min=config.model.sigma_min, sigma_max=config.model.sigma_max, N=config.model.num_scales)
    
    return score_model, state, sde, rng, config





### Plot results ###

def plot_results(chain, chain_score, truth, labels, filepath, index):
    chain = np.array(chain)
    chain_score = np.array(chain_score)

    N = chain.shape[0]
    batch_size = chain.shape[1]
    ndim = chain.shape[2]
    
    truth = np.array(truth).reshape(ndim)

    fig, axs = plt.subplots(ndim, figsize=(8,6*ndim))
    for j, ax in enumerate(axs):
      for i in range(batch_size):
        ax.plot(chain[:,i,j])
      ax.axhline(truth[j], color='k', label='Truth')
      ax.set_title('Langevin Chain Param {}'.format(labels[j]))
    plt.legend()
    plt.savefig(filepath + 'chain{}.png'.format(index), bbox_inches='tight')

    fig, axs = plt.subplots(ndim, figsize=(8,6*ndim))
    for j, ax in enumerate(axs):
      for i in range(batch_size):
        ax.plot(chain_score[:,i,j])
      ax.set_title('Score Chain Param {}'.format(labels[j]))
    plt.savefig(filepath + 'chain_score{}.png'.format(index), bbox_inches='tight')

    samples = np.array(chain).reshape(-1,ndim)
    fig = plt.figure(figsize=(20,16))
    fig = corner.corner(
        samples,
        color='black',
        labels=labels,
        hist2d_kwargs={"normed": True}, 
        truths=(truth),
        fig=fig, bins=20
        )
    plt.savefig(filepath + 'corner{}.png'.format(index), bbox_inches='tight')


def compute_precond_matrix(score_fn, theta_ref, data, args):
    def partial_gradient(theta, data, args, i):
        gradient_vector = score_fn(theta, data, args)
        return gradient_vector[0, i]

    partial_grad = grad(partial_gradient, argnums=0)
    vmap_partial_grad = vmap(partial_grad, in_axes=(None, None, None, 0))
    hessian_matrix = vmap_partial_grad(theta_ref, data, args, jnp.arange(theta_ref.shape[0]))
    mass_matrix = np.linalg.inv(hessian_matrix[:,0,:])
    
    # Adding noise until the matrix is non-singular
    noise_std = 1e-6
    success = False
    factor_used = None
    while not success:
        try:
            noisy_mass_matrix = mass_matrix + np.eye(mass_matrix.shape[0])*np.random.normal(loc=0, scale=noise_std, size=mass_matrix.shape[0])
            mass_matrix = np.linalg.cholesky(noisy_mass_matrix)
            factor_used = noise_std
            success = True
        except np.linalg.LinAlgError: 
            # This will catch the error if Cholesky decomposition fails.
            noise_std *= 10  # Increase the noise std by an order of magnitude.
            if noise_std > 1: # If you don't want to exceed 1
                raise Exception("Noise added is too large, still couldn't compute Cholesky decomposition.")
    return mass_matrix, factor_used