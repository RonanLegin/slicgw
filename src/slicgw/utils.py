import numpy as np
import matplotlib.pyplot as plt
import corner

import pandas as pd
import h5py
import os


def read_data(path, **kws):
    with h5py.File(path, 'r') as f:
        t0 = f['meta/GPSstart'][()]
        T = f['meta/Duration'][()]
        h = f['strain/Strain'][:]
        dt = T/len(h)
        time = t0 + dt*np.arange(len(h))
        return pd.Series(h, index=time, **kws)

def plot_complex_scatter(fourier_data, filepath, index, x_range=(-1.5, 1.5), y_range=(-1.5, 1.5), dot_size=2):
    # reshape the array to be at least 2D
    fourier_data = np.atleast_2d(fourier_data)

    # separate real and imaginary parts
    real_part = fourier_data[..., 0]
    imag_part = fourier_data[..., 1]
    
    A = np.sqrt(real_part**2 + imag_part**2)
    real_part_normalized = real_part/A
    imag_part_normalized  = imag_part/A
    
    # plot
    fig, axs = plt.subplots(fourier_data.shape[0], figsize=(8,6*fourier_data.shape[0]))
    axs = np.atleast_1d(axs)  # ensure axs is an array, even when ndim = 1

    for i, ax in enumerate(axs):
        ax.scatter(real_part_normalized [i], imag_part_normalized [i], s=dot_size, alpha=0.3)  # scatter plot
        ax.set_xlabel('Real Part')
        ax.set_ylabel('Imaginary Part')
        ax.set_xlim(x_range)  # set x-axis limits
        ax.set_ylim(y_range)  # set y-axis limits
        ax.set_title(f'Scatter plot for batch {i+1}')
    plt.savefig(filepath + 'scatter{}.png'.format(index), bbox_inches='tight')
    plt.close(fig)

    
def plot_phase_vs_frequency(fourier_data, f, filepath, index, dot_size=5):
    # reshape the array to be at least 3D, and combine real and imaginary parts
    fourier_data = np.atleast_3d(fourier_data)
    complex_data = fourier_data[..., 0] + 1j * fourier_data[..., 1]

    # calculate phase
    phase = np.angle(complex_data)

    # plot
    fig, axs = plt.subplots(fourier_data.shape[0], figsize=(8,6*fourier_data.shape[0]))
    axs = np.atleast_1d(axs)  # ensure axs is an array, even when ndim = 1

    for i, ax in enumerate(axs):
        ax.scatter(f, phase[i], s=dot_size)  # scatter plot with specified dot size
        ax.set_xlabel('Frequency')
        ax.set_ylabel('Phase')
        ax.set_title(f'Phase vs Frequency plot for batch {i+1}')
    plt.savefig(filepath + 'phase_vs_frequency{}.png'.format(index), bbox_inches='tight')
    plt.close(fig)

    
    
def plot_results(chain, chain_score, truth, labels, filepath, index):
    chain = np.array(chain)
    chain_score = np.array(chain_score)
    
    N = chain.shape[0]
    batch_size = chain.shape[1]
    ndim = chain.shape[2]
    
    truth = np.array(truth).reshape(ndim)

    fig, axs = plt.subplots(ndim, figsize=(8,6*ndim))
    axs = np.atleast_1d(axs)  # ensure axs is an array, even when ndim = 1
    for j, ax in enumerate(axs):
      for i in range(batch_size):
        ax.plot(chain[:,i,j])
      ax.axhline(truth[j], color='k', label='Truth')
      ax.set_title('Langevin Chain Param {}'.format(labels[j]))
    plt.legend()
    plt.savefig(filepath + 'chain{}.png'.format(index), bbox_inches='tight')
    plt.close(fig)

    fig, axs = plt.subplots(ndim, figsize=(8,6*ndim))
    axs = np.atleast_1d(axs)  # ensure axs is an array, even when ndim = 1
    for j, ax in enumerate(axs):
      for i in range(batch_size):
        ax.plot(chain_score[:,i,j])
      ax.set_title('Score Chain Param {}'.format(labels[j]))
    plt.savefig(filepath + 'chain_score{}.png'.format(index), bbox_inches='tight')
    plt.close(fig)

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
    plt.close(fig)


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
