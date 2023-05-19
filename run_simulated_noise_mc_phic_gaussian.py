import numpy as np
import matplotlib.pyplot as plt
import corner
import os

import jax
import jax.numpy as jnp
from jax import grad, vmap, jacfwd, jit, jacrev, vjp
import orbax

jax.config.update("jax_enable_x64", False)
#jax.config.update("jax_platform_name", "cpu") 

from scoregen.models import State, init_model, get_config, get_score_fn
from scoregen import VESDE

from samplers import get_mask_slic_input, get_gaussian_log_likelihood, get_sample_fn
from samplers import mala_step as step_fn
from samplers import ula_step as step_burn_fn

from simulation import model_mc_phic as gw_source
from ripple import ms_to_Mc_eta
from ripple.waveforms import IMRPhenomD

import pandas as pd
import h5py
from scipy.signal.windows import tukey

dir_name = 'simulated_noise_mc_phic_gaussian/'
os.makedirs(dir_name, exist_ok=True)


srate = 4096
seglen = 4
f_low = 20.
f_ref = 20.

N = srate*seglen
dt = 1/srate

# corresponding frequency grid
f = np.fft.rfftfreq(N, d=dt)

# length of the frequency array with the zero-freq DC component
# (should be N // 2 + 1)
Nsize = len(f)


# strain scaling factor used in training the network
strain_scale = 1E-23

# Check that the array was constructed properly: df = 1/T
#seglen, 1/(f[1] - f[0])
#Nsize, N // 2 + 1

# Let's get real LIGO data around GW150914.

# !wget https://www.gw-openscience.org/eventapi/html/GWTC-1-confident/GW150914/v3/H-H1_GWOSC_4KHZ_R1-1126257415-4096.hdf
#!wget https://gwosc.org/eventapi/html/GWTC-1-confident/GW150914/v3/H-H1_GWOSC_4KHZ_R1-1126257415-4096.hdf5

def read_data(path, **kws):
    with h5py.File(path, 'r') as f:
        t0 = f['meta/GPSstart'][()]
        T = f['meta/Duration'][()]
        h = f['strain/Strain'][:]
        dt = T/len(h)
        time = t0 + dt*np.arange(len(h))
        return pd.Series(h, index=time, **kws)

path_template = "{i}-{i}1_GWOSC_4KHZ_R1-1126257415-4096.hdf5"
full_data = read_data(path_template.format(i='H'))

# true signal arrives in Hanford around the following trigger time
# (note the `tc` notation refers to "coalescence time")
tc_trigger = 1126259462.423

# find closest time stamp to trigger time
i0 = np.argmin(np.abs(full_data.index - tc_trigger))
tc = full_data.index[i0]

# pick a 4 seglen around that trigger time
i_start = i0 - N // 2
i_end = i_start + N
data_td = full_data.iloc[i_start:i_end]

# make sure that we got the right 4s
len(data_td)*dt, data_td.index[0], data_td.index[-1], tc in data_td.index

# Note that the signal will be right at the middle of the segment, so we will have `tc = 2`, since the segment is 4s long. Also, below we will rescale the data by the factor assumed by the network in training (`strain_scale`).

# let's Fourier transform the data
tukey_alpha = 0.1
w = tukey(len(data_td), tukey_alpha)
data_fd = np.fft.rfft(data_td*w)*dt

# define some fixed signal parameters

ifo = 'H1'

# antenna patterns consistent with GW150914
antenna_patterns = {
    'H1': [0.578742411175002, -0.4509478210953121],
    'L1': [-0.5274334329518102, 0.20520960891727436]
}
Fp, Fc = antenna_patterns[ifo]

# orientation of the orbital angular momentum wrt the line of sight
inclination = np.pi

# note polarization angle angle argument to waveform must be zero;
# this is already accounted for by the antenna patterns



# ------------------------------------
# to be EXTRA certain that we have the right data, plot the whitened data around the trigger time: we should see the signal clearly.
import scipy.signal as sl

# Welch data to get PSD, with settings to give an FFT length of 4s
psd_freq, psd = sl.welch(full_data, fs=srate, nperseg=4*srate)
#print('psd1', jnp.count_nonzero(psd))
psd = jnp.array(psd/strain_scale**2, dtype=jnp.float32)
# plt.loglog(psd_freq, psd)
# plt.xlabel("Frequency (Hz)")
# plt.ylabel("PSD");
# plt.savefig('psd.png')
# # whiten data segment with signal (up to constants)
# data_fd_white = data_fd / np.sqrt(psd)
# data_td_white = np.fft.irfft(data_fd_white) / dt

# t = data_td.index
# m = (t > tc - 0.1) & (t < tc + 0.05)
# plt.plot(t[m], data_td_white[m])
# plt.axvline(tc, c='gray', ls='--')


""" Initialize SLIC model """

# checkpoint directories for slic model
checkpoint_dir = 'checkpoints/98/default/'
config_path = 'configH.json'

# Get SLIC model config
config = get_config(config_path)

rng = jax.random.PRNGKey(config.seed)
rng, step_rng = jax.random.split(rng)

""" Define forward model and score """

mask_slic_input = get_mask_slic_input(f, f_low, Nsize)
forward_fn = vmap(gw_source, in_axes=(0, None))
gaussian_log_likelihood = get_gaussian_log_likelihood(gw_source, mask_slic_input[0])
score_likelihood_fn = vmap(grad(gaussian_log_likelihood, argnums=0), in_axes=(0, 0, None))




""" Create simulated data """

m1_msun = 35
m2_msun = 32
chi1 = 0.
chi2 = 0.
tc = 2.0
phic = 0.0
dist_mpc = 400

Mc, eta = ms_to_Mc_eta(np.array([m1_msun, m2_msun]))

args = {
    'inclination': inclination,
    'polarization_angle': 0.,
    'f': jnp.array(f, dtype=jnp.float32),
    'f_ref': f_ref,
    'strain_scale': strain_scale,
    'Fp': Fp,
    'Fc': Fc,
    'eta': eta,
    'chi1': chi1,
    'chi2': chi2,
    'dist_mpc': dist_mpc,
    'tc': tc,
    'psd': psd
}

# note these aren't really "true" values, simply likely close to the truth
theta_true = jnp.array([[Mc, phic]], dtype=jnp.float32)
theta_labels = ["Mc", "phic"]

noise = jnp.array(np.load('simulated_noise.npy'), dtype=jnp.float32)[1]
signal = gw_source(theta_true[0], args)
data = signal[1:] + noise
data = data[None]
data = jnp.concatenate((jnp.zeros((1,1,2), dtype=jnp.float32), data), axis=1)
data *= mask_slic_input # this applies f < f_low = 0


# # plot real data
# fig, axs = plt.subplots(1,2, figsize=(12,6))
# axs[0].plot(f, data[0,:,0])
# axs[0].set_title('Real part')
# axs[0].set_ylim(-20,20)
# axs[1].plot(f, data[0,:,1])
# axs[1].set_title('Imaginary part')
# axs[1].set_ylim(-20,20)
# plt.axvline(20, ls='--', c='gray')
# plt.axvline(250, ls='-.', c='gray')
# plt.show()





""" Define mass matrix """

# def partial_gradient(theta, data, args, i):
#     gradient_vector = score_likelihood_fn(theta, data, args)
#     return gradient_vector[0, i]
 
# partial_grad = grad(partial_gradient, argnums=0)
# vmap_partial_grad = vmap(partial_grad, in_axes=(None, None, None, 0))
# hessian_matrix = vmap_partial_grad(theta_true, data, args, jnp.arange(theta_true.shape[1]))
# mass_matrix = np.linalg.inv(hessian_matrix[:,0,:])

mass_matrix = jnp.sqrt(jnp.array(
[[0.01080182, 0.0],
 [0.01422976, 0.01905867]],
 dtype=jnp.float32))
print(mass_matrix @ mass_matrix.T)





""" Sample """

num_steps = 50000
step_size = 0.003
batch_size = 32

def wrap_score_func(data, args):
    def score_func(theta):
        return score_likelihood_fn(theta, jnp.tile(data, (theta.shape[0],1,1)), args)
    return score_func

#score_fn = lambda theta: score_likelihood_fn(theta, jnp.tile(data, (batch_size,1,1)), args)
score_fn = wrap_score_func(data, args)

step_burn_fn_jit = jax.jit(step_burn_fn, static_argnums=(2,3))
step_fn_jit = jax.jit(step_fn, static_argnums=(2,3))
sample_fn = get_sample_fn(step_fn_jit, step_burn_fn_jit, score_fn, mass_matrix)

# Define starting walkers
theta_initial = jnp.array([[theta_true[0,0], theta_true[0,1]]])
rng, step_rng = jax.random.split(rng)
theta_initial = jnp.tile(theta_initial, (batch_size,1))
theta_initial += jnp.array([[2.0, 3.0]])*jax.random.normal(rng, shape=(batch_size,2))
rng, step_rng = jax.random.split(rng)


print('Begin sampling.')
chain, chain_score = sample_fn(rng, theta_initial, step_size, num_steps, burn_in=2000)



### Plot results ###

chain = np.array(chain)
chain_score = np.array(chain_score)
np.save(dir_name + 'chain.npy', chain)
np.save(dir_name + 'chain_score.npy', chain_score)

ndim = chain.shape[-1]

fig, axs = plt.subplots(ndim, figsize=(8,6*ndim))
for j, ax in enumerate(axs):
  for i in range(batch_size):
    ax.plot(chain[:,i,j])
  ax.axhline(theta_true[0,j], color='k', label='Truth')
  ax.set_title('Langevin Chain Param {}'.format(theta_labels[j]))
plt.legend()
plt.savefig(dir_name + 'chain.png', bbox_inches='tight')

fig, axs = plt.subplots(ndim, figsize=(8,6*ndim))
for j, ax in enumerate(axs):
  for i in range(batch_size):
    ax.plot(chain_score[:,i,j])
  ax.set_title('Score Chain Param {}'.format(theta_labels[j]))
plt.savefig(dir_name + 'chain_score.png', bbox_inches='tight')

samples = np.array(chain).reshape(-1,ndim)
fig = plt.figure(figsize=(20,16))
fig = corner.corner(
    samples,
    color='black',
    labels=theta_labels,
    hist2d_kwargs={"normed": True}, 
    truths=(theta_true[0,:]),
    fig=fig, bins=20
    )
plt.savefig(dir_name + 'corner.png', bbox_inches='tight')

