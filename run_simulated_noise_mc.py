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

from samplers import get_mask_slic_input, get_score_likelihood, get_sample_fn
from samplers import mala_step as step_fn
from samplers import ula_step as step_burn_fn

from simulation import model_mc as gw_source
from ripple import ms_to_Mc_eta
from ripple.waveforms import IMRPhenomD

import pandas as pd
import h5py
from scipy.signal.windows import tukey

dir_name = 'simulated_noise_mc/'
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
# import scipy.signal as sl

# Welch data to get PSD, with settings to give an FFT length of 4s
# psd_freq, psd = sl.welch(full_data, fs=srate, nperseg=4*srate)

# plt.loglog(psd_freq, psd)
# plt.xlabel("Frequency (Hz)")
# plt.ylabel("PSD");

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

# Initialize model.
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
state = ckptr.restore(checkpoint_dir, item=state)

# Setup SDE
sde = VESDE(sigma_min=config.model.sigma_min, sigma_max=config.model.sigma_max, N=config.model.num_scales)
score_noise_fn = get_score_fn(sde, score_model, state.params_ema, state.model_state, train=False, return_state=False)

""" Define forward model and score """

mask_slic_input = get_mask_slic_input(f, f_low, Nsize)
forward_fn = vmap(gw_source, in_axes=(0, None))
jacobian_fn = vmap(jacfwd(gw_source, argnums=0), in_axes=(0, None))
score_likelihood_fn = get_score_likelihood(forward_fn, jacobian_fn, score_noise_fn, mask_slic_input)


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
    'phic': phic
}

# note these aren't really "true" values, simply likely close to the truth
theta_true = jnp.array([[Mc]], dtype=jnp.float32)
theta_labels = ["Mc"]

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

mass_matrix = jnp.diag(jnp.array([8.72e-02], dtype=jnp.float32))



score_fn = lambda theta: score_likelihood_fn(theta, data, args)
step_burn_fn_jit = jax.jit(step_burn_fn, static_argnums=(2,3))
step_fn_jit = jax.jit(step_fn, static_argnums=(2,3))
sample_fn = get_sample_fn(step_fn_jit, step_burn_fn_jit, score_fn, mass_matrix)


""" Sample """

num_steps = 430
step_size = 0.003
batch_size = 32

# Define starting walkers
theta_initial = jnp.array([[theta_true[0,0]]])
rng, step_rng = jax.random.split(rng)
theta_initial += 0.5*jax.random.normal(rng, shape=(batch_size,1))#jnp.tile(theta_initial, (batch_size,1))
rng, step_rng = jax.random.split(rng)



print('Begin sampling.')
chain, chain_score = sample_fn(rng, theta_initial, step_size, num_steps, burn_in=30)



### Plot results ###

chain = np.array(chain)
chain_score = np.array(chain_score)
np.save(dir_name + 'chain.npy', chain)
np.save(dir_name + 'chain_score.npy', chain_score)

ndim = chain.shape[-1]

fig, ax = plt.subplots(ndim, figsize=(8,6*ndim))

for i in range(batch_size):
    ax.plot(chain[:,i,0])
ax.axhline(theta_true[0,0], color='k', label='Truth')
ax.set_title('Langevin Chain Param {}'.format(theta_labels[0]))
plt.legend()
plt.savefig(dir_name + 'chain.png', bbox_inches='tight')

fig, ax = plt.subplots(ndim, figsize=(8,6*ndim))

for i in range(batch_size):
    ax.plot(chain_score[:,i,0])
ax.set_title('Score Chain Param {}'.format(theta_labels[0]))
plt.savefig(dir_name + 'chain_score.png', bbox_inches='tight')

samples = np.array(chain)[100:].reshape(-1,ndim)
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

