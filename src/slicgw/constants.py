import numpy as np
import jax.numpy as jnp

jax_dtype = jnp.float32
#jax_dtype = jnp.float64

### Define constants

srate = 4096
seglen = 4
f_low = 20.
f_ref = 20.

N = srate*seglen
dt = 1/srate

# corresponding frequency grid
f = np.fft.rfftfreq(N, d=dt)

# strain scaling factor used in training the network
strain_scale = 1E-23

# for window function
tukey_alpha = 0.1

# define some fixed signal parameters
ifo = 'H1'

# antenna patterns consistent with GW150914
antenna_patterns = {
    'H1': [0.578742411175002, -0.4509478210953121],
    'L1': [-0.5274334329518102, 0.20520960891727436]
}
Fp, Fc = antenna_patterns[ifo]

whiten_scale_factor = 100
td_crop_index_start = 6000
td_crop_index_end = 6000+4096