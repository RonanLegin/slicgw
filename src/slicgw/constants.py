import numpy as np

### Define constants

seglen_upfactor = 2
srate = 4096
seglen = 4
f_low = 20.
f_ref = 20.

N = seglen_upfactor*srate*seglen
dt = 1/srate

# corresponding frequency grid
f = np.fft.rfftfreq(N, d=dt)

# for window function
tukey_alpha = 0.2

# antenna patterns consistent with GW150914
antenna_patterns = {
    'H1': [0.578742411175002, -0.4509478210953121],
    'L1': [-0.5274334329518102, 0.20520960891727436]
}


sim_args = {
    'seglen_upfactor': seglen_upfactor,
    'srate': srate,
    'seglen': seglen,
    'f_low': f_low,
    'f_ref': f_ref,
    'tukey_alpha': tukey_alpha,
    'white_noise_std': 1.0,
    'Mc': 31.0, # default value
    'eta': 0.249, # default value
    'chi1': 0., # default value
    'chi2': 0., # default value
    'dist_mpc': 600., # default value
    'tc': 2., # default value
    'phic': -0.3, # default value
    'inclination': np.pi # default value
    
    
}