import jax
import jax.numpy as jnp
from ripple.waveforms import IMRPhenomD

from slicgw.constants import *




def model1(theta, args):
    
    f = args['f']
    f_ref = args['f_ref']
    strain_scale = args['strain_scale']
    Fp = args['Fp']
    Fc = args['Fc']
    
    Mc = theta[0]
    eta = theta[1]
    chi1 = theta[2]
    chi2 = theta[3]
    dist_mpc = theta[4] * strain_scale
    tc = theta[5]
    phic = theta[6]
    inclination = theta[7]

    theta = jnp.array([Mc, eta, chi1, chi2, dist_mpc, tc, phic, inclination, 0.], dtype=jax_dtype)
    hp_ripple, hc_ripple = IMRPhenomD.gen_IMRPhenomD_polar(f, theta, f_ref) # Remove zero frequency

    source_p = Fp * jnp.stack([hp_ripple.real, hp_ripple.imag])
    source_c = Fc * jnp.stack([hc_ripple.real, hc_ripple.imag])
    source = source_p + source_c # the signal at a given detector is h = Fp*hc + Fc*hc
    source = jnp.transpose(source, axes=(1,0))
    source = jnp.nan_to_num(source)
    return jnp.asarray(source, dtype=jax_dtype)


def model_mc_phic(theta, args):
    inclination = args['inclination']
    #polarization_angle = args['polarization_angle'] # polarization angle should always be zero when calling the waveform, so there's no point to this (MAX)
    f = args['f']
    f_ref = args['f_ref']
    strain_scale = args['strain_scale']
    Fp = args['Fp']
    Fc = args['Fc']

    Mc = theta[0]
    phic = theta[1]

    eta = args['eta']#theta[1]
    chi1 = args['chi1']#theta[2]
    chi2 = args['chi2']#theta[3]
    dist_mpc = args['dist_mpc'] * strain_scale #theta[4] * strain_scale # the strain is inversely proportional to the distance, so we can just multiply the distance by the strain scale instead of dividing the strain below
    tc = args['tc']#theta[5]

    theta = jnp.array([Mc, eta, chi1, chi2, dist_mpc, tc, phic, inclination, 0.], dtype=jax_dtype)
    hp_ripple, hc_ripple = IMRPhenomD.gen_IMRPhenomD_polar(f[1:], theta, f_ref) # Remove zero frequency

    source_p = Fp * jnp.stack([hp_ripple.real, hp_ripple.imag])
    source_c = Fc * jnp.stack([hc_ripple.real, hc_ripple.imag])
    source = source_p + source_c # the signal at a given detector is h = Fp*hc + Fc*hc
    source = jnp.transpose(source, axes=(1,0))
    source = jnp.concatenate((jnp.zeros((1,2), jax_dtype), source), axis=0) # Add back zero frequency
    source = jnp.nan_to_num(source)
    return jnp.asarray(source, dtype=jax_dtype)[81:593]