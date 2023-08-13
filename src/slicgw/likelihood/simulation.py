import jax
import jax.numpy as jnp
from ripple.waveforms import IMRPhenomD

from slicgw.constants import *
from slicgw.constants import dt, jax_dtype, td_crop_index_start, td_crop_index_end



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

def model1_td_white(theta, args):
    
    f = args['f']
    f_ref = args['f_ref']
    #strain_scale = args['strain_scale']
    Fp = args['Fp']
    Fc = args['Fc']
    
    psd_sqrt = args['psd_sqrt']
    
    Mc = theta[0]
    eta = theta[1]
    chi1 = theta[2]
    chi2 = theta[3]
    dist_mpc = theta[4]# * strain_scale
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
    
    source_fd = source[:,0] + 1j*source[:,1]
    source_fd_whiten = source_fd / psd_sqrt#jnp.sqrt(psd_avg)
    source_td_whiten = jnp.fft.irfft(source_fd_whiten) / dt
    source_td_whiten = source_td_whiten[td_crop_index_start:td_crop_index_end] / whiten_scale_factor
    
    return jnp.asarray(source_td_whiten, dtype=jax_dtype)

def model_td_toy(theta, args):
    amp = theta[0]
    x = jnp.linspace(-100., 100.,  td_crop_index_end-td_crop_index_start, dtype=jax_dtype)
    source_td = jnp.where(x == 0, amp * 1.0, amp * jnp.sin(jnp.pi * x) / (jnp.pi * x))
    return jnp.asarray(source_td, dtype=jax_dtype)
    
