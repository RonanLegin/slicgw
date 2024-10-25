import jax
import jax.numpy as jnp
from ripple.waveforms import IMRPhenomD

from slicgw.constants import *
from slicgw.constants import dt

jax.config.update("jax_enable_x64", True)


def sample_uniform_prior(rng, param_bounds):
    
    theta = []
    for bounds in param_bounds:
        rng, step_rng = jax.random.split(rng)
        param = jax.random.uniform(step_rng, shape=(), dtype=jnp.float64, minval=bounds[0], maxval=bounds[1])
        theta.append(param)
    return rng, jnp.array(theta)


def model_td_3d(theta, args):
    
        
    f = args['f']
    f_filtered = args['f_filtered']
    f_ref = args['f_ref']
    Fp = args['Fp']
    Fc = args['Fc']
    fourier_mean_avg = args['fourier_mean_avg']
    fourier_sigma_avg = args['fourier_sigma_avg']
    
    Mc = theta[0]
    eta = args['eta']
    chi1 = args['chi1']
    chi2 = args['chi2']
    dist_mpc = theta[1]
    tc = theta[2]
    phic = args['phic']
    inclination = args['inclination']
    
    theta = jnp.array([Mc, eta, chi1, chi2, dist_mpc, tc, phic, inclination, 0.], dtype=jnp.float64)
    
    hp_ripple, hc_ripple = IMRPhenomD.gen_IMRPhenomD_hphc(f_filtered, theta, f_ref)
    
    source_fd = Fp * hp_ripple + Fc * hc_ripple
    source_fd /= jnp.sqrt(f.shape[0])
    source_fd = (source_fd - fourier_mean_avg)/fourier_sigma_avg
    source_fd = jnp.pad(source_fd, (f.shape[0] - source_fd.shape[0], 0), constant_values=0.)
    source_td_whiten = jnp.fft.irfft(source_fd, norm='ortho')
    
    return source_td_whiten


def model_td_4d(theta, args):
    
    f = args['f']
    f_filtered = args['f_filtered']
    f_ref = args['f_ref']
    Fp = args['Fp']
    Fc = args['Fc']
    fourier_mean_avg = args['fourier_mean_avg']
    fourier_sigma_avg = args['fourier_sigma_avg']
    window = args['window']
    
    Mc = theta[0]
    eta = theta[1]
    chi1 = args['chi1']
    chi2 = args['chi2']
    dist_mpc = theta[2]
    tc = args['tc']
    phic = theta[3]
    inclination = args['inclination']
    
    theta = jnp.array([Mc, eta, chi1, chi2, dist_mpc, tc, phic, inclination, 0.], dtype=jnp.float64)
    
    hp_ripple, hc_ripple = IMRPhenomD.gen_IMRPhenomD_hphc(f_filtered, theta, f_ref)
    
    source_fd = Fp * hp_ripple + Fc * hc_ripple
    source_fd /= jnp.sqrt(f.shape[0])
    source_fd = (source_fd - fourier_mean_avg)/fourier_sigma_avg
    source_fd = jnp.pad(source_fd, (f.shape[0] - source_fd.shape[0], 0), constant_values=0.)
    source_td_whiten = jnp.fft.irfft(source_fd, norm='ortho') * window
    return source_td_whiten


def model_td_5d(theta, args):
        
    f = args['f']
    f_filtered = args['f_filtered']
    f_ref = args['f_ref']
    Fp = args['Fp']
    Fc = args['Fc']
    fourier_mean_avg = args['fourier_mean_avg']
    fourier_sigma_avg = args['fourier_sigma_avg']
    window = args['window']
    
    Mc = theta[0]
    eta = theta[1]
    chi1 = args['chi1']
    chi2 = args['chi2']
    dist_mpc = theta[2]
    tc = theta[3]
    phic = theta[4]
    inclination = args['inclination']
    
    theta = jnp.array([Mc, eta, chi1, chi2, dist_mpc, tc, phic, inclination, 0.], dtype=jnp.float64)
    
    hp_ripple, hc_ripple = IMRPhenomD.gen_IMRPhenomD_hphc(f_filtered, theta, f_ref)
    
    source_fd = Fp * hp_ripple + Fc * hc_ripple
    source_fd /= jnp.sqrt(f.shape[0])
    source_fd = (source_fd - fourier_mean_avg)/fourier_sigma_avg
    source_fd = jnp.pad(source_fd, (f.shape[0] - source_fd.shape[0], 0), constant_values=0.)
    source_td_whiten = jnp.fft.irfft(source_fd, norm='ortho') * window
    return source_td_whiten



def model_td_7d(theta, args):
    
    f = args['f']
    f_filtered = args['f_filtered']
    f_ref = args['f_ref']
    Fp = args['Fp']
    Fc = args['Fc']
    fourier_mean_avg = args['fourier_mean_avg']
    fourier_sigma_avg = args['fourier_sigma_avg']
    window = args['window']
    
    Mc = theta[0]
    eta = theta[1]
    chi1 = theta[2]
    chi2 = theta[3]
    dist_mpc = theta[4]
    tc = theta[5]
    phic = theta[6]
    inclination = args['inclination']
    
    theta = jnp.array([Mc, eta, chi1, chi2, dist_mpc, tc, phic, inclination, 0.], dtype=jnp.float64)
    
    hp_ripple, hc_ripple = IMRPhenomD.gen_IMRPhenomD_hphc(f_filtered, theta, f_ref)
    
    source_fd = Fp * hp_ripple + Fc * hc_ripple
    source_fd = (source_fd - fourier_mean_avg)/fourier_sigma_avg
    source_fd = jnp.pad(source_fd, (f.shape[0] - source_fd.shape[0], 0), constant_values=0.)
    source_td_whiten = jnp.fft.irfft(source_fd, norm='ortho') * window
    return source_td_whiten
    
    
    
def model_dummy(theta, args):
    mu = theta[0]
    sigma = theta[1]
    f = args['f']
    x = jnp.linspace(-20, 20, 2*f.shape[0])
    gaussian = jnp.exp(-((x - mu) ** 2) / (2 * sigma ** 2))
    y = jnp.sin(x) * gaussian 
    return y
    


