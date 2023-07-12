import jax.numpy as jnp
import jax


def get_gaussian_log_likelihood(forward_model, mask_slic_input, mask_additional=None):
    def gaussian_log_likelihood(theta, data, args):
        f = args['f']
        psd = args['psd']
        residual = data - forward_model(theta, args)
        residual *= mask_slic_input
        log_like = -2.*jnp.sum((residual[:,0]**2 + residual[:,1]**2)/psd)*(f[1] - f[0])
        # the last term is delta_f = 1/T where T for us is 4 s
        return log_like
    return gaussian_log_likelihood