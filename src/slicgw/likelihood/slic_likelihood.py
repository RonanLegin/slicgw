import jax.numpy as jnp
import jax

from slicgw.constants import *

def get_score_likelihood(forward_fn_vmap, jacobian_fn_vmap, score_fn, time_slic=0., mask_slic=None, mask_additional=None):
    def score_likelihood(theta, data):
        
        x = data - forward_fn_vmap(theta)
        jacobian = jacobian_fn_vmap(theta)
        
        #if mask_slic is not None:
        #    x *= mask_slic_input
        
       
        score = score_fn(x, time_slic*jnp.ones((theta.shape[0],), dtype=jax_dtype))
        
        #if mask_additional is not None:
        #    score *= mask_additional

        score = -jnp.einsum('bi,bij->bj', score.reshape(theta.shape[0],-1), jacobian.reshape(theta.shape[0],-1,theta.shape[-1]))
        return score
    
    return score_likelihood