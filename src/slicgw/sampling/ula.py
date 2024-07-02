import jax.numpy as jnp
import jax

from slicgw.constants import *

def ula_step(rng, current_state, epsilon, score_fn, mass_matrix):

    current_position, current_score = current_state
    rng, step_rng = jax.random.split(rng)
    
    mass_matrix_2 = jnp.matmul(mass_matrix, mass_matrix.T)

    proposed_position = current_position + epsilon * jnp.matmul(current_score, mass_matrix_2.T) + jnp.sqrt(2 * epsilon) * jnp.matmul(jax.random.normal(step_rng, current_position.shape, dtype=jax_dtype), mass_matrix.T)
        
    proposed_score = score_fn(proposed_position)

    return (proposed_position, proposed_score), None , rng
