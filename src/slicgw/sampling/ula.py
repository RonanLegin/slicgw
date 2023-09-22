import jax.numpy as jnp
import jax



def ula_step(rng, current_state, epsilon, score_fn, mass_matrix=None):

    current_position, current_score = current_state
    rng, step_rng = jax.random.split(rng)

    if mass_matrix is None:
        proposed_position = current_position + epsilon * current_score + jnp.sqrt(2 * epsilon) * jax.random.normal(step_rng, current_position.shape)
    else:
        proposed_position = current_position + epsilon * jnp.einsum('ij,bj->bi', mass_matrix @ mass_matrix.T, current_score) + jnp.sqrt(2 * epsilon) * jnp.einsum('ij,bj->bi', mass_matrix, jax.random.normal(step_rng, current_position.shape))
        
    proposed_score = score_fn(proposed_position)

    return (proposed_position, proposed_score), None , rng
