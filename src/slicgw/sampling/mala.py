import jax.numpy as jnp
import jax

from slicgw.constants import *


def simps(f, a: float, b: float, T: int = 2) -> jnp.array:
    if T % 2 == 1:
        raise ValueError("N must be an even integer.")
    dx = (b - a) / T
    x = jnp.linspace(a, b, T + 1)
    y = f(x)
    S = dx / 3 * jnp.sum(y[0:-1:2] + 4 * y[1::2] + y[2::2], axis=0)
    return S


def delta_logp(current_state, proposed_state, score_fn, delta_logp_steps=2):
    v = proposed_state - current_state
    cs = current_state
    T = delta_logp_steps + 1
    B, *D = cs.shape
    
    def integrand(t):
        gamma_t = t.reshape(T, 1, *[1]*len(D)) * v[jnp.newaxis] + cs[jnp.newaxis]
        score = score_fn(gamma_t.reshape(T * B, *D))
        v_repeated = jnp.tile(v, (T, *[1]*len(D)))
        return jnp.einsum("td, td -> t", score.reshape(T*B, -1), v_repeated.reshape(T*B, -1)).reshape(T, B)

    return simps(integrand, 0., 1., delta_logp_steps)


def accept(rng, log_alpha):
    rng, step_rng = jax.random.split(rng)
    accepted = jnp.log(jax.random.uniform(step_rng, log_alpha.shape, dtype=jax_dtype)) < log_alpha
    return accepted, rng



def mala_step(rng, current_state, epsilon, score_fn, mass_matrix=None):

    current_position, current_score = current_state

    dims = list(range(1, len(current_position.shape)))
    rng, step_rng = jax.random.split(rng)


    if mass_matrix is None:
        proposed_position = current_position + epsilon * current_score + jnp.sqrt(2 * epsilon) * jax.random.normal(step_rng, current_position.shape, dtype=jax_dtype)
        
        proposed_score = score_fn(proposed_position)

        delta_logp_val = delta_logp(current_position, proposed_position, score_fn, delta_logp_steps=10)

        kernel_forward = jax.scipy.stats.multivariate_normal.logpdf(
            proposed_position,
            current_position + epsilon * current_score,
            2 * epsilon)

        kernel_backward = jax.scipy.stats.multivariate_normal.logpdf(
            current_position, 
            proposed_position + epsilon * proposed_score,
            2 * epsilon)
        
    else:
        proposed_position = current_position + epsilon * jnp.einsum('ij,bj->bi', mass_matrix @ mass_matrix.T, current_score) + jnp.sqrt(2 * epsilon) * jnp.einsum('ij,bj->bi', mass_matrix, jax.random.normal(step_rng, current_position.shape, dtype=jax_dtype))
        
        proposed_score = score_fn(proposed_position)

        delta_logp_val = delta_logp(current_position, proposed_position, score_fn, delta_logp_steps=10)

        kernel_forward = jax.scipy.stats.multivariate_normal.logpdf(
            proposed_position,
            current_position + epsilon * jnp.einsum('ij,bj->bi', mass_matrix @ mass_matrix.T, current_score),
            2 * epsilon * mass_matrix @ mass_matrix.T)

        kernel_backward = jax.scipy.stats.multivariate_normal.logpdf(
            current_position, 
            proposed_position + epsilon * jnp.einsum('ij,bj->bi', mass_matrix @ mass_matrix.T, proposed_score),
            2 * epsilon * mass_matrix @ mass_matrix.T)
    

    # ###### THIS ASSUMES ETA IS LAST PARAMETER
    # condition = (0. <= proposed_position[..., -1]) & (proposed_position[..., -1] <= 0.25)
    # condition = condition[:, None]  # Expand dimensions to match proposed_state and current_state
    # proposed_position = jnp.where(condition, proposed_position, current_position)
    # ##############
    



    log_alpha = delta_logp_val - kernel_backward + kernel_forward
    accepted, rng = accept(rng, log_alpha)

    proposed_position = jnp.where(accepted[..., jnp.newaxis], proposed_position, current_position)
    proposed_score = jnp.where(accepted[..., jnp.newaxis], proposed_score, current_score)

    return (proposed_position, proposed_score), accepted, rng  # Also return the updated key
