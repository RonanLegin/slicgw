import numpy as np
import jax.numpy as jnp
import jax
from tqdm import tqdm

jax.config.update("jax_enable_x64", False)


def get_mask_slic_input(f, f_low, Nsize, dtype='float32'):
  mask = np.ones((1,Nsize, 2), dtype=dtype)
  mask[:,f < f_low,:] = 0. # Mask input same as training data
  return jnp.array(mask)

def log_prior_eta(state, min_val, max_val):
    variable = state[:, 1]
    return jnp.where((min_val <= variable) & (variable <= max_val), 0., -jnp.inf)

def get_score_likelihood(forward_fn, jacobian_fn, score_fn, mask_slic_input, mask_additional=None):
    
    def score_likelihood(theta, data, args):
        x = data - forward_fn(theta, args)
        jacobian = jacobian_fn(theta, args)

        x *= mask_slic_input
        score = score_fn(x[:,1:], jnp.zeros((theta.shape[0],), jnp.float32))
        score = jnp.concatenate((jnp.zeros((theta.shape[0],1,2), jnp.float32), score), axis=1)
        score *= mask_slic_input

        #score *= mask_additional
        score = -jnp.einsum('bi,bij->bj', score.reshape(theta.shape[0],-1), jacobian.reshape(theta.shape[0],-1,theta.shape[-1]))
        return score
    
    return score_likelihood




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




def accept(key, log_alpha):
    key, subkey = jax.random.split(key)
    accepted = jnp.log(jax.random.uniform(subkey, log_alpha.shape)) < log_alpha
    return accepted, key


def ula_step(key, current_state, epsilon, score_fn, mass_matrix=None):

    current_position, current_score = current_state

    key, subkey = jax.random.split(key)

    if mass_matrix is None:
        proposed_position = current_position + epsilon * current_score + jnp.sqrt(2 * epsilon) * jax.random.normal(subkey, current_position.shape)
    else:
        proposed_position = current_position + epsilon * jnp.einsum('ij,bj->bi', mass_matrix @ mass_matrix.T, current_score) + jnp.sqrt(2 * epsilon) * jnp.einsum('ij,bj->bi', mass_matrix, jax.random.normal(subkey, current_position.shape))
    
    proposed_score = score_fn(proposed_position)

    return (proposed_position, proposed_score), None , key


def mala_step(key, current_state, epsilon, score_fn, mass_matrix=None):

    current_position, current_score = current_state

    dims = list(range(1, len(current_position.shape)))
    key1, key2 = jax.random.split(key, 2)

    #score_current_state = score_current_state#score_fn(current_state)

    if mass_matrix is None:
        proposed_position = current_position + epsilon * current_score + jnp.sqrt(2 * epsilon) * jax.random.normal(key1, current_position.shape)
    else:
        proposed_position = current_position + epsilon * jnp.einsum('ij,bj->bi', mass_matrix @ mass_matrix.T, current_score) + jnp.sqrt(2 * epsilon) * jnp.einsum('ij,bj->bi', mass_matrix, jax.random.normal(key1, current_position.shape))
    
    #condition = (0. <= proposed_position[..., 1]) & (proposed_position[..., 1] <= 0.25)
    #condition = condition[:, None]  # Expand dimensions to match proposed_state and current_state
    #proposed_state = jnp.where(condition, proposed_state, current_state)

    proposed_score = score_fn(proposed_position)


    delta_logp_val = delta_logp(current_position, proposed_position, score_fn)
    kernel_forward = jnp.sum((proposed_position - current_position - epsilon * current_score) ** 2, axis=dims) / 4 / epsilon
    kernel_backward = jnp.sum((current_position - proposed_position - epsilon * proposed_score) ** 2, axis=dims) / 4 / epsilon
    log_alpha = delta_logp_val - kernel_backward + kernel_forward
    accepted, key2 = accept(key2, log_alpha)

    proposed_position = jnp.where(accepted[..., jnp.newaxis], proposed_position, current_position)
    proposed_score = jnp.where(accepted[..., jnp.newaxis], proposed_score, current_score)

    return (proposed_position, proposed_score), accepted, key2  # Also return the updated key



def get_sample_fn(step_fn, step_burn_fn, score_fn, mass_matrix=None):

  def sample_fn(key, current_position, epsilon, n, burn_in = 0):
      
      current_score = score_fn(current_position)
      current_state = (current_position, current_score)

      chain = []
      chain_score = []
      accepted_count = 0
      total_proposed = 0
      for step in (pbar := tqdm(range(n))):
          if step < burn_in:
                current_state, _, key = step_burn_fn(key, current_state, epsilon, score_fn, mass_matrix)
          else:
                current_state, accepted, key = step_fn(key, current_state, epsilon, score_fn, mass_matrix)
                accepted_count += jnp.sum(accepted)  # Count accepted samples
                total_proposed += accepted.size  # Count total
                pbar.set_description(f"Average acceptance = {accepted_count / total_proposed:.2f}")
                 
          chain.append(current_state[0])
          chain_score.append(current_state[1])
      return chain, chain_score
  return sample_fn






