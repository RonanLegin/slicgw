import jax.numpy as jnp
import jax
from tqdm import tqdm


def get_sample_fn(step_fn, step_burn_fn, score_fn, mass_matrix=None):

  def sample_fn(rng, current_position, epsilon, n, burn_in = 0):
      
      current_score = score_fn(current_position)
      current_state = (current_position, current_score)
    
      chain = []
      chain_score = []

      chain.append(current_state[0])
      chain_score.append(current_state[1])
      
      accepted_count = 0
      total_proposed = 0
      for step in (pbar := tqdm(range(n))):
          if step < burn_in:
                current_state, _, rng = step_burn_fn(rng, current_state, epsilon, score_fn, mass_matrix)
          else:
                current_state, accepted, rng = step_fn(rng, current_state, epsilon, score_fn, mass_matrix)
                accepted_count += jnp.sum(accepted)  # Count accepted samples
                total_proposed += accepted.size  # Count total
                pbar.set_description(f"Average acceptance = {accepted_count / total_proposed:.2f}")
                 
          chain.append(current_state[0])
          chain_score.append(current_state[1])
      return chain, chain_score
  return sample_fn