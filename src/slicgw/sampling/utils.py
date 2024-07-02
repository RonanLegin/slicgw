import jax.numpy as jnp
import jax
from tqdm import tqdm
import numpy as np

def get_sample_fn(step_fn, step_burn_fn, score_fn, n, burn_in, mass_matrix=None, param_bounds=None, adapt_stepsize=True):
    
    if mass_matrix is None:
        mass_matrix = jnp.eye(n_dim)
        
    def sample_fn(rng, current_position, data, args, step_size):
        
        n_walkers = current_position.shape[0]
        n_dim = current_position.shape[1]
            
        adapted_mass_matrix = jnp.copy(mass_matrix)
        
        def wrap_score_fn(score_fn, data, args):
            data = jnp.expand_dims(data, axis=0)
            def wrapped_score_fn(theta):
                bdata = jnp.tile(data, (theta.shape[0], 1))
                return score_fn(theta, bdata, args)
            return wrapped_score_fn
            
        wscore_fn = wrap_score_fn(score_fn, data, args)
        
        current_score = wscore_fn(current_position)
        current_state = (current_position, current_score)

        warmup_chain = []
        chain = []
        chain_score = []

        chain.append(np.array(current_state[0]))
        chain_score.append(np.array(current_state[1]))
        
        target_acceptance_rate = 0.60  # Target acceptance rate for MALA
        K_p = 0.5  # Proportional gain factor
        ema_rate = 0.8
        
        max_window_size = 1000
        cov_window_size = 100
        step_window = 100
        
        acceptance_rate = 0.
        accepted_count = 0
        total_proposed = 0
        for step in (pbar := tqdm(range(n + burn_in))):
            if step < burn_in:
                current_state, accepted, rng = step_burn_fn(rng, current_state, step_size, wscore_fn, adapted_mass_matrix, param_bounds)
                accepted_count += jnp.sum(accepted)  # Count accepted samples
                total_proposed += accepted.size  # Count total
                acceptance_rate += jnp.sum(accepted)/accepted.size
                
                if (step + 1) % 5 == 0:
                    acceptance_rate /= 5
                    error = target_acceptance_rate - acceptance_rate
                    scaling_factor = 1 + K_p * error
                    step_size /= scaling_factor
                    pbar.set_description(f"Warmup acceptance = {acceptance_rate:.2f}, step_size = {step_size}")
                    acceptance_rate = 0.
                    
                if ((step + 1) == step_window + cov_window_size) or (step == burn_in - 200): 
                    n_usable = cov_window_size - cov_window_size % int(cov_window_size//5)
                    cov = jnp.cov(np.transpose(np.mean(np.array(chain[-n_usable:]).reshape(-1, int(cov_window_size//5), n_walkers, n_dim), axis=1).reshape(-1, n_dim)), ddof=1)
                    adapted_mass_matrix = jnp.linalg.cholesky(cov)
                    
                    if cov_window_size <= max_window_size:
                        cov_window_size *= 1.4
                        cov_window_size = int(cov_window_size)
                        print('cov_window_size:', cov_window_size)
                        print(jnp.diag(adapted_mass_matrix))
                    step_window = (step + 1)    
            else:
                current_state, accepted, rng = step_fn(rng, current_state, step_size, wscore_fn, adapted_mass_matrix, param_bounds)
                accepted_count += jnp.sum(accepted)  # Count accepted samples
                total_proposed += accepted.size  # Count total
                acceptance_rate = jnp.sum(accepted)/accepted.size
                
                pbar.set_description(f"Acceptance = {accepted_count / total_proposed:.2f}, step_size = {step_size}")

            chain.append(np.array(current_state[0]))
            chain_score.append(np.array(current_state[1]))

        chain = np.array(chain)
        chain_score = np.array(chain_score)
        return chain, chain_score
    return sample_fn
