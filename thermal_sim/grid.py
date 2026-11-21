import jax.numpy as jnp

def face_counts(active):
    """Count active and exposed faces for each cell
    
    active : bool array, shape (nx, ny, nz)
    returns (n_act, n_exp), both int arrays of shape (nx, ny, nz)
        n_act[i] = number of the 6 face neighbours of i that are active
                    (outside the grid counts as inactive)
        n_exp[i] = 6 - n_act[i] if i is active, else 0
    """
    p = jnp.pad(active, 1, constant_values=False)
    n_act = p[2:, 1:-1, 1:-1].astype(jnp.int32) + p[:-2, 1:-1, 1:-1].astype(jnp.int32) + p[1:-1,2:,1:-1].astype(jnp.int32) + p[1:-1,:-2,1:-1].astype(jnp.int32) + p[1:-1,1:-1,2:].astype(jnp.int32) + p[1:-1,1:-1,:-2].astype(jnp.int32)
    return n_act, jnp.where(active, 6-n_act,0)