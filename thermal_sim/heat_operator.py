"""Matrix-free implicit Euler operator A = C/dt + K (conduction + convection).

For active cell i:
    (A T)_i = C_dt*T_i + G*(n_act_i*T_i - sum_{active nbrs j} T_j) + H*n_exp_i*T_i
For inactive cell i:
    (A T)_i = T_i            (identity row, keeps A SPD)

C_dt = rho*c*dx**3/dt,  G = k*dx,  H = h*dx**2   (all W/K)
"""
import jax.numpy as jnp


def neighbour_sum(x):
    """Sum of the 6 face neighbours of each cell; outside the grid counts as 0.

    x : array, shape (nx, ny, nz)
    returns array, shape (nx, ny, nz)
    """
    p = jnp.pad(x, 1)  # pads with zeros
    return (p[2:, 1:-1, 1:-1] + p[:-2, 1:-1, 1:-1]     # +x, -x
            + p[1:-1, 2:, 1:-1] + p[1:-1, :-2, 1:-1]   # +y, -y
            + p[1:-1, 1:-1, 2:] + p[1:-1, 1:-1, :-2])  # +z, -z


def apply_A(T, active, n_act, n_exp, C_dt, G, H):
    """Return A @ T.

    T      : float64 (nx, ny, nz)
    active : bool    (nx, ny, nz)
    n_act, n_exp : int (nx, ny, nz), from grid.face_counts(active)
    C_dt, G, H   : scalars (W/K)
    """
    Tm = jnp.where(active, T, 0.0)        # inactive neighbours contribute nothing
    nbr = neighbour_sum(Tm)
    out_active = C_dt * T + G * (n_act * T - nbr) + H * n_exp * T
    return jnp.where(active, out_active, T)


def diag_A(active, n_act, n_exp, C_dt, G, H):
    """Diagonal of A (Jacobi preconditioner). Shape (nx, ny, nz)."""
    return jnp.where(active, C_dt + G * n_act + H * n_exp, 1.0)
