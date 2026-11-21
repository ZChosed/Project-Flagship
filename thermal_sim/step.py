"""One implicit (backward) Euler step:  A T+ = b,  A = C/dt + K.

b_i = C_dt*T_i + H*n_exp_i*T_inf + Q_i   (active)
b_i = T_i                                (inactive: identity row)
"""
import jax
import jax.numpy as jnp

from heat_operator import apply_A, diag_A
from pcg import pcg


def build_b(T, active, n_exp, C_dt, H, T_inf, Q):
    """Right-hand side. All arrays (nx, ny, nz); C_dt, H, T_inf scalars. Q in W per cell."""
    return jnp.where(active, C_dt * T + H * n_exp * T_inf + Q, T)


@jax.jit
def step(T, active, n_act, n_exp, C_dt, G, H, T_inf, Q, tol=1e-12):
    """Advance T by one step. Returns (T_new, iters, relres)."""
    b = build_b(T, active, n_exp, C_dt, H, T_inf, Q)
    d = diag_A(active, n_act, n_exp, C_dt, G, H)
    A = lambda x: apply_A(x, active, n_act, n_exp, C_dt, G, H)
    return pcg(A, b, x0=T, diag=d, tol=tol)   # warm start from the old field
