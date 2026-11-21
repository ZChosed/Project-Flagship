import jax.numpy as jnp
import math

def point_source(shape, dx, x_s, y_s, q):
    """Heat input from a point torch on the top surface.

    shape : (nx, ny, nz)
    dx    : voxel size [m]
    x_s, y_s : torch position [m]; require 0.5*dx <= x_s <= (nx - 1.5)*dx
    q     : absorbed power eta*P [W]
    returns Q : float64 (nx, ny, nz), W per cell (the units build_b expects)
    """
    s_x = x_s / dx - 1/2
    j0 = jnp.floor(y_s / dx).astype(jnp.int32)
    i0 = jnp.floor(s_x).astype(jnp.int32)
    w = s_x - i0
    Q = jnp.zeros(shape, dtype=jnp.float64)
    Q = Q.at[i0,j0,-1].add((1-w) * q)
    Q = Q.at[i0 + 1, j0, -1].add(w * q)
    return Q
    