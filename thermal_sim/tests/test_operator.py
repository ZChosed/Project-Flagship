import jax.numpy as jnp
import numpy as np

from grid import face_counts
from heat_operator import apply_A, diag_A

# Placeholder parameters (not a material choice), dt = 0.1 s for tests
dx, dt, k, rho, c, h = 1e-3, 0.1, 20.0, 8000.0, 500.0, 10.0
C_dt = rho * c * dx**3 / dt   # 0.04 W/K
G = k * dx                    # 0.02 W/K
H = h * dx**2                 # 1e-5 W/K


def random_mask(shape, seed=0):
    return jnp.asarray(np.random.default_rng(seed).random(shape) < 0.5)


def A(T, active, H=H):
    n_act, n_exp = face_counts(active)
    return apply_A(T, active, n_act, n_exp, C_dt, G, H)


def test_two_cells_by_hand():
    # 1x1x2, both active: each has 1 active neighbour and 5 exposed faces.
    # A = [[C_dt + G + 5H, -G], [-G, C_dt + G + 5H]]
    active = jnp.ones((1, 1, 2), dtype=bool)
    T = jnp.array([1.0, 0.0]).reshape(1, 1, 2)
    out = A(T, active).ravel()
    assert jnp.allclose(out, jnp.array([C_dt + G + 5 * H, -G]), rtol=1e-14)


def test_constant_field_no_conduction():
    # Uniform T: conduction cancels; with h = 0 only the storage term is left.
    active = random_mask((4, 5, 6))
    T = jnp.full((4, 5, 6), 700.0)
    out = A(T, active, H=0.0)
    assert jnp.allclose(out[active], C_dt * 700.0, rtol=1e-14)


def test_inactive_cells_unchanged():
    active = random_mask((4, 5, 6))
    T = jnp.asarray(np.random.default_rng(1).random((4, 5, 6)))
    out = A(T, active)
    assert jnp.array_equal(out[~active], T[~active])


def test_diagonal_matches_unit_vector():
    active = random_mask((4, 5, 6))
    n_act, n_exp = face_counts(active)
    d = diag_A(active, n_act, n_exp, C_dt, G, H)
    # one active and one inactive cell
    cells = [tuple(np.argwhere(np.asarray(active))[0]),
             tuple(np.argwhere(~np.asarray(active))[0])]
    for cell in cells:
        e = jnp.zeros((4, 5, 6)).at[cell].set(1.0)
        assert jnp.isclose(A(e, active)[cell], d[cell], rtol=1e-14)


def test_symmetric_positive():
    active = random_mask((4, 5, 6))
    rng = np.random.default_rng(2)
    x = jnp.asarray(rng.standard_normal((4, 5, 6)))
    y = jnp.asarray(rng.standard_normal((4, 5, 6)))
    xAy = jnp.vdot(x, A(y, active))
    Axy = jnp.vdot(A(x, active), y)
    assert abs(xAy - Axy) <= 1e-12 * abs(xAy)
    assert jnp.vdot(x, A(x, active)) > 0
