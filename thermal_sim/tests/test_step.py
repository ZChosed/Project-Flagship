import jax.numpy as jnp

from grid import face_counts
from heat_operator import apply_A
from step import build_b, step

# Placeholder parameters (not a material choice)
dx, k, rho, c, h, T_inf = 1e-3, 20.0, 8000.0, 500.0, 10.0, 300.0
G = k * dx
C = rho * c * dx**3                    # J/K per cell
SHAPE = (10, 10, 10)


def hot_cube(shape=SHAPE, T_hot=1500.0):
    T = jnp.full(shape, T_inf)
    return T.at[3:7, 3:7, 3:7].set(T_hot)


def setup(shape=SHAPE):
    active = jnp.ones(shape, dtype=bool)
    n_act, n_exp = face_counts(active)
    return active, n_act, n_exp, jnp.zeros(shape)


def run(T, dt, n_steps, h=h):
    """Take n_steps; return final T and the list of CG iteration counts."""
    active, n_act, n_exp, Q = setup(T.shape)
    iters = []
    for _ in range(n_steps):
        T, it, _ = step(T, active, n_act, n_exp, C / dt, G, h * dx**2, T_inf, Q)
        iters.append(int(it))
    return T, iters


def test_equilibrium_stays_put():
    T = jnp.full(SHAPE, T_inf)
    T_new, iters = run(T, dt=0.1, n_steps=1)
    assert jnp.allclose(T_new, T_inf, rtol=1e-12)
    assert iters == [0]                # warm start is already the solution


def test_adiabatic_box():
    T0 = hot_cube()
    T, iters = run(T0, dt=0.1, n_steps=500, h=0.0)    # 50 s >> diffusion time ~20 s
    assert abs(jnp.sum(C * T) - jnp.sum(C * T0)) <= 1e-10 * jnp.sum(C * T0)
    assert jnp.max(T) - jnp.min(T) < 1e-3              # uniform ...
    assert jnp.allclose(T, jnp.mean(T0), atol=1e-3)    # ... at the volume mean
    print("adiabatic: CG iterations per step, max =", max(iters))


def test_hot_cube_with_convection():
    dt = 0.1
    active, n_act, n_exp, Q = setup()
    C_dt, H = C / dt, h * dx**2
    T = hot_cube()
    worst = 0.0
    for _ in range(200):
        T_new, it, _ = step(T, active, n_act, n_exp, C_dt, G, H, T_inf, Q)
        assert jnp.max(T_new) <= jnp.max(T) + 1e-9      # max T never rises
        # Energy balance per step, divided by dt:  sum C_dt (T+ - T) = -sum H n_exp (T+ - T_inf)
        stored = jnp.sum(C_dt * (T_new - T))
        lost = jnp.sum(H * n_exp * (T_new - T_inf))
        gross = jnp.sum(jnp.abs(C_dt * (T_new - T)))      # scale: total heat moved this step
        worst = max(worst, float(abs(stored + lost) / gross))
        T = T_new
    assert worst < 1e-8
    print("energy balance: worst relative error =", worst)


def test_time_convergence_first_order():
    # T at one cell at t = 10 s with dt, dt/2, dt/4. First order => differences halve.
    # dt = 0.4 s is not yet in the asymptotic range (ratio ~3); from 0.1 s it is (~2.06).
    cell = (5, 5, 2)                   # between the hot cube and the surface
    T_a = run(hot_cube(), dt=0.1, n_steps=100)[0][cell]
    T_b = run(hot_cube(), dt=0.05, n_steps=200)[0][cell]
    T_c = run(hot_cube(), dt=0.025, n_steps=400)[0][cell]
    ratio = (T_a - T_b) / (T_b - T_c)
    print("time convergence ratio =", float(ratio))
    assert 1.8 < ratio < 2.2
