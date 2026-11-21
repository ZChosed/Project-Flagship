import jax.numpy as jnp
import numpy as np

from pcg import pcg


def test_pcg_matches_dense_solve():
    # Random SPD matrix (as a function), compared with numpy's direct solve.
    rng = np.random.default_rng(0)
    n = 30
    M = rng.standard_normal((n, n))
    A = jnp.asarray(M @ M.T + n * np.eye(n))
    b = jnp.asarray(rng.standard_normal(n))
    x, iters, relres = pcg(lambda v: A @ v, b, jnp.zeros(n), jnp.diag(A), tol=1e-12)
    assert jnp.allclose(x, np.linalg.solve(A, b), rtol=1e-9)
    assert relres <= 1e-12
    assert 0 < iters <= n            # exact arithmetic: CG finishes in at most n steps


def test_pcg_zero_iterations_if_already_solved():
    A = jnp.diag(jnp.arange(1.0, 6.0))
    x_true = jnp.ones(5)
    _, iters, _ = pcg(lambda v: A @ v, A @ x_true, x_true, jnp.diag(A))
    assert iters == 0
