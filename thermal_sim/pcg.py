"""Jacobi-preconditioned conjugate gradient, matrix-free, with iteration count.

Solves A x = b for symmetric positive-definite A (Hestenes & Stiefel 1952; Saad 2003, Alg. 9.1).
Written by hand because jax.scipy.sparse.linalg.cg does not return the iteration count.
"""
import jax
import jax.numpy as jnp


def pcg(apply_A, b, x0, diag, tol=1e-10, maxiter=500):
    """Solve A x = b.

    apply_A : function x -> A x (same shape as b)
    b, x0, diag : arrays, all the same shape; diag = diag(A) for the Jacobi preconditioner
    Stops when ||b - A x|| <= tol * ||b||, or after maxiter iterations.

    returns (x, iters, relres)   relres = ||r|| / ||b|| of the recursively updated residual
    """
    bnorm = jnp.linalg.norm(b)
    r = b - apply_A(x0)
    z = r / diag                          # z = M^-1 r  (Jacobi)
    rz = jnp.vdot(r, z)
    state = (x0, r, z, z, rz, 0)          # (x, r, z, p, r.z, k); first search direction p = z

    def not_done(s):
        _, r, _, _, _, k = s
        return (jnp.linalg.norm(r) > tol * bnorm) & (k < maxiter)

    def iterate(s):
        x, r, z, p, rz, k = s
        Ap = apply_A(p)
        alpha = rz / jnp.vdot(p, Ap)      # step length along p
        x = x + alpha * p
        r = r - alpha * Ap
        z = r / diag
        rz_new = jnp.vdot(r, z)
        p = z + (rz_new / rz) * p         # new direction, A-conjugate to the old ones
        return (x, r, z, p, rz_new, k + 1)

    x, r, _, _, _, k = jax.lax.while_loop(not_done, iterate, state)
    return x, k, jnp.linalg.norm(r) / bnorm
