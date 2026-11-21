import jax.numpy as jnp
import numpy as np

from grid import face_counts

def test_3x3x3():
    n_act, n_exp = face_counts(jnp.ones((3,3,3), dtype=bool))
    assert n_exp.sum() == 54
    assert n_exp[0,0,0] == 3   # corner
    assert n_exp[1,0,0] == 2   # edge
    assert n_exp[1,1,0] == 1   # face centre
    assert n_exp[1,1,1] == 0   # centre

def test_single_active_corner():
    arr = jnp.zeros((3,3,3), dtype=bool)
    arr = arr.at[2,2,2].set(True)
    n_act, n_exp = face_counts(arr)
    assert n_act[2,2,2] == 0   # the active cell has no active neighbours
    assert n_exp[2,2,2] == 6
    assert n_act.sum() == 3    # its 3 in-grid neighbours each see 1 active cell

def test_single_active_center():
    arr = jnp.zeros((3,3,3), dtype=bool)
    arr = arr.at[1,1,1].set(True)
    n_act, n_exp = face_counts(arr)
    assert n_act[1,1,1] == 0
    assert n_exp[1,1,1] == 6
    assert n_act.sum() == 6    # all 6 neighbours are in the grid

def test_random():
    mask = np.random.default_rng(0).random((4,5,6)) < 0.5
    n_act, n_exp = face_counts(mask)
    assert n_exp[~mask].sum() == 0

def test_symmetry():
    active = np.random.default_rng(0).random((4,5,6)) < 0.5
    n_act, _ = face_counts(active)                        # count the original once

    for axis in (0, 1, 2):
        n_act_flipped, _ = face_counts(jnp.flip(active, axis=axis))  # Route A: flip mask, then count
        expected = jnp.flip(n_act, axis=axis)                        # Route B: count, then flip
        assert jnp.array_equal(n_act_flipped, expected)
