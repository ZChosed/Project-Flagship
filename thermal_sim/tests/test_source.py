import jax.numpy as jnp

from source import point_source

def test_x_center():
    assert point_source((3,2,3),3,4.5,3,3612).sum() == 3612

def test_x_split():
    assert point_source((3,2,3),3,6,3,3612).sum() == 3612

def test_actual():
    dx = 1e-3
    shape = (100,100,100)
    x_s = 55.4e-3
    y_s = 50e-3
    Q = point_source(shape,dx,x_s,y_s,1000)
    assert jnp.isclose(Q.sum(), 1000)
    assert jnp.isclose(Q[54,50,-1], 100)
    assert jnp.isclose(Q[55,50,-1], 900)