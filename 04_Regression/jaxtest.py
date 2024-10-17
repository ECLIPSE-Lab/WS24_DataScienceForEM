# %%
import jax.numpy as jnp
from jax import grad, jit, vmap
from jax import random
from jax.numpy.fft import fft2, ifft2, fftshift, ifftshift

import matplotlib.pyplot as plt
from skimage.data import astronaut

O = astronaut()[..., 0]
# %%


def plot(x):
    fig, ax = plt.subplots()
    ax.imshow(x)
    plt.tight_layout()
    plt.show()


def split(O, r, M):
    # print(r[0])
    print(r.shape)
    return O[r[0]:r[0]+M[0], r[1]:r[1]+M[1]]


r = jnp.zeros((10, 2), dtype=jnp.int64)
M = jnp.array([32, 32], dtype=jnp.int64)
Oi = split(O, r[0], M)
plot(Oi)
# %%
batched_split = vmap(split, in_axes=(None, 0, None))

Osplit = batched_split(O, r, M)

# %%
