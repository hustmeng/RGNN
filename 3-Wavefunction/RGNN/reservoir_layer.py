#Reference:

# [1] D. Pfau, J.S. Spencer, A.G.D.G. Matthews, W.M.C. Foulkes, Ab initio solution of the many-electron Schrödinger equation with deep neural networks, Phys. Rev. Res. 2(3) (2020) 033429.

#This code is extended from https://github.com/google-deepmind/ferminet.git, which has the Apache License, Version 2.0, January 2004.

# Thanks to the original authors for their generous sharing. If you need to use the code, please refer to the original version of the code and literature.

"""Neural network building blocks."""

import functools
import itertools
from typing import MutableMapping, Optional, Sequence, Tuple
import random
import chex
import jax
import jax.numpy as jnp
from absl import logging
import numpy as np
import random
from RGNN import base_config


def quantization(embedding, gweights=None, key=None, nbit=8):
    """
    This function performs quantization of the input embedding and simulates the
    hardware noise during the vector-matrix multiplication.
    
    Args:
        embedding: The input embedding matrix to be quantized.
        gweights: The random weights from the resistive memory array. If not provided,
                  no hardware noise will be simulated.
        key: The random seed used to generate noise. If not provided, no hardware noise
             will be simulated.
        nbit: The number of bits used in the quantization process. Default is 8.

    Returns:
        z: The quantized embedding matrix after simulating hardware noise (if applicable).
    """

    # Load the default configuration for the noise level
    cfg = base_config.default()

    # Compute the number of quantization levels based on nbit
    nsplit = (2**nbit - 1)
    
    # Find the minimum and maximum values of the input embedding
    min_val = jnp.min(embedding)
    max_val = jnp.max(embedding)

    # Normalize the input embedding to the range [0, nsplit]
    new_embedding = ((embedding - min_val) / (max_val - min_val) * nsplit)

    # If gweights and key are provided, simulate the hardware noise
    if gweights is not None and key is not None:
        noise = jnp.abs(gweights) * jax.random.normal(key, shape=gweights.shape) * cfg.noise_level
        gweights = gweights + noise

    # Perform the quantized vector-matrix multiplication
    z = jnp.dot(new_embedding, gweights)

    # Compute the bias term
    bias = jnp.sum(min_val * gweights, axis=0)

    # De-normalize the result and add the bias term
    z = z / nsplit * (max_val - min_val) + bias

    return z

def reservoir_layer(
        x: jnp.ndarray,
        key,
        w: jnp.ndarray,
        b: Optional[jnp.ndarray] = None,
) -> jnp.ndarray:
    """
    This function computes the reservoir layer using the random resistive memory array.

    Args:
        x: The input matrix.
        key: The random seed used to generate noise in the quantization process.
        w: The random weights from the resistive memory array.
        b: Optional bias term. If provided, it will be added to the output.

    Returns:
        y: The output of the reservoir layer after applying quantization and adding the bias (if provided).
    """

    # Compute the reservoir layer output using the quantization function
    y = quantization(x, w[0], key)

    # Add the bias term (if provided) and return the result
    return y + b if b is not None else y

vmap_reservoir_layer = jax.vmap(reservoir_layer, in_axes=(0, None, None,None), out_axes=0)

def array_partitions(sizes: Sequence[int]) -> Sequence[int]:
  return list(itertools.accumulate(sizes))[:-1]


def split_into_blocks(block_arr: jnp.ndarray,
                      block_dims: Tuple[int, ...]) -> Sequence[jnp.ndarray]:
  partitions = array_partitions(block_dims)
  block1 = jnp.split(block_arr, partitions, axis=0)
  block12 = [jnp.split(arr, partitions, axis=1) for arr in block1]
  return tuple(itertools.chain.from_iterable(block12))


def init_linear_layer(
    key: chex.PRNGKey, in_dim: int, out_dim: int, include_bias: bool = False
) -> MutableMapping[str, jnp.ndarray]:
  
  key1, key2 = jax.random.split(key)
  weight = (
      jax.random.normal(key1, shape=(in_dim, out_dim)) /
      jnp.sqrt(float(in_dim)))
  if include_bias:
    bias = jax.random.normal(key2, shape=(out_dim,))
    return {'w': weight, 'b': bias}
  else:
    return {'w': weight}


def linear_layer(x: jnp.ndarray,
                 w: jnp.ndarray,
                 b: Optional[jnp.ndarray] = None) -> jnp.ndarray:
  #Evaluates a linear layer, x w + b.

  y = jnp.dot(x, w)
  return y + b if b is not None else y

vmap_linear_layer = jax.vmap(linear_layer, in_axes=(0, None, None), out_axes=0)

def int_to_binary(arr, nbits=24):
    arr_uint16 = arr.astype(jnp.uint32)
    arr_uint8 = jnp.right_shift(arr_uint16.reshape(-1, 1), jnp.arange(nbits - 1, -1, -1, dtype=jnp.uint32))
    binary_array = jnp.bitwise_and(arr_uint8, 1).astype(jnp.uint8)
    return binary_array.reshape(*arr.shape, nbits)

def last_dimension_slice(arr: jnp.ndarray,i) -> jnp.ndarray:
    return arr[..., i]

def jnp_flatten(embedding, start_dim=0, end_dim=-2):
    shape = jnp.shape(embedding)
    keep_dim_range = list(range(start_dim)) + list(range(end_dim+1, len(shape)))
    flatten_dim_range = list(range(start_dim, end_dim+1))
    flattened_embedding = jnp.reshape(embedding, (-1,) + shape[end_dim+1:])
    flattened_embedding = jnp.transpose(flattened_embedding, keep_dim_range + flatten_dim_range)
    return flattened_embedding


def slogdet(x):
  if x.shape[-1] == 1:
    if x.dtype == jnp.complex64 or x.dtype == jnp.complex128:
      sign = x[..., 0, 0] / jnp.abs(x[..., 0, 0])
    else:
      sign = jnp.sign(x[..., 0, 0])
    logdet = jnp.log(jnp.abs(x[..., 0, 0]))
  else:
    sign, logdet = jnp.linalg.slogdet(x)

  return sign, logdet


def logdet_matmul(
    xs: Sequence[jnp.ndarray], w: Optional[jnp.ndarray] = None
) -> Tuple[jnp.ndarray, jnp.ndarray]:
  #Combines determinants and takes dot product with weights in log-domain.

  det1d = functools.reduce(lambda a, b: a * b,
                           [x.reshape(-1) for x in xs if x.shape[-1] == 1], 1)
  # Pass initial value to functools so sign_in = 1, logdet = 0 if all matrices
  # are 1x1.
  phase_in, logdet = functools.reduce(
      lambda a, b: (a[0] * b[0], a[1] + b[1]),
      [slogdet(x) for x in xs if x.shape[-1] > 1], (1, 0))

  # log-sum-exp trick
  maxlogdet = jnp.max(logdet)
  det = phase_in * det1d * jnp.exp(logdet - maxlogdet)
  if w is None:
    result = jnp.sum(det)
  else:
    result = jnp.matmul(det, w)[0]
  # return phase as a unit-norm complex number, rather than as an angle
  if result.dtype == jnp.complex64 or result.dtype == jnp.complex128:
    phase_out = jnp.angle(result)  
  else:
    phase_out = jnp.sign(result)
  log_out = jnp.log(jnp.abs(result)) + maxlogdet
  return phase_out, log_out
