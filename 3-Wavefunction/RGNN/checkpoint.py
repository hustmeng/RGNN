#Reference:

# [1] D. Pfau, J.S. Spencer, A.G.D.G. Matthews, W.M.C. Foulkes, Ab initio solution of the many-electron Schrödinger equation with deep neural networks, Phys. Rev. Res. 2(3) (2020) 033429.

#This code is extended from https://github.com/google-deepmind/ferminet.git, which has the Apache License, Version 2.0, January 2004.

# Thanks to the original authors for their generous sharing. If you need to use the code, please refer to the original version of the code and literature.

"""Super simple checkpoints using numpy."""

import dataclasses
import datetime
import os
from typing import Optional
import zipfile

from absl import logging
from RGNN import networks
import jax
import jax.numpy as jnp
import numpy as np


def find_last_checkpoint(ckpt_path: Optional[str] = None) -> Optional[str]:
  """Finds most recent valid checkpoint in a directory.

  Args:
    ckpt_path: Directory containing checkpoints.

  Returns:
    Last QMC checkpoint (ordered by sorting all checkpoints by name in reverse)
    or None if no valid checkpoint is found or ckpt_path is not given or doesn't
    exist. A checkpoint is regarded as not valid if it cannot be read
    successfully using np.load.
  """
  if ckpt_path and os.path.exists(ckpt_path):
    files = [f for f in os.listdir(ckpt_path) if 'qmcjax_ckpt_' in f]
    # Handle case where last checkpoint is corrupt/empty.
    for file in sorted(files, reverse=True):
      fname = os.path.join(ckpt_path, file)
      with open(fname, 'rb') as f:
        try:
          np.load(f, allow_pickle=True)
          return fname
        except (OSError, EOFError, zipfile.BadZipFile):
          logging.info('Error loading checkpoint %s. Trying next checkpoint...',
                       fname)
  return None


def create_save_path(save_path: Optional[str]) -> str:
  """Creates the directory for saving checkpoints, if it doesn't exist.

  Args:
    save_path: directory to use. If false, create a directory in the working
      directory based upon the current time.

  Returns:
    Path to save checkpoints to.
  """
  timestamp = datetime.datetime.now().strftime('%Y_%m_%d_%H:%M:%S')
  default_save_path = os.path.join(os.getcwd(), f'ferminet_{timestamp}')
  ckpt_save_path = save_path or default_save_path
  if ckpt_save_path and not os.path.isdir(ckpt_save_path):
    os.makedirs(ckpt_save_path)
  return ckpt_save_path


def get_restore_path(restore_path: Optional[str] = None) -> Optional[str]:
  """Gets the path containing checkpoints from a previous calculation.

  Args:
    restore_path: path to checkpoints.

  Returns:
    The path or None if restore_path is falsy.
  """
  if restore_path:
    ckpt_restore_path = restore_path
  else:
    ckpt_restore_path = None
  return ckpt_restore_path


def save(save_path: str, t: int, data, params, opt_state, mcmc_width) -> str:
  """Saves checkpoint information to a npz file.

  Args:
    save_path: path to directory to save checkpoint to. The checkpoint file is
      save_path/qmcjax_ckpt_$t.npz, where $t is the number of completed
      iterations.
    t: number of completed iterations.
    data: MCMC walker configurations.
    params: pytree of network parameters.
    opt_state: optimization state.
    mcmc_width: width to use in the MCMC proposal distribution.

  Returns:
    path to checkpoint file.
  """
  ckpt_filename = os.path.join(save_path, f'qmcjax_ckpt_{t:06d}.npz')
  logging.info('Saving checkpoint %s', ckpt_filename)
  with open(ckpt_filename, 'wb') as f:
    np.savez(
        f,
        t=t,
        data=dataclasses.asdict(data),
        params=params,
        opt_state=opt_state,
        mcmc_width=mcmc_width)
  return ckpt_filename


def restore(restore_filename: str, batch_size: Optional[int] = None):
  """Restores data saved in a checkpoint.

  Args:
    restore_filename: filename containing checkpoint.
    batch_size: total batch size to be used. If present, check the data saved in
      the checkpoint is consistent with the batch size requested for the
      calculation.

  Returns:
    (t, data, params, opt_state, mcmc_width) tuple, where
    t: number of completed iterations.
    data: MCMC walker configurations.
    params: pytree of network parameters.
    opt_state: optimization state.
    mcmc_width: width to use in the MCMC proposal distribution.

  Raises:
    ValueError: if the leading dimension of data does not match the number of
    devices (i.e. the number of devices being parallelised over has changed) or
    if the total batch size is not equal to the number of MCMC configurations in
    data.
  """
  logging.info('Loading checkpoint %s', restore_filename)
  with open(restore_filename, 'rb') as f:
    ckpt_data = np.load(f, allow_pickle=True)
    # Retrieve data from npz file. Non-array variables need to be converted back
    # to natives types using .tolist().
    t = ckpt_data['t'].tolist() + 1  # Return the iterations completed.
    data = networks.FermiNetData(**ckpt_data['data'].item())
    params = ckpt_data['params'].tolist()
    opt_state = ckpt_data['opt_state'].tolist()
    mcmc_width = jnp.array(ckpt_data['mcmc_width'].tolist())
    if data.positions.shape[0] != jax.device_count():
      raise ValueError(
          'Incorrect number of devices found. Expected'
          f' {data.positions.shape[0]}, found {jax.device_count()}.'
      )
    if (
        batch_size
        and data.positions.shape[0] * data.positions.shape[1] != batch_size
    ):
      raise ValueError(
          f'Wrong batch size in loaded data. Expected {batch_size}, found '
          f'{data.positions.shape[0] * data.positions.shape[1]}.')
  return t, data, params, opt_state, mcmc_width
