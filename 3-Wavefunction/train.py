import sys
import numpy as np
from absl import logging
from RGNN.utils import system
from RGNN import base_config
from RGNN import train
import pickle
import jax
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

logging.get_absl_handler().python_handler.stream = sys.stdout
logging.set_verbosity(logging.INFO)

cfg = base_config.default()
cfg.system.electrons = (1,1)  # (alpha electrons, beta electrons)
cfg.system.molecule = [system.Atom('H', (1,  0, 0)),
                       system.Atom('H', (-0.4, 0, 0)), 
                      ]

cfg.batch_size = 1600
cfg.pretrain.iterations = 200
cfg.log.restore_path = None
cfg.optim.optimizer = "kfac"

cfg.optim.iterations = 10000
cfg.mcmc.steps = 10
cfg.mcmc.adapt_frequency = 100
cfg.log.stats_frequency = 100


params, data,_ = train.train(cfg,)# 

np.savez("params.npz",params = params)
with open('data.pkl', 'wb') as f:
    pickle.dump(data, f)
