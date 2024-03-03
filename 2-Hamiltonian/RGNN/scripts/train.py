#References：
#1. H. Li, Z. Wang, N. Zou, M. Ye, R. Xu, X. Gong, W. Duan, Y. Xu, Deep-learning density functional theory Hamiltonian for efficient ab initio electronic-structure calculation, Nat. Comput. Sci. 2(6) (2022) 367-377.

#This code is extended from https://github.com/mzjb/DeepH-pack.git, which has the GNU LESSER GENERAL PUBLIC LICENSE. 

# Thanks to the original authors for their generous sharing. If you need to use the code, please refer to the original version of the code and literature.

import argparse
import torch
from RGNN import DeepHKernel, get_config
import os

def main():
    parser = argparse.ArgumentParser(description='Deep Hamiltonian')
    parser.add_argument('--config', default=[], nargs='+', type=str, metavar='N')
    args = parser.parse_args()

    print(f'User config name: {args.config}')
    config = get_config(args.config)
    only_get_graph = config.getboolean('basic', 'only_get_graph')
    kernel = DeepHKernel(config)
    train_loader, val_loader, test_loader, transform = kernel.get_dataset(only_get_graph)
    if only_get_graph:
        return
    kernel.build_model()
    kernel.set_train()
    conductance_dir = config.get('basic', 'conductance_dir')
    g_weights_dir = os.path.join(conductance_dir, 'Conductance.pkl')
    g_weights = torch.load(g_weights_dir,map_location=config.get('basic', 'device'))
    kernel.train(train_loader, val_loader, test_loader, g_weights)

if __name__ == '__main__':
    main()
