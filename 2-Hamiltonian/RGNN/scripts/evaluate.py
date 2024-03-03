#References：
#1. H. Li, Z. Wang, N. Zou, M. Ye, R. Xu, X. Gong, W. Duan, Y. Xu, Deep-learning density functional theory Hamiltonian for efficient ab initio electronic-structure calculation, Nat. Comput. Sci. 2(6) (2022) 367-377.

#This code is extended from https://github.com/mzjb/DeepH-pack.git, which has the GNU LESSER GENERAL PUBLIC LICENSE. 

# Thanks to the original authors for their generous sharing. If you need to use the code, please refer to the original version of the code and literature.

import csv
import os
import argparse
import time
import warnings
from configparser import ConfigParser

import numpy as np
import torch
import math
from pymatgen.core.structure import Structure

from RGNN import get_graph, DeepHKernel, collate_fn

def get_subdirectories(path):
        subdirectories = []
        for entry in os.scandir(path):
            if entry.is_dir():
               subdirectories.append(entry.path)
        return subdirectories


def main():
    parser = argparse.ArgumentParser(description='Predict Hamiltonian')
    parser.add_argument('--trained_model_dir', type=str,
                        help='path of trained model')
    parser.add_argument('--input_dir', type=str,
                        help='')
    parser.add_argument('--output_dir', type=str,
                        help='')
    parser.add_argument('--conductance_dir', type=str,
                        help='')
    parser.add_argument('--disable_cuda', action='store_true', help='Disable CUDA')
    parser.add_argument('--save_csv', action='store_true', help='Save the result for each edge in csv format')
    parser.add_argument(
        '--interface',
        type=str,
        default='h5',
        choices=['h5', 'npz'])
    parser.add_argument('--huge_structure', type=bool, default=False, help='')
    args = parser.parse_args()

    old_version = False
    assert os.path.exists(os.path.join(args.trained_model_dir, 'config.ini'))
    if os.path.exists(os.path.join(args.trained_model_dir, 'best_model.pt')) is False:
        old_version = True
        assert os.path.exists(os.path.join(args.trained_model_dir, 'best_model.pkl'))
        assert os.path.exists(os.path.join(args.trained_model_dir, 'src'))

    os.makedirs(args.output_dir, exist_ok=True)

    config = ConfigParser()
    #config.read(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'default.ini'))
    config.read(os.path.join(args.trained_model_dir, 'config.ini'))
    config.set('basic', 'save_dir', os.path.join(args.output_dir))
    config.set('basic', 'disable_cuda', str(args.disable_cuda))
    config.set('basic', 'save_to_time_folder', 'False')
    config.set('basic', 'tb_writer', 'False')
    config.set('train', 'pretrained', '')
    config.set('train', 'resume', '')
    
    kernel = DeepHKernel(config)

    if old_version is False:
        checkpoint = kernel.build_model(args.trained_model_dir, old_version)
    g_weights_dir = os.path.join(args.conductance_dir, 'Conductance.pkl')
    g_weights = torch.load(g_weights_dir,map_location=kernel.device)

    sum_mae, sum_mse  = 0, 0
    with torch.no_grad():
      dirs = get_subdirectories(args.input_dir)
      for input_dir in dirs:
        structure = Structure(np.loadtxt(os.path.join(input_dir, 'lat.dat')).T,
                              np.loadtxt(os.path.join(input_dir, 'element.dat')),
                              np.loadtxt(os.path.join(input_dir, 'site_positions.dat')).T,
                              coords_are_cartesian=True,
                              to_unit_cell=False)
        cart_coords = torch.tensor(structure.cart_coords, dtype=torch.get_default_dtype())
        frac_coords = torch.tensor(structure.frac_coords, dtype=torch.get_default_dtype())
        numbers = kernel.Z_to_index[torch.tensor(structure.atomic_numbers)]
        structure.lattice.matrix.setflags(write=True)
        lattice = torch.tensor(structure.lattice.matrix, dtype=torch.get_default_dtype())
        inv_lattice = torch.inverse(lattice)

        if os.path.exists(os.path.join(input_dir, 'graph.pkl')):
            data = torch.load(os.path.join(input_dir, 'graph.pkl'))
            print(f"Load processed graph from {os.path.join(input_dir, 'graph.pkl')}")
        else:
            begin = time.time()
            data = get_graph(cart_coords, frac_coords, numbers, 0,
                             r=kernel.config.getfloat('graph', 'radius'),
                             max_num_nbr=kernel.config.getint('graph', 'max_num_nbr'),
                             numerical_tol=1e-8, lattice=lattice, default_dtype_torch=torch.get_default_dtype(),
                             tb_folder=input_dir, interface=args.interface,
                             num_l=kernel.config.getint('network', 'num_l'),
                             create_from_DFT=kernel.config.getboolean('graph', 'create_from_DFT', fallback=True),
                             if_lcmp_graph=kernel.config.getboolean('graph', 'if_lcmp_graph', fallback=True),
                             separate_onsite=kernel.separate_onsite,
                             target=kernel.config.get('basic', 'target'), huge_structure=args.huge_structure)
            torch.save(data, os.path.join(input_dir, 'graph.pkl'))
            print(f"Save processed graph to {os.path.join(input_dir, 'graph.pkl')}, cost {time.time() - begin} seconds")
        data = data.cpu()
        dataset_mask = kernel.make_mask([data])
        batch, subgraph = collate_fn(dataset_mask)
        sub_atom_idx, sub_edge_idx, sub_edge_ang, sub_index = subgraph
        


        output = kernel.model(batch.x.to(kernel.device), batch.edge_index.to(kernel.device),
                              batch.edge_attr.to(kernel.device),
                              batch.batch.to(kernel.device),
                              sub_atom_idx.to(kernel.device), sub_edge_idx.to(kernel.device),
                              sub_edge_ang.to(kernel.device), sub_index.to(kernel.device),
                              huge_structure=args.huge_structure, g_weights=g_weights)

        label = batch.label.cpu()
        mask = batch.mask.cpu()
        output = output.cpu().reshape(label.shape)
        
        assert label.shape == output.shape == mask.shape
        mse = torch.pow((label - output)*1000, 2)
        mae = torch.abs((label - output)*1000)
        print(output.shape)
        mses,maes = [],[]
        for index_orb, orbital_single in enumerate(kernel.orbital):
            if index_orb != 0:
                print('================================================================')
            print('orbital:', orbital_single)
            if kernel.spinful == False:
                mses.append(torch.masked_select(mse[:, index_orb], mask[:, index_orb]).mean().item())
                maes.append(torch.masked_select(mae[:, index_orb], mask[:, index_orb]).mean().item())

                print(f'mse: {torch.masked_select(mse[:, index_orb], mask[:, index_orb]).mean().item()}, '
                      f'mae: {torch.masked_select(mae[:, index_orb], mask[:, index_orb]).mean().item()}')
            else:
                for index_soc, str_soc in enumerate([
                    'left_up_real', 'left_up_imag', 'right_down_real', 'right_down_imag',
                    'right_up_real', 'right_up_imag', 'left_down_real', 'left_down_imag',
                ]):
                    if index_soc != 0:
                        print('----------------------------------------------------------------')
                    print(str_soc, ':')
                    index_out = index_orb * 8 + index_soc
                    print(f'mse: {torch.masked_select(mse[:, index_out], mask[:, index_out]).mean().item()}, '
                          f'mae: {torch.masked_select(mae[:, index_out], mask[:, index_out]).mean().item()}')
        sum_mse = np.array(mses) + sum_mse
        sum_mae = np.array(maes) + sum_mae
        
        if args.save_csv:
            edge_stru_index = torch.squeeze(batch.batch[batch.edge_index[0]]).numpy()
            edge_slices = torch.tensor(batch.__slices__['x'])[edge_stru_index].view(-1, 1)
            atom_ids = torch.squeeze(batch.edge_index.T - edge_slices).tolist()
            atomic_numbers = torch.squeeze(kernel.index_to_Z[batch.x[batch.edge_index.T]]).tolist()
            edge_infos = torch.squeeze(batch.edge_attr[:, :7].detach().cpu()).tolist()

            with open(os.path.join(kernel.config.get('basic', 'save_dir'), 'error_distance.csv'), 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['index', 'atom_id', 'atomic_number', 'dist', 'atom1_x', 'atom1_y', 'atom1_z',
                                 'atom2_x', 'atom2_y', 'atom2_z']
                                + ['target'] * kernel.out_fea_len + ['pred'] * kernel.out_fea_len + [
                                    'mask'] * kernel.out_fea_len)
                for index_edge in range(batch.edge_attr.shape[0]):
                    writer.writerow([
                        index_edge,
                        atom_ids[index_edge],
                        atomic_numbers[index_edge],
                        *(edge_infos[index_edge]),
                        *(label[index_edge].tolist()),
                        *(output[index_edge].tolist()),
                        *(mask[index_edge].tolist()),
                    ])

    mean_mae = sum_mae / len(dirs) #[i/len(dirs) for i in sum_mae]
    mean_mse = sum_mse / len(dirs) #[i/len(dirs) for i in sum_mse]
    x = int(math.sqrt(mean_mae.shape[0]))
    mean_mae = mean_mae.reshape((x,x))
    mean_mse = mean_mse.reshape((x,x))
    mean_mse = np.sqrt( mean_mse) 
    local_h = torch.sum(batch.label,dim = 0).cpu().numpy().reshape(x,x) / batch.label.shape[0]
    np.savetxt("local_h.txt",local_h)
    
    with open(os.path.join(args.output_dir, "mae_mse.dat"), 'w', newline='') as f: 
      print("Average mae (meV):",np.mean(mean_mae),file=f)
      
      for i in mean_mae:
        for j in i:
            
            print(j, end=' ',file=f)  
        print(file=f) 
       
      print("Average mse (meV):",np.mean(mean_mse),file=f)
      for i in mean_mse:
        for j in i:
            print(j, end=' ',file=f)  
        print(file=f)
    
    with open(os.path.join(args.output_dir, "mae.dat"), 'w', newline='') as f:
         mean_mae = mean_mae.reshape(-1)
         for i in mean_mae:
            print(i,file=f)
    print("Max mae (meV):", np.max(mean_mae))
    print("Mean mae (meV):", np.mean(mean_mae))
    print("Mean rmse (meV)", np.mean(mean_mse))
if __name__ == '__main__':
    main()
