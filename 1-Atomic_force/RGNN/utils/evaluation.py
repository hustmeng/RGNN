#Reference:

#1. C.W. Park, M. Kornbluth, J. Vandermause, C. Wolverton, B. Kozinsky, J.P. Mailoa, Accurate and scalable graph neural network force field and molecular dynamics with direct force architecture, npj Comput. Mater. 7(1) (2021) 73. 
#This code is extended from https://github.com/ken2403/gnnff.git, which has the MIT License.

# Thanks to the original authors for their generous sharing. If you need to use the code, please refer to the original version of the code and literature.


import os
import csv
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
__all__ = ["evaluate"]


def evaluate(
    args,
    model,
    train_loader,
    val_loader,
    test_loader,
    device,
    metrics,
    conductance
):

    forces_dft = []
    forces_dft = []
    header = []
    results = []

    eval_file = os.path.join(args.modelpath, "evaluation.txt") 
    loaders = dict(train=train_loader, validation=val_loader, test=test_loader)
    args.split = ["test"]
    for datasplit in args.split:
        
        header += [
            "{} MAE".format(datasplit, args.predict_property),
            "{} RMSE".format(datasplit, args.predict_property),
        ]
        force_dft, force_pre, atom_num = evaluate_dataset(metrics, model, loaders[datasplit], device, conductance)
        dft=force_dft.cpu().numpy().reshape(-1)
        pre=force_pre.cpu().numpy().reshape(-1)
        c = atom_num.cpu().numpy().reshape(-1)
        c = np.repeat(c, 3)


        dft_0 = dft[c == 3]
        dft_1 = dft[c == 8]
        dft_2 = dft[c == 15]

        pre_0 = pre[c == 3]
        pre_1 = pre[c == 8]
        pre_2 = pre[c == 15]

        f_mae_0 = np.sum(abs(pre_0 - dft_0)) / np.sum(abs(dft_0))
        f_mae_1 = np.sum(abs(pre_1 - dft_1)) / np.sum(abs(dft_1))
        f_mae_2 = np.sum(abs(pre_2 - dft_2)) / np.sum(abs(dft_2))
        print(f_mae_0, f_mae_1, f_mae_2)
        
        
        atom_color = np.repeat(atom_num, 3)        

        with open("output/data.txt", "w") as file:        
          for d,p,n in zip(dft,pre,atom_color):
              line = str(d)+"    "+str(p) + "    " + str(n)
              print(line,file = file)


def evaluate_dataset(metrics, model, loader, device, conductance):
    model.eval()
    results_dft = []
    results_pre = []
    results_num = []
    
    for metric in metrics:
        metric.reset()
    
    feature = []
    color = []
    i = 0
    for batch in loader:
        i = i + 1
        if i > 100:
          break
        atom_num = batch['_atomic_numbers'].view(-1)
        batch = {k: v.to(device) for k, v in batch.items()}
        result = model(batch, conductance)
        
        feature.append(np.sum(result["edge_list"][3][0],axis =1))        
        color.append( torch.sqrt(torch.sum(batch["forces"][0]**2,dim=1)).detach().cpu().numpy() )
        
        force_pre = result["forces"].view(-1)
        force_dft = batch["forces"].view(-1)
        results_pre.append(force_pre)
        results_dft.append(force_dft)
        results_num.append(atom_num)

    results_pre = torch.cat(results_pre)
    results_dft = torch.cat(results_dft)
    results_num = torch.cat(results_num) 
    return results_dft, results_pre, results_num
