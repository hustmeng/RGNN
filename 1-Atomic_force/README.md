References：
1. C.W. Park, M. Kornbluth, J. Vandermause, C. Wolverton, B. Kozinsky, J.P. Mailoa, Accurate and scalable graph neural network force field and molecular dynamics with direct force architecture, npj Comput. Mater. 7(1) (2021) 73. 
https://github.com/ken2403/gnnff.git

2. Chen, C. & Ong, S. P. A universal graph deep learning interatomic potential for the periodic table. Nat. Comput. Sci. 2, 718-728 (2022).
https://github.com/materialsvirtuallab/matgl.git

3. Batzner, S. et al. E(3)-equivariant graph neural networks for data-efficient and accurate interatomic potentials. Nat. Commun. 13, 2453 (2022).

4. Musaelian, A. et al. Learning local equivariant representations for large-scale atomistic dynamics. Nat. Commun. 14, 579 (2023).

**This code is extended from https://github.com/ken2403/gnnff.git, which has the MIT License.**

**Thanks to the original authors for their generous sharing. If you need to use the code, please refer to the original version of the code and literature.**
 

We adopted the reservoir MP layers to update the node and edge embeddings, which could take advantage of the programming stochasticity of resistive memory. At the same time, we provided conductivity data from resistive memory chips, which can be used to implement random weights for reservoir MP layers, thereby facilitating the training and validation process of the RGNN model. In accordance with the characteristics of hybrid analogue-digital systems, we added a quantization module to convert analog data into multi-bit binary vectors. Moreover, based on the conductance fluctuations of resistive memory chips, we set corresponding Gaussian noise, which was added to the random weights of reservoir MP layers for testing. The code provided here offers a comprehensive resistance memory-based simulator, where the multibit vector-matrix multiplication of reservoir MP layers can be implemented using resistive memory chips. 

The PyTorch environment for running the code can be referenced from GNNFF. 

To facilitate the reproduction of our expremental results, the graph dataset is transformed from the AIMD results of amorphous LiPO.

Dataset "graph_atomic_force.pkl" is available at https://figshare.com/account/articles/25330930.

Example: (Please set the graphpath, conductance_path, checkpoint_path and modelpath in the train.json and eval.json)

Train - python main_force.py from_json work/train.json  

Eval - python main_force.py from_json work/eval.json 

Finally, the trained model and the ASE software package can be used to simulate molecular dynamics.

Thanks to the original authors for their generous sharing. If you need to use the code, please refer to the original version of the code and literature.
