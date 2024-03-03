# RGNN
Resistive memory-based reservoir graph neural network simulater.

(The current version is for reviewers only, and the article will be further revised and improved before publication. If you have any questions or disputes, or find any issues with the code, please do not hesitate to inform the author (mengxu@hku.hk). Thank you very much!)


References:

1. C.W. Park, M. Kornbluth, J. Vandermause, C. Wolverton, B. Kozinsky, J.P. Mailoa, Accurate and scalable graph neural network force field and molecular dynamics with direct force architecture, npj Comput. Mater. 7(1) (2021) 73. 
https://github.com/ken2403/gnnff.git

2. H. Li, Z. Wang, N. Zou, M. Ye, R. Xu, X. Gong, W. Duan, Y. Xu, Deep-learning density functional theory Hamiltonian for efficient ab initio electronic-structure calculation, Nat. Comput. Sci. 2(6) (2022) 367-377.
https://github.com/mzjb/DeepH-pack.git

3. D. Pfau, J.S. Spencer, A.G.D.G. Matthews, W.M.C. Foulkes, Ab initio solution of the many-electron Schrödinger equation with deep neural networks, Phys. Rev. Res. 2(3) (2020) 033429.
https://github.com/google-deepmind/ferminet.git


**All the code is extended based on the above references, with the primary goal of deploying these models onto hardware systems based on resistive memory chips. Thanks to the original authors for their generous sharing. If you need to use the codes, please refer to the original version of the code and literature.**

The Resistor Memory-based Graph Neural Network (RGNN) is designed for modeling ion and electron interaction systems. It is evaluated in computational tasks such as atomic force, Hamiltonian, and ground state many-body wave function calculations. Building on the literature of GNNs, the programming randomness of resistive memory is utilized, employing reservoir MP layers to update node and edge embeddings. In the reservoir MP layers, matrix-vector multiplications are performed using random weights derived from resistive memory.

In terms of software implementation, we provide conductivity data from resistive memory chips, which can be used for training and testing the RGNN model. During the training process, the parameters of the reservoir MP layers do not require training, and only a small fraction of weights (<10%) need to be optimized. Therefore, compared to conventional GNNs, the training complexity is significantly reduced.

In accordance with the characteristics of hybrid analogue-digital systems, we added a quantization module to convert analog data into multi-bit binary vectors. Moreover, based on the conductance fluctuations of resistive memory chips, we set corresponding Gaussian noise, which was added to the random weights of reservoir MP layers for testing.

The code provided here offers a comprehensive resistance memory-based simulator, where the multibit vector-matrix multiplication of reservoir MP layers can be implemented using resistive memory chips.

Data set is available at https://figshare.com/articles/dataset/Efficient_modelling_of_ionic_and_electronic_interactions_by_resistive_memory-based_reservoir_graph_neural_network/25330930

Atomic force: We have provided a simple way to train the RGNN and evaluate the model. You can configure the PyTorch environment by referring to the requirements in Reference 1 (GNNFF). We have prepared the graph data from the amorphous LiPO and you can start the model training directly according to the guidance. Finally, the trained model and the ASE software package can be used to simulate molecular dynamics.

Hamiltonian: We have provided a simple way to train and evaluate the RGNN. You can configure the PyTorch environment by referring to the requirements in Reference 2 (DeepH). We have prepared the graph data from the graphene dataset, and you can start the model training directly according to the guidance. Finally, the trained model and the OpenMX software package can be used to calculate the energy band.

Wavefunction: We have provided a simple way to train and evaluate the RGNN. You can configure the JAX environment by referring to the requirements in Reference 3 (Ferminet). Our experiment is based on the H2 molecule.

**More details are in each task directory.**

Thanks to the original authors for their generous sharing. If you need to use the code, please refer to the original version of the code and literature.
