# RGNN
Resistive memory-based reservoir graph neural network simulater.


References:

1. C.W. Park, M. Kornbluth, J. Vandermause, C. Wolverton, B. Kozinsky, J.P. Mailoa, Accurate and scalable graph neural network force field and molecular dynamics with direct force architecture, npj Comput. Mater. 7(1) (2021) 73. 
https://github.com/ken2403/gnnff.git

2. H. Li, Z. Wang, N. Zou, M. Ye, R. Xu, X. Gong, W. Duan, Y. Xu, Deep-learning density functional theory Hamiltonian for efficient ab initio electronic-structure calculation, Nat. Comput. Sci. 2(6) (2022) 367-377.
https://github.com/mzjb/DeepH-pack.git

3. D. Pfau, J.S. Spencer, A.G.D.G. Matthews, W.M.C. Foulkes, Ab initio solution of the many-electron Schrödinger equation with deep neural networks, Phys. Rev. Res. 2(3) (2020) 033429.
https://github.com/google-deepmind/ferminet.git

The codes are extended from above references. Thanks to the original authors for their generous sharing. If you need to use the code, please refer to the original version of the code and literature.

The resistive memory-based reservoir graph neural network (RGNN) is designed for modeling systems with ionic and electronic interactions. It is evaluated in computational tasks such as atomic forces, Hamiltonian, and ground-state many-body wavefunction calculations. Based on the references, we adopted the reservoir MP layers to update the node and edge embeddings, which could take advantage of the programming stochasticity of resistive memory. We provide the software part of the article here, as well as the conductivity data derived from the resistive memory chip, which allows for simulation of the memristor hardware. Additionally, calculations with random weights applied in this code can be integrated into the hardware system for completion. 


The graph is composed of nodes v_i representing particles (atoms or electrons) and edges e_ij representing the influence between node v_i and neighbor node v_j. The RGNN employs a reservoir computing mechanism to update node and edge features for inferring fundamental material properties. Specifically, node v_i and edge e_ij of the graph are initialized to represent the hidden physical and chemical states. Within the reservoir message-passing (MP) layers, the node embedding h_i^l and edge embedding h_((i,j))^l are aggregated (a_v^l) and passed through the fully-connected layers θ_v^l employing random weights to generate the new node embedding h_i^(l+1). Then, h_i^(l+1) and h_((i,j))^l are aggregated (a_e^l) to update the edge embedding h_((i,j))^(l+1) through random fully-connected layers θ_e^l. After a given number of iterations within the same reservoir random MP layers, random node and edge embeddings are passed to a lightweight trainable embedding layer in the digital domain. The output layers utilize the final embeddings to infer properties such as atomic forces, Hamiltonian, and wavefunction.

We have provided the conductance of the resistive memory array to implement the random weight in the reservoir MP layers. Of course it could be replaced by other conductance data or random weights of Gaussian distribution.

Atomic force: We have provided a simple way to train the RGNN and evaluate the model. You can configure the PyTorch environment by referring to the requirements in Reference 1 (GNNFF). We have prepared the graph data from the amorphous LiPO and you can start the model training directly according to the guidance. Finally, the trained model and the ASE software package can be used to simulate molecular dynamics.

Hamiltonian: We have provided a simple way to train and evaluate the RGNN. You can configure the PyTorch environment by referring to the requirements in Reference 2 (DeepH). We have prepared the graph data from the graphene dataset, and you can start the model training directly according to the guidance. Finally, the trained model and the OpenMX software package can be used to calculate the energy band.

Wavefunction: We have provided a simple way to train and evaluate the RGNN. You can configure the JAX environment by referring to the requirements in Reference 3 (Ferminet). Our experiment is based on the H2 molecule.

Thanks to the original authors for their generous sharing. If you need to use the code, please refer to the original version of the code and literature.
