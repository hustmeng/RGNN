References：

H. Li, Z. Wang, N. Zou, M. Ye, R. Xu, X. Gong, W. Duan, Y. Xu, Deep-learning density functional theory Hamiltonian for efficient ab initio electronic-structure calculation, Nat. Comput. Sci. 2(6) (2022) 367-377. https://github.com/mzjb/DeepH-pack.git

H. Li, Z. Tang, X. Gong, N. Zou, W. Duan, Y. Xu, Deep-learning electronic-structure calculation of magnetic superstructures, Nat. Comput. Sci. 3(4) (2023) 321-327.

X. Gong, H. Li, N. Zou, R. Xu, W. Duan, Y. Xu, General framework for E(3)-equivariant neural network representation of density functional theory Hamiltonian, Nat. Commun. 14(1) (2023) 2848.

This code is extended from https://github.com/mzjb/DeepH-pack.git,which has the GNU LESSER GENERAL PUBLIC LICENSE.

Thanks to the original authors for their generous sharing. If you need to use the code, please refer to the original version of the code and literature.

We adopted the reservoir MP layers to update the node and edge embeddings, which could take advantage of the programming stochasticity of resistive memory. At the same time, we provided conductivity data from resistive memory chips, which can be used to implement random weights for reservoir MP layers, thereby facilitating the training and validation process of the RGNN model. In accordance with the characteristics of hybrid analogue-digital systems, we added a quantization module to convert analog data into multi-bit binary vectors. Moreover, based on the conductance fluctuations of resistive memory chips, we set corresponding Gaussian noise, which was added to the random weights of reservoir MP layers for testing. The code provided here offers a comprehensive resistance memory-based simulator, where the multibit vector-matrix multiplication of reservoir MP layers can be implemented using resistive memory chips.

To facilitate the reproduction of our expremental results, the graph dataset is transformed from the AIMD results of graphene.

The PyTorch environment for running the code can be referenced from DeepH.

Data set is available at (https://figshare.com/articles/dataset/Efficient_modelling_of_ionic_and_electronic_interactions_by_resistive_memory-based_reservoir_graph_neural_network/25330930) and https://doi.org/10.5281/zenodo.10774321
"Graph-npz-graphene-5l-6.0r0mn.pkl" is the graph data for training RGNN. "test" is for evaluating RGNN.

Examples:

Train: ./train.sh (Please set "graph_dir" and "conductance_dir" in the train.ini)

Evalu: ./eval.sh (Please set "trained_model_dir" "input_dir" "output_dir" "conductance_dir" in "eval.sh")

Finally, the trained model and the OpenMX software package can be used to calculate the energy band.

Thanks to the original authors for their generous sharing. If you need to use the code, please refer to the original version of the code and literature.
