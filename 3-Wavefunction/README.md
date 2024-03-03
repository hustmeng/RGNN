References:

1. D. Pfau, J.S. Spencer, A.G.D.G. Matthews, W.M.C. Foulkes, Ab initio solution of the many-electron Schrödinger equation with deep neural networks, Phys. Rev. Res. 2(3) (2020) 033429.
https://github.com/google-deepmind/ferminet.git

2. Scherbela, M., Reisenhofer, R., Gerard, L., Marquetand, P. & Grohs, P. Solving the electronic Schrödinger equation for multiple nuclear geometries with weight-sharing deep neural networks. Nat. Comput. Sci. 2, 331-341 (2022).

3. Cassella G, Sutterud H, Azadi S, et al. Discovering quantum phase transitions with fermionic neural networks[J]. Physical review letters, 2023, 130(3): 036401.

4. Spencer J S, Pfau D, Botev A, et al. Better, faster fermionic neural networks[J]. arXiv preprint arXiv:2011.07125, 2020.

This code is extended from https://github.com/google-deepmind/ferminet.git, which has the Apache License, Version 2.0, January 2004.

Thanks to the original authors for their generous sharing. If you need to use the code, please refer to the original version of the code and literature.

We adopted the reservoir MP layers to update the node and edge embeddings, which could take advantage of the programming stochasticity of resistive memory. At the same time, we provided conductivity data from resistive memory chips, which can be used to implement random weights for reservoir MP layers, thereby facilitating the training and validation process of the RGNN model. In accordance with the characteristics of hybrid analogue-digital systems, we added a quantization module to convert analog data into multi-bit binary vectors. Moreover, based on the conductance fluctuations of resistive memory chips, we set corresponding Gaussian noise, which was added to the random weights of reservoir MP layers for testing. The code provided here offers a comprehensive resistance memory-based simulator, where the multibit vector-matrix multiplication of reservoir MP layers can be implemented using resistive memory chips. 

It is particularly worth noting that, since resistive memory-based analogue system do not support reverse gradient computation, we provide an effective differential method to calculate the Hamiltonian. 

Various hardware-related parameters can be set in the base_configure.py file. By the way, during the training process, you can try to gradually increase the noise range, which can effectively improve the model's anti-interference ability.

We have provided the conductance of the resistive memory array to implement the random weight in the reservoir MP layers. Of course it could be replaced by other conductance data or random weights of Gaussian distribution.


Example for H2：(Please note the parameters in base_configure.py)

Train: python train.py

Evalu: python evalu.py

Thanks to the original authors for their generous sharing. If you need to use the code, please refer to the original version of the code and literature.
