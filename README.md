# SWL

The official code of the paper **[A Complete Expressiveness Hierarchy for Subgraph GNNs via Subgraph Weisfeiler-Lehman Tests](https://arxiv.org/pdf/2302.07090.pdf)**.

<img src="algorithm.png" alt="algorithm" style="zoom:36%;" />

## Training logs

- Each log file in the `logs` folder contains the training result of a model configuration. Each file is named by the corresponding dataset (ZINC-subset or ZINC-full), model string, batch size, distance encoding hyper-parameter, and random seed, with the form `logs/<subset>.<model>.<batch size>.<max distance>.<seed>.txt`. In each file, the i-th row contains four metrics at the i-th training epoch: learning rate, training loss, validation MAE, and test MAE.


## History

- 02/13: initial commit.
- 02/15: add training logs.