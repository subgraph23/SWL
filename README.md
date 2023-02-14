# SWL

The official code of the paper **[A Complete Expressiveness Hierarchy for Subgraph GNNs via Subgraph Weisfeiler-Lehman Tests]()**.

<img src="algorithm.png" alt="algorithm" style="zoom:36%;" />

## Code structure

- `result/<model>-<subset>-<seed>.txt`: result of each model. Each file is named by corresponding model string, dataset, and random seed. Row is indexed by epoch and each column is a `training_loss` - `validation_mae` - `test_mae` triple.


## History

- 02/13: initial commit.
