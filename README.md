# YAMLChem
![yamlchem](https://github.com/sanjaradylov/yamlchem/actions/workflows/package.yml/badge.svg)
[![PythonVersion](https://img.shields.io/badge/python-3.8-blue)](https://www.python.org/downloads/release/python-388/)

**yamlchem** is a Python package for applying machine learning methods in
computational chemistry.

## Example Usage

```python
>>> import mxnet as mx
>>> import yamlchem as yc
>>> batch_size, n_epochs, valid_ratio, lr, verbose = 32, 40, 0.1, 0.01, 10
>>> dataset = yc.data.sets.ESOLDataset(force_reload=True)
>>> train_data, valid_data = yc.data.splitter.train_test_split(
...     dataset, valid_ratio, True, False)
>>> batchify_fn = yc.data.loader.BatchifyGraph(labeled=True, masked=False)
>>> dataloader = mx.gluon.data.DataLoader(
...     train_data, batch_size=batch_size, last_batch='rollover',
...     shuffle=True, batchify_fn=batchify_fn)
>>> valid_dataloader = mx.gluon.data.DataLoader(
...     valid_data, batch_size=batch_size, batchify_fn=batchify_fn)
>>> loss_fn = mx.gluon.loss.L2Loss(prefix='MSE')
>>> lr_scheduler = mx.lr_scheduler.FactorScheduler(len(dataloader), 0.9, lr)
>>> optimizer = mx.optimizer.Adam(
...     learning_rate=lr, lr_scheduler=lr_scheduler)
>>> metric = mx.metric.RMSE(name='RMSE')
>>> gcn = yc.nn.block.graph.GCN(yc.feature.graph.N_DEFAULT_ATOM_FEATURES)
>>> readout = yc.nn.block.graph.WeightSum()
>>> predictor = yc.nn.block.graph.NodeGNNPredictor(gcn, readout, 1)
>>> yc.nn.model.train_gnn_predictor(
...     gnn=predictor, feature_name='h', dataloader=dataloader,
...     loss_fn=loss_fn, n_epochs=n_epochs, optimizer=optimizer,
...     metric=metric, valid_dataloader=valid_dataloader, verbose=verbose)
Using backend: mxnet
Epoch: 10, Time: 0:00:02, T.MSE: 0.409, V.MSE: 0.313, T.RMSE: 0.918, V.RMSE: 0.706
Epoch: 20, Time: 0:00:02, T.MSE: 0.239, V.MSE: 0.191, T.RMSE: 0.656, V.RMSE: 0.627
Epoch: 30, Time: 0:00:02, T.MSE: 0.198, V.MSE: 0.280, T.RMSE: 0.606, V.RMSE: 0.732
Epoch: 40, Time: 0:00:02, T.MSE: 0.126, V.MSE: 0.156, T.RMSE: 0.457, V.RMSE: 0.627
```

## Installation

### Dependencies

yamlchem requires [MXNet](mxnet.apache.org) 1.7+ for deep learning,
[DGL](dgl.ai) 0.6+ for graph processing, and [RDKit](rdkit.org) for chemistry.

### Conda Installation

Create a conda environment `yamlchem` and install dependencies:
```bash
conda env create -f environment.yml
```
Install the package:
```bash
pip install git+https://github.com/sanjaradylov/yamlchem.git
```