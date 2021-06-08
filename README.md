# YAMLChem
![yamlchem](https://github.com/sanjaradylov/yamlchem/actions/workflows/package.yml/badge.svg)
[![PythonVersion](https://img.shields.io/badge/python-3.8-blue)](https://www.python.org/downloads/release/python-388/)

**yamlchem** is a Python package for applying machine learning methods in
computational chemistry.

## Example Usage

```python
>>> import mxnet as mx
>>> import yamlchem as yc
>>> dataset = yc.data.sets.ESOLDataset(force_reload=True)
>>> batchify_fn = yc.data.loader.BatchifyGraph(labeled=True, masked=False)
>>> dataloader = mx.gluon.data.DataLoader(
...     dataset, batch_size=32, last_batch='rollover', shuffle=True,
...     batchify_fn=batchify_fn)
>>> loss_fn = mx.gluon.loss.L2Loss(prefix='MSE')
>>> lr_scheduler = mx.lr_scheduler.FactorScheduler(len(dataloader), 0.9, 0.01)
>>> optimizer = mx.optimizer.Adam(learning_rate=0.01, lr_scheduler=lr_scheduler)
>>> metric = mx.metric.RMSE(name='RMSE')
>>> gcn = yc.nn.block.graph.GCN(
...     yc.feature.graph.N_DEFAULT_ATOM_FEATURES, hidden_dim=64, n_layers=2,
...     activation='relu', norm='both', dropout=0.2, batchnorm=True, residual=True)
>>> readout = yc.nn.block.graph.WeightSum()
>>> predictor = yc.nn.block.graph.NodeGNNPredictor(gcn, readout, 1)
>>> yc.nn.model.train_gnn_predictor(
...     gnn=predictor, feature_name='h', dataloader=dataloader, loss_fn=loss_fn,
...     n_epochs=50, optimizer=optimizer, metric=metric, validation_fraction=0.2,
...     verbose=10)
Using backend: mxnet
Epoch: 10, Time: 0:00:02, T.MSE: 0.409, V.MSE: 0.313, T.RMSE: 0.918, V.RMSE: 0.706
Epoch: 20, Time: 0:00:02, T.MSE: 0.239, V.MSE: 0.191, T.RMSE: 0.656, V.RMSE: 0.627
Epoch: 30, Time: 0:00:02, T.MSE: 0.198, V.MSE: 0.280, T.RMSE: 0.606, V.RMSE: 0.732
Epoch: 40, Time: 0:00:02, T.MSE: 0.126, V.MSE: 0.156, T.RMSE: 0.457, V.RMSE: 0.627
Epoch: 50, Time: 0:00:02, T.MSE: 0.129, V.MSE: 0.098, T.RMSE: 0.509, V.RMSE: 0.406
```

## Installation

---

### Dependencies

yamlchem requires MXNet 1.7+ for deep learning, DGL 0.6+ for graph processing,
and RDKit for chemistry.

### Conda Installation

Create a conda environment `yamlchem` and install dependencies:
```bash
conda env create -f environment.yml
```
Install the package:
```bash
pip install git+https://github.com/sanjaradylov/yamlchem.git
```