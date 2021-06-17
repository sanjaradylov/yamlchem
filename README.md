# YAMLChem
![yamlchem](https://github.com/sanjaradylov/yamlchem/actions/workflows/package.yml/badge.svg)
[![PythonVersion](https://img.shields.io/badge/python-3.8-blue)](https://www.python.org/downloads/release/python-388/)

**yamlchem** is a Python package for applying machine learning methods in
computational chemistry.

## Example Usage

```python
>>> import mxnet as mx
>>> import yamlchem as yc
>>> batch_size, n_epochs, learning_rate = 32, 40, 0.01
>>> valid_ratio, test_ratio, feature_name = 0.1, 0.1, 'h'
>>> data = yc.data.sets.ESOLDataset(force_reload=True)
>>> train_data, valid_data, test_data = yc.data.splitter.train_valid_test_split(
...     data, valid_ratio=valid_ratio, test_ratio=test_ratio,
...     shuffle=True, use_same_dataset_class=False)
>>> batchify_fn = yc.data.loader.BatchifyGraph(labeled=True, masked=False)
>>> train_loader = mx.gluon.data.DataLoader(
...     train_data, batch_size=batch_size, last_batch='rollover',
...     shuffle=True, batchify_fn=batchify_fn)
>>> valid_loader = mx.gluon.data.DataLoader(
...     valid_data, batch_size=len(valid_data), batchify_fn=batchify_fn)
>>> test_loader = mx.gluon.data.DataLoader(
...     test_data, batch_size=len(test_data), batchify_fn=batchify_fn)
>>> loss_fn = mx.gluon.loss.L2Loss(prefix='MSE')
>>> lr_scheduler = mx.lr_scheduler.FactorScheduler(
...     len(train_loader), factor=0.95, stop_factor_lr=5e-4,
...     base_lr=learning_rate)
>>> optimizer = mx.optimizer.Adam(
...     learning_rate=learning_rate, lr_scheduler=lr_scheduler)
>>> metric = mx.metric.RMSE(name='RMSE')
>>> gcn = yc.nn.block.graph.GCN(yc.feature.graph.N_DEFAULT_ATOM_FEATURES)
>>> readout = yc.nn.block.graph.WeightSum()
>>> predictor = yc.nn.block.graph.NodeGNNPredictor(gcn, readout, 1)
>>> yc.nn.model.train_gnn_predictor(
...     gnn=predictor, feature_name=feature_name, dataloader=train_loader,
...     loss_fn=loss_fn, n_epochs=n_epochs, optimizer=optimizer,
...     metric=metric, valid_dataloader=valid_loader, verbose=10)
Using backend: mxnet
Epoch: 10, Time: 0:00:02, T.MSE: 0.409, V.MSE: 0.313, T.RMSE: 0.918, V.RMSE: 0.706
Epoch: 20, Time: 0:00:02, T.MSE: 0.239, V.MSE: 0.191, T.RMSE: 0.656, V.RMSE: 0.627
Epoch: 30, Time: 0:00:02, T.MSE: 0.198, V.MSE: 0.280, T.RMSE: 0.606, V.RMSE: 0.732
Epoch: 40, Time: 0:00:02, T.MSE: 0.126, V.MSE: 0.156, T.RMSE: 0.457, V.RMSE: 0.627
>>> test_batch = next(iter(test_loader))
>>> predictions = predictor(
...     test_batch.graph, test_batch.graph.ndata[feature_name])
>>> metric.reset()
>>> metric.update(
...     preds=predictions, labels=test_batch.label.reshape_like(predictions))
>>> print(f'Test {metric.name}: {metric.get()[1]:.3f}')
Test RMSE: 0.460
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