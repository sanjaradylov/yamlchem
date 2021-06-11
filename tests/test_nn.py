"""Test graph blocks and models.
"""

import dgl
import mxnet as mx

from yamlchem.data.loader import BatchifyGraph
from yamlchem.data.sets import ESOLDataset
from yamlchem.data.splitter import train_test_split
from yamlchem.nn.block.graph import (GCN, NeuralFPs, NodeGNNPredictor,
                                     WeightSum)
from yamlchem.nn.model.graph import train_gnn_predictor


def test_convolution_block():
  """Test GCN and NeuralFPs.
  """
  graph = dgl.graph(([0, 1, 2], [1, 3, 0]))
  n_nodes = graph.number_of_nodes()
  features = mx.nd.random.uniform(shape=(n_nodes, 8), dtype='float32')
  graph.ndata['h'] = features

  output_dim = 4
  block = GCN(features.shape[1], n_layers=1, hidden_dim=output_dim)
  block.initialize()
  h = block(graph, features)

  assert h.shape == (n_nodes, output_dim)

  block = NeuralFPs(features.shape[1], n_layers=1, hidden_dim=output_dim)
  block.initialize()
  h = block(graph, features)

  assert h.shape == (n_nodes, output_dim)


def test_gcn_model(capsys):
  """Test GCN model (verbosity level = 5 epochs).
  """
  batch_size, n_epochs, valid_ratio, lr, verbose = 32, 40, 0.1, 0.01, 10
  data = ESOLDataset()
  train_data, valid_data = train_test_split(data, valid_ratio, True, False)
  batchify_fn = BatchifyGraph(labeled=True, masked=False)
  dataloader = mx.gluon.data.DataLoader(
      train_data, batch_size=batch_size, last_batch='rollover', shuffle=True,
      batchify_fn=batchify_fn)
  valid_dataloader = mx.gluon.data.DataLoader(
      valid_data, batch_size=batch_size, batchify_fn=batchify_fn)
  loss_fn = mx.gluon.loss.L2Loss(prefix='MSE')
  lr_scheduler = mx.lr_scheduler.FactorScheduler(len(dataloader), 0.9, lr)
  optimizer = mx.optimizer.Adam(learning_rate=lr, lr_scheduler=lr_scheduler)
  metric = mx.metric.RMSE('RMSE')

  gcn = GCN(
      feature_dim=74,
      hidden_dim=64,
      n_layers=2,
      activation='relu',
      norm='both',
      dropout=0.2,
      batchnorm=True,
      residual=True,
  )
  readout = WeightSum()
  predictor = NodeGNNPredictor(gcn, readout, 1)
  with capsys.disabled():
    batch = next(iter(dataloader))
    predictor.initialize()
    print()
    predictor.summary(batch.graph, batch.graph.ndata['h'])

    train_gnn_predictor(
        gnn=predictor, feature_name='h', dataloader=dataloader,
        loss_fn=loss_fn, n_epochs=n_epochs, optimizer=optimizer, metric=metric,
        valid_dataloader=valid_dataloader, verbose=verbose)
