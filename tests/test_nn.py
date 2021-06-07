"""Test graph blocks and models.
"""

import dgl
import mxnet as mx

from yamlchem.data.loader import BatchifyGraph
from yamlchem.data.sets import ESOLDataset
from yamlchem.nn.block.graph import (GCN, NeuralFPs, NodeGNNPredictor,
                                     StandardReadout)
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
  batch_size, n_epochs, verbose = 32, 99, 5
  data = ESOLDataset()
  batchify_fn = BatchifyGraph(labeled=True, masked=False)
  loader = mx.gluon.data.DataLoader(
      data, batch_size=batch_size, last_batch='rollover', shuffle=True,
      batchify_fn=batchify_fn)
  loss_fn = mx.gluon.loss.L2Loss(prefix='MSE')
  lr_scheduler = mx.lr_scheduler.FactorScheduler(len(loader), 0.5, 5e-4)
  optimizer = mx.optimizer.Adam(learning_rate=5e-3, lr_scheduler=lr_scheduler)
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
  pool = StandardReadout('mean')
  model = NodeGNNPredictor(gcn, pool, 1)
  with capsys.disabled():
    batch = next(iter(loader))
    model.initialize()
    print()
    model.summary(batch.graph, batch.graph.ndata['h'])

    train_gnn_predictor(model, 'h', loader, loss_fn, n_epochs, optimizer,
                        metric=metric, verbose=verbose)
