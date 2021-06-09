"""Blocks for the Neural Fingerprints graph neural network model.

Classes:
  NeuralFPs: Multilayer Neural Fingerprints.
  NeuralFPsBlock: Neural Fingerprints block.
"""

__all__ = (
    'NeuralFPs',
)

from typing import Optional

import dgl
import dgl.function as fn
import dgl.nn.mxnet as dgl_mx
import mxnet as mx

from ...._types import ActivationT


class NeuralFPs(mx.gluon.Block):
  """Multilayer Neural Fingerprints.

  Args:
    feature_dim: The dimension of node feature space of the graph that will be
      processed.
    max_degree: (Optional, defaults to 10). The maximum node degree.
    n_layers: (Optional, defaults to 2). The number of layers.
    hidden_dim: (Optional, defaults to 64).
      The number of neurons in hidden and output layers.
    activation: (Optional, defaults to 'relu').
      The activation of the type supported by MXNet.
    batchnorm: (Optional, defaults to True).
      Whether to use batch normalization after dropout.
    dropout: (Optional, defaults to 0.2).
      The dropout rate.

  References:
    D. Duvenaud et al. Convolutional Networks on Graphs for Learning Molecular
    Fingerprints.
    (https://arxiv.org/abs/1609.02907)
  """
  def __init__(
      self,
      feature_dim: int,
      *,
      max_degree: int = 10,
      n_layers: int = 2,
      hidden_dim: int = 64,
      batchnorm: bool = True,
      activation: ActivationT = 'relu',
      dropout: float = 0.2,

      prefix: Optional[str] = None,
      params: Optional[mx.gluon.ParameterDict] = None,
  ):
    super().__init__(prefix=prefix, params=params)

    if n_layers == 1:
      self.blocks = NeuralFPsBlock(feature_dim,
                                   output_dim=hidden_dim,
                                   batchnorm=batchnorm,
                                   activation=activation,
                                   dropout=dropout,
                                   max_degree=max_degree)
    else:
      dims = [feature_dim] + [hidden_dim] * n_layers
      self.blocks = dgl_mx.Sequential()
      for layer_no in range(n_layers):
        self.blocks.add(NeuralFPsBlock(feature_dim=dims[layer_no],
                                       output_dim=dims[layer_no + 1],
                                       batchnorm=batchnorm,
                                       activation=activation,
                                       dropout=dropout,
                                       max_degree=max_degree))

  def forward(
      self,
      dgl_graph: dgl.DGLGraph,
      node_features: mx.nd.NDArray,
  ) -> mx.nd.NDArray:
    """Applies series of GCNBlock graph convolutions.

    Input shape: (nodes, atom_features)
    Output shape: (nodes, `hidden_dim`)
    """
    return self.blocks(dgl_graph, node_features)


class NeuralFPsBlock(mx.gluon.Block):
  """Duvenaud et al. Neural Fingerprints block.

  Args:
    feature_dim: The dimension of node feature space of the graph that will be
      processed.
    max_degree: (Optional, defaults to 10). The maximum node degree.
    output_dim: (Optional, defaults to 64).
      The number of neurons in hidden and output layers.
    activation: (Optional, defaults to 'relu').
      The activation of the type supported by MXNet.
    batchnorm: (Optional, defaults to True).
      Whether to use batch normalization after dropout.
    dropout: (Optional, defaults to 0.2).
      The dropout rate.

  References:
    D. Duvenaud et al. Convolutional Networks on Graphs for Learning Molecular
    Fingerprints.
    (https://arxiv.org/abs/1609.02907)

  See also:
    DGL-LifeSci implementation of Neural Fingerprints:
      `dgllife.model.NFPredictor`
      `dgllife.model.NFGNN`
  """

  def __init__(
      self,
      feature_dim: int,
      output_dim: int,
      activation: ActivationT = None,
      max_degree: int = 10,
      dropout: float = 0.0,
      batchnorm: bool = False,

      prefix: Optional[str] = None,
      params: Optional[mx.gluon.ParameterDict] = None,
  ):
    super().__init__(prefix=prefix, params=params)

    self._vertex_w = [mx.gluon.nn.Dense(output_dim, in_units=feature_dim)
                      for _ in range(max_degree)]
    self._neighbor_w = [mx.gluon.nn.Dense(output_dim, in_units=feature_dim)
                        for _ in range(max_degree)]
    for block in self._vertex_w:
      self.register_child(block)
    for block in self._neighbor_w:
      self.register_child(block)

    self._batch_norm = mx.gluon.nn.BatchNorm() if batchnorm else lambda x: x
    self._activation = mx.gluon.nn.Activation(activation)
    self._dropout = mx.gluon.nn.Dropout(dropout) if dropout else lambda x: x

    self._output_dim = output_dim
    self._max_degree = max_degree

  def forward(
      self,
      graph: dgl.DGLGraph,
      features: mx.nd.NDArray,
  ) -> mx.nd.NDArray:
    """Feature processing -> batchnorm -> activation -> dropout -> pooling.
    """
    # shape=graph.number_of_nodes()
    in_degrees = graph.in_degrees().as_in_context(features.ctx)
    in_degrees = mx.nd.clip(in_degrees, 0, self._max_degree)
    node_idx = mx.nd.arange(in_degrees.shape[0], dtype=int)

    degree_membership = []
    for degree in range(self._max_degree + 1):
      try:
        mask = mx.nd.contrib.boolean_mask(node_idx, in_degrees == degree)
      except mx.MXNetError:
        mask = mx.nd.array([], dtype=int)
      degree_membership.append(mask)

    cur_max_deg = in_degrees.max().asscalar()

    with graph.local_scope():
      graph.ndata['h'] = features
      # noinspection PyUnresolvedReferences
      graph.update_all(fn.copy_u('h', 'm'), fn.sum('m', 'h'))
      h = graph.ndata.pop('h')

      x = mx.nd.empty((h.shape[0], self._output_dim))

      for degree in range(1, self._max_degree + 1):
        if degree > cur_max_deg:
          break
        nodes = degree_membership[degree]
        if nodes.shape[0] == 0:
          continue

        i = degree - 1
        y = self._neighbor_w[i](h[nodes]) + self._vertex_w[i](features[nodes])
        x = mx.nd.contrib.index_copy(x, nodes, y)

      x = self._batch_norm(x)
      x = self._activation(x)
      x = self._dropout(x)

    graph_self_loop = dgl.add_self_loop(graph)
    graph_self_loop.ndata['h'] = x
    # noinspection PyUnresolvedReferences
    graph_self_loop.update_all(fn.copy_u('h', 'm'), fn.max('m', 'h'))
    return graph_self_loop.ndata['h']
