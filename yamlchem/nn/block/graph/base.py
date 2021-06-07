"""The :mod:`yamlchem.nn.model.base` implements the NN model base classes.

Classes:
  NodeGNNPredictor: Supervised learning, node-only graph convolutional model.
"""

__all__ = (
    'NodeGNNPredictor',
)

from typing import Optional

import dgl
import mxnet as mx


class NodeGNNPredictor(mx.gluon.nn.Block):
  """Supervised learning, node-only graph convolutional model.

  Args:
    convolution: Graph convolutional network returning node feature space.
    readout: Graph readout block.
    output_dim: The number of output units/tasks.

  Examples:
    >>> import mxnet as mx
    >>> import yamlchem as yc
    >>> data = yc.data.sets.ESOLDataset()
    >>> loader = mx.gluon.data.DataLoader(
    ...     data, batch_size=32, last_batch='rollover',
    ...     batchify_fn=yc.data.loader.batchify_labeled_graphs)
    >>> loss = mx.gluon.loss.L2Loss()
    >>> optimizer = mx.optimizer.Adam(learning_rate=0.025)
    >>> gcn = yc.nn.block.graph.GCN(yc.feature.graph.N_DEFAULT_ATOM_FEATURES)
    >>> readout = yc.nn.block.graph.StandardReadout('mean')
    >>> model = NodeGNNPredictor(gcn, readout, 1)
  """

  def __init__(
      self,
      convolution: mx.gluon.nn.Block,
      readout: mx.gluon.HybridBlock,
      output_dim: int,

      prefix: Optional[str] = None,
      params: Optional[mx.gluon.ParameterDict] = None,
  ):
    super().__init__(prefix=prefix, params=params)

    with self.name_scope():
      self.convolution = convolution
      self.readout = readout
      self.output = mx.gluon.nn.Dense(output_dim,
                                      weight_initializer=mx.init.Xavier())

  def forward(
      self,
      dgl_graph: dgl.DGLGraph,
      node_features: mx.nd.NDArray,
  ) -> mx.nd.NDArray:
    """Runs graph convolution -> readout -> feed-forward.
    """
    # Shape(num_nodes,feature_dim)
    h = self.convolution(dgl_graph, node_features)
    h = self.readout(dgl_graph, h)
    return self.output(h)
