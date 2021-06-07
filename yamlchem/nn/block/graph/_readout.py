"""Graph readout blocks.

Classes:
  StandardReadout: Max/Mean/Sum readout of graph node features.
  WeightSum: Weighted sum of node features.
"""

__all__ = (
    'StandardReadout',
    'WeightSum',
)

from typing import Literal, Optional

import dgl
import mxnet as mx


class StandardReadout(mx.gluon.HybridBlock):
  """Max/Mean/Sum readout of graph node features.
  """

  def __init__(
      self,
      mode: Literal['max', 'mean', 'sum'],
      prefix: Optional[str] = None,
  ):
    super().__init__(prefix=prefix, params=None)

    self.mode = {
        'max': dgl.max_nodes,
        'mean': dgl.mean_nodes,
        'sum': dgl.sum_nodes,
    }[mode]

  # noinspection PyMethodOverriding
  def hybrid_forward(
      self,
      _,
      g: dgl.DGLGraph,
      features: mx.nd.NDArray,
    ) -> mx.nd.NDArray:
    """Summarizes node features.
    """
    with g.local_scope():
      g.ndata['h'] = features
      return self.mode(g, 'h')  # Shape(batch_size, feature_dim)


class WeightSum(mx.gluon.HybridBlock):
  """Weighted sum of node features.
  """

  def __init__(
      self,
      prefix: Optional[str] = None,
      params: Optional[mx.gluon.ParameterDict] = None,
  ):
    super().__init__(prefix=prefix, params=params)

    self.weight = mx.gluon.nn.Dense(1, activation='sigmoid',
                                    weight_initializer=mx.init.Xavier())

  # noinspection PyMethodOverriding
  def hybrid_forward(
      self,
      _,
      g: dgl.DGLGraph,
      features: mx.nd.NDArray,
  ) -> mx.nd.NDArray:
    """Compute feature weights and sum.
    """
    with g.local_scope():
      g.ndata['h'] = features
      g.ndata['w'] = self.weight(g.ndata['h'])
      return dgl.sum_nodes(g, 'h', 'w')
