"""Graph Convolutional Network blocks.

Classes:
  GCN: Kipf-Welling Multilayer Graph Convolutional Network.
  GCNBlock: Graph Convolutional Network block.
"""

__all__ = (
    'GCN',
)

from typing import Literal, Optional

import dgl
import dgl.nn.mxnet as dgl_mx
import mxnet as mx

from ...._types import ActivationT


GCNNormT = Literal['both', 'none', 'right']


class GCN(mx.gluon.Block):
  """Kipf-Welling (Multilayer) Graph Convolutional Network.

  Args:
    feature_dim: The dimension of node feature space of the graph that will be
      processed.
    hidden_dim: (Optional, defaults to 64).
      The number of neurons in hidden and output layers.
    n_layers: (Optional, defaults to 2). The number of layers.
    norm: (Optional, one of 'none', 'right', 'both' (default)).
      The GCN normalizer.
    activation: (Optional, defaults to 'relu').
      The activation of the type supported by MXNet.
    residual: (Optional, defaults to True).
      Whether to have residual connection between features and convolution.
    batchnorm: (Optional, defaults to True).
      Whether to use batch normalization after dropout.
    dropout: (Optional, defaults to 0.2). The dropout rate.

  See also:
    `dgl.nn.mxnet.GraphConv`

  References:
    T. N. Kipf and M. Welling. 2017. Semi-supervised classification
    with graph convolutional networks. In Proc. of ICLR.
    (https://arxiv.org/abs/1609.02907)
  """

  def __init__(
      self,
      feature_dim: int,
      *,
      hidden_dim: int = 64,
      n_layers: int = 2,
      activation: ActivationT = 'relu',
      norm: GCNNormT = 'both',
      dropout: float = 0.2,
      batchnorm: bool = True,
      residual: bool = True,

      prefix: Optional[str] = None,
      params: Optional[mx.gluon.ParameterDict] = None,
  ):
    super().__init__(prefix=prefix, params=params)

    with self.name_scope():
      if n_layers == 1:
        self.blocks = GCNBlock(feature_dim=feature_dim,
                               output_dim=hidden_dim,
                               norm=norm,
                               activation=activation,
                               residual=residual,
                               batchnorm=batchnorm,
                               dropout=dropout)
      else:
        dims = [feature_dim] + [hidden_dim] * n_layers
        self.blocks = dgl_mx.Sequential()
        for layer_no in range(n_layers):
          self.blocks.add(GCNBlock(feature_dim=dims[layer_no],
                                   output_dim=dims[layer_no + 1],
                                   norm=norm,
                                   activation=activation,
                                   residual=residual,
                                   batchnorm=batchnorm,
                                   dropout=dropout))

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


class GCNBlock(mx.gluon.Block):
  """Graph Convolutional Network block. Uses DGL implementation GCN.

  Args:
    feature_dim: The dimension of node feature space of the graph that will be
      processed.
    output_dim: (Optional, defaults to 64).
      The number of neurons in hidden and output layers.
    norm: (Optional, one of 'none', 'right', 'both' (default)).
      The GCN normalizer.
    activation: (Optional, defaults to 'relu').
      The activation of the type supported by MXNet.
    residual: (Optional, defaults to True).
      Whether to have residual connection between features and convolution.
    batchnorm: (Optional, defaults to True).
      Whether to use batch normalization after dropout.
    dropout: (Optional, defaults to 0.2). The dropout rate.

  See also:
    `dgl.nn.mxnet.GraphConv`

  References:
    T. N. Kipf and M. Welling. 2017. Semi-supervised classification
    with graph convolutional networks. In Proc. of ICLR.
    (https://arxiv.org/abs/1609.02907)
  """

  def __init__(
      self,
      feature_dim: int,
      *,
      output_dim: int = 64,
      activation: ActivationT = None,
      norm: GCNNormT = 'both',
      residual: bool = False,
      dropout: float = 0.0,
      batchnorm: bool = False,

      prefix: Optional[str] = None,
      params: Optional[mx.gluon.ParameterDict] = None,
  ):
    super().__init__(prefix=prefix, params=params)

    with self.name_scope():
      self.activation = mx.gluon.nn.Activation(activation)
      self.conv = dgl_mx.GraphConv(in_feats=feature_dim,
                                   out_feats=output_dim,
                                   norm=norm,
                                   activation=self.activation,
                                   allow_zero_in_degree=True)
      self._residual = residual and mx.gluon.nn.Dense(
          output_dim, weight_initializer=mx.init.Xavier())
      self.dropout = mx.gluon.nn.Dropout(dropout) if dropout else lambda x: x
      self.batchnorm = mx.gluon.nn.BatchNorm() if batchnorm else lambda x: x

  def forward(
      self,
      dgl_graph: dgl.DGLGraph,
      node_features: mx.ndarray.NDArray,
  ) -> mx.ndarray.NDArray:
    """Applies graph convolution, (optionally) dropout and batch normalization.

    Input shape: (nodes, atom_features)
    Output shape: (nodes, `output_dim`)
    """
    h = self.conv(dgl.add_self_loop(dgl_graph), node_features)
    if self._residual:
      h = h + self._residual(node_features)
      h = self.activation(h)
    h = self.batchnorm(h)
    return self.dropout(h)
