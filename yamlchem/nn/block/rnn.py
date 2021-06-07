"""RNN language models.

Classes:
  SMILESRNN: SMILES-based generative recurrent neural network.
"""

__all__ = (
    'SequenceAvgPool',
    'SMILESRNN',
    'SMILESRNNPredictor',
)

from typing import List, Literal, Optional, Tuple

import mxnet as mx

from ..._types import ActivationT


class SMILESRNN(mx.gluon.HybridBlock):
  """SMILES-based generative recurrent neural network.

  Args:
    vocab_size: Vocabulary dimension (input and output dimensions).
    use_one_hot: (Optional, defaults to False). Whether to apply learnable
      embedding or one-hot encoding.
    embedding_dim: (Optional, defaults to 32).
    embedding_dropout: (Optional, defaults to 0.4).
    embedding_dropout_axes: (Optional, defaults to 1).
      Apply dropout to features (1), tokens (2), or both (0).
    rnn_mode: (Optional, one of 'lstm' (default), 'gru', 'vanilla').
    rnn_num_layers: (Optional, defaults to 2).
    rnn_hidden_size: (Optional, defaults to 256).
    rnn_dropout: (Optional, defaults to 0.6).
      Regular dropout between hidden layers or RNN.
    output_dropout: (Optional, defaults to 0.0).
      Dropout between the final layer of RNN and output layer.

  Call args:
    x: Input sequences of type mxnet.nd.NDArray and
      Shape(batch_size, sequence_length).
    states: List of hidden states of type mxnet.nd.NDArray and
      Shape(rnn_num_layers, batch_size, rnn_hidden_size).

  Returns:
    x: Output logits of type mxnet.nd.NDArray and
      Shape(batch_size, sequence_length, vocab_size).
    states: List of processed hidden states of type mxnet.nd.NDArray and
      Shape(rnn_num_layers, batch_size, rnn_hidden_size).

  Attributes:
    embedding: One-hot encoding or mxnet.gluon.nn.Embedding optionally
      followed by dropout.
    rnn: RNN layer.
    dropout: Feature dropout.
    output: Linear projection (without activation).
  """

  def __init__(
      self,
      vocab_size: int,
      *,
      use_one_hot: bool = False,
      embedding_dim: int = 32,
      embedding_dropout: float = 0.4,
      embedding_dropout_axes: Literal[0, 1] = 1,
      rnn_mode: Literal['lstm', 'gru', 'vanilla'] = 'lstm',
      rnn_num_layers: int = 2,
      rnn_hidden_size: int = 256,
      rnn_dropout: float = 0.6,
      output_dropout: float = 0.0,

      prefix: Optional[str] = None,
      params: Optional[mx.gluon.ParameterDict] = None,
  ):
    super().__init__(prefix=prefix, params=params)

    self._use_one_hot = use_one_hot
    self._vocab_size = vocab_size

    with self.name_scope():
      if not self._use_one_hot:
        embedding = mx.gluon.nn.Embedding(
            vocab_size, embedding_dim,
            weight_initializer=mx.init.Xavier())
        if embedding_dropout != 0.0:
          self.embedding = mx.gluon.nn.HybridSequential()
          with self.embedding.name_scope():
            self.embedding.add(embedding)
            self.embedding.add(mx.gluon.nn.Dropout(embedding_dropout,
                                                   embedding_dropout_axes))
        else:
          self.embedding = embedding
      self.rnn = {
          'vanilla': mx.gluon.rnn.RNN,
          'lstm': mx.gluon.rnn.LSTM,
          'gru': mx.gluon.rnn.GRU,
      }[rnn_mode](rnn_hidden_size, rnn_num_layers,
                  dropout=rnn_dropout,
                  i2h_weight_initializer=mx.init.Xavier(),
                  h2h_weight_initializer=mx.init.Orthogonal())
      self.dropout = mx.gluon.nn.Dropout(output_dropout)
      self.output = mx.gluon.nn.Dense(vocab_size, flatten=False,
                                      weight_initializer=mx.init.Xavier())

  # noinspection PyMethodOverriding
  def hybrid_forward(
      self, f,
      x: mx.nd.NDArray,
      states: List[mx.nd.NDArray],
  ) -> Tuple[mx.nd.NDArray, List[mx.nd.NDArray]]:
    """Runs embedding -> rnn -> dropout -> output.

    Args:
      f: (Ignored) mxnet.nd or mxnet.sym.
      x: Input sequences of type mxnet.nd.NDArray and
        Shape(batch_size, sequence_length).
      states: List of hidden states of type mxnet.nd.NDArray and
        Shape(rnn_num_layers, batch_size, rnn_hidden_size).

    Returns:
      x: Output logits of type mxnet.nd.NDArray and
        Shape(batch_size, sequence_length, vocab_size).
      states: List of processed hidden states of type mxnet.nd.NDArray and
        Shape(rnn_num_layers, batch_size, rnn_hidden_size).
    """
    # b=batch size, t=time steps, v=vocab dim, e=embed dim, h=hidden units
    if self._use_one_hot:
      x = f.one_hot(f.transpose(x), self._vocab_size)  # Shape(t, b, v)
    else:
      x = self.embedding(f.transpose(x))               # Shape(t, b, e)
    x, states = self.rnn(x, states)                    # Shape(t, b, h)
    x = self.dropout(x)                                # Shape(t, b, h)
    x = self.output(x)                                 # Shape(t, b, v)
    return x.swapaxes(0, 1), states                    # Shape(b, t, v)


class SequenceAvgPool(mx.gluon.HybridBlock):
  """Average sequence representations of
  Shape(sequence length, batch_size, hidden_size) over all time steps.

  Call args:
    x: Feature representations of
      Shape(sequence length, batch size, hidden size).
    valid_lens: Number of non-padding tokens, Shape(batch size,)

  Returns:
    Averaged feature representation of Shape(batch size, hidden size).
  """

  # noinspection PyMethodOverriding
  def hybrid_forward(
      self,
      f,
      x: mx.nd.NDArray,
      valid_lens: mx.nd.NDArray,
  ) -> mx.nd.NDArray:
    """Forward propagation.

    Args:
      f: (Ignored). mxnet.nd or mxnet.sym.
      x: Feature representations of
        Shape(sequence length, batch size, hidden size).
      valid_lens: Number of non-padding tokens, Shape(batch size,)

    Returns:
      Averaged feature representation of Shape(batch size, hidden size).
    """
    x_masked = f.SequenceMask(x, sequence_length=valid_lens,
                              use_sequence_length=True)
    return f.broadcast_div(f.sum(x_masked, axis=0),
                           f.expand_dims(valid_lens, axis=1))


class SMILESRNNPredictor(mx.gluon.HybridBlock):
  """Supervised learning using SMILESRNN language model.

  Call args:
    x: Input sequences of Shape(batch size, sequence length).
    states: List of hidden states of type mxnet.nd.NDArray and
      Shape(rnn_num_layers, batch_size, rnn_hidden_size).
    valid_lens: Number of non-padding tokens, Shape(batch size,)

  Returns:
    x: Logits of Shape(batch size, output size).
    states: List of processed hidden states of type mxnet.nd.NDArray and
      Shape(rnn_num_layers, batch_size, rnn_hidden_size).

  Examples:
    >>> from yamlchem.nn.block.rnn import SMILESRNN
    >>> from yamlchem.nn.block.rnn import SequenceAvgPool
    >>> smilesrnn = SMILESRNN(10)
    >>> # Training `smilesrnn` ...
    >>> pool = SequenceAvgPool()
    >>> predictor = SMILESRNNPredictor(smilesrnn, pool, output_size=1)
    >>> predictor.output.initialize()
    >>> # Train and evaluate `predictor`.
  """

  def __init__(
      self,
      smilesrnn: SMILESRNN,
      pool: Optional[mx.gluon.HybridBlock] = None,
      dropout: float = 0.0,
      output_size: int = 1,
      activation: ActivationT = None,

      prefix: Optional[str] = None,
      params: Optional[mx.gluon.ParameterDict] = None,
  ):
    super().__init__(prefix=prefix, params=params)

    with self.name_scope():
      self.embedding = smilesrnn.embedding
      self.rnn = smilesrnn.rnn
      self.pool = pool or SequenceAvgPool()
      self.output = mx.gluon.nn.HybridSequential()
      with self.output.name_scope():
        self.output.add(mx.gluon.nn.Dropout(dropout))
        self.output.add(mx.gluon.nn.Dense(output_size, activation=activation,
                                          weight_initializer=mx.init.Xavier(),
                                          flatten=False))

  # noinspection PyMethodOverriding
  def hybrid_forward(
      self,
      f,
      x: mx.nd.NDArray,
      states: List[mx.nd.NDArray],
      valid_lens: mx.nd.NDArray,
  ):
    """Runs embedding -> rnn -> pool -> dropout -> output.

    Args:
      f: (Ignored). mxnet.nd or mxnet.sym.
      x: Input sequences of Shape(batch size, sequence length).
      states: List of hidden states of type mxnet.nd.NDArray and
        Shape(rnn_num_layers, batch_size, rnn_hidden_size).
      valid_lens: Number of non-padding tokens, Shape(batch size,)

    Returns:
      x: Logits of Shape(batch size, output size).
      states: List of processed hidden states of type mxnet.nd.NDArray and
        Shape(rnn_num_layers, batch_size, rnn_hidden_size).
    """
    # b=batch size, t=time steps, o=output dim, e=embed dim, h=hidden units
    x = f.transpose(x)               # Shape(t, b)
    x = self.embedding(x)            # Shape(t, b, e)
    x, states = self.rnn(x, states)  # Shape(t, b, h)
    x = self.pool(x, valid_lens)     # Shape(h, b)
    x = self.output(x)               # Shape(o, b)
    return x, states                 # Shape(b, o)
