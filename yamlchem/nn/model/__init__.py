"""The :mod:`yamlchem.nn.model` implements neural network estimators.

Functions:
  train_gnn_predictor: Trains GNN for supervised learning tasks.
  train_rnn_lm: Trains SMILESRNN language model.
  train_rnn_predictor: Trains SMILESRNN-based model for supervised learning.
"""

__all__ = (
    'train_gnn_predictor',
    'train_rnn_lm',
    'train_rnn_predictor',
)

from .rnn import train_rnn_lm, train_rnn_predictor
from .graph import train_gnn_predictor
