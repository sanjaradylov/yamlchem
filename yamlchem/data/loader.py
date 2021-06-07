"""The :mod:`yamlchem.data.loader` module includes tools to initialize data
loaders (mini-batch samplers).
"""

__all__ = (
    'batchify_labeled_graphs',
    'batchify_labeled_masked_graphs',
    'BatchifyGraph',
    'BatchifySMILES',
)

import random
from dataclasses import dataclass
from typing import Collection, List, Literal, Optional, Tuple

import dgl
import mxnet as mx

from ..utils.smiles import Vocabulary


def batchify_labeled_graphs(samples) -> Tuple[dgl.DGLGraph, mx.nd.NDArray]:
  """Returns a tuple of graph and its label. For Gluon data loaders.
  """
  graphs = dgl.batch([graph for graph, label in samples])
  labels = mx.nd.array([label for graph, label in samples])
  return graphs, labels


def batchify_labeled_masked_graphs(samples) \
    -> Tuple[dgl.DGLGraph, mx.nd.NDArray, mx.nd.NDArray]:
  """Returns a tuple of graph, its label and label mask. For Gluon data
  loaders.
  """
  graphs = dgl.batch([graph for graph, label, mask in samples])
  labels = mx.nd.array([label for graph, label, mask in samples])
  masks = mx.nd.array([mask for graph, label, mask in samples])
  return graphs, labels, masks


@dataclass(init=True, repr=True, eq=True, frozen=False)
class BatchifyGraph:
  """Functor for Gluon data loaders to batchify SMILES graphs.
  """
  labeled: bool = True
  masked: bool = True

  @dataclass(init=True, repr=False, eq=False, frozen=True)
  class Batch:
    """Mini-batch of graphs, labels, and label masks.
    """
    graph: dgl.DGLGraph
    label: Optional[mx.nd.NDArray] = None
    mask: Optional[mx.nd.NDArray] = None

  def __call__(self, dataset: Collection) -> Batch:
    """Returns a (mini-)batch of graphs and optionally, labels and masks.
    """
    graphs = dgl.batch([sample[0] for sample in dataset])
    labels = masks = None
    if self.labeled:
      labels = mx.nd.array([sample[1] for sample in dataset])
    if self.masked:
      masks = mx.nd.array([sample[2] for sample in dataset])
    return self.Batch(graphs, labels, masks)


@dataclass(init=True, repr=True, eq=True, frozen=False)
class BatchifySMILES:
  """Functor for Gluon data loaders to batchify SMILES sequences.
  """
  vocabulary: Vocabulary
  clip_length: Optional[int] = None
  max_offset: int = 0
  offset_direction: Literal['left', 'both', 'right'] = 'left'
  include_special: bool = True
  labeled: bool = False
  masked: bool = False

  @dataclass(init=True, repr=False, eq=False, frozen=True)
  class Batch:
    """Mini-batch of sequences, labels, and label masks.
    """
    sequence: mx.nd.NDArray      # Shape(batch size, sequence length)
    valid_length: mx.nd.NDArray  # Shape(batch size,)
    label: Optional[mx.nd.NDArray] = None
    mask: Optional[mx.nd.NDArray] = None

  def __call__(self, dataset: Collection) -> Batch:
    """Returns a (mini-)batch of sequences, valid lengths,  and optionally,
    labels and masks.
    """
    clip_length: int = self.clip_length or max(map(len, dataset))

    tokens_list: List[List[int]] = []
    valid_lens: List[int] = []
    label_list: Optional[list] = []
    mask_list: Optional[list] = []

    for sample in dataset:
      if self.labeled:
        if self.masked:
          smiles, label, mask = sample
          label_list.append(label)
          mask_list.append(mask)
        else:
          smiles, label = sample
          label_list.append(label)
      else:
        smiles = sample

      if self.include_special:
        smiles = (f'{self.vocabulary.tokenizer.bos}'
                  f'{smiles}'
                  f'{self.vocabulary.tokenizer.eos}')
      token_ids: List[int] = self.vocabulary.encode(smiles)

      offset = random.randint(0, self.max_offset)
      if self.offset_direction == 'left':
        token_ids = token_ids[offset:]
      elif self.offset_direction == 'right':
        token_ids = token_ids[:-offset] if offset > 0 else token_ids
      elif self.offset_direction == 'both':
        direction = random.choice(['left', 'right'])
        if direction == 'left':
          token_ids = token_ids[offset:]
        else:
          token_ids = token_ids[:-offset] if offset > 0 else token_ids

      smiles_size = len(token_ids)
      if smiles_size < clip_length:
        valid_lens.append(smiles_size)
        token_ids += [self.vocabulary.pad] * (clip_length-smiles_size)
      else:
        valid_lens.append(clip_length)
        token_ids = token_ids[:clip_length]

      tokens_list.append(token_ids)

    # Shape(len(dataset), self.clip_length)
    sequences = mx.nd.array(tokens_list, dtype=int)
    # Shape(len(dataset),)
    valid_lengths = mx.nd.array(valid_lens, dtype=int)
    if self.labeled:
      if self.masked:
        return self.Batch(sequences, valid_lengths,
                          mx.nd.array(label_list), mx.nd.array(mask_list))
      return self.Batch(sequences, valid_lengths, mx.nd.array(label_list))
    return self.Batch(sequences, valid_lengths)
