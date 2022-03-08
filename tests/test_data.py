"""Test datasets.
"""

import pathlib
import shutil

import dgl
import mxnet as mx
import numpy as np

from yamlchem.data.loader import batchify_labeled_masked_graphs
from yamlchem.data.sets import ESOLDataset, Tox21Dataset
from yamlchem.feature.graph import N_DEFAULT_ATOM_FEATURES
from yamlchem.data.splitter import train_test_split, train_valid_test_split


def test_esol_dataset():
  """Test ESOLDataset and temporary data in disk.
  """
  dataset = ESOLDataset(verbose=False, force_reload=True)

  for graph, label in dataset:
    assert isinstance(graph, dgl.DGLGraph)
    assert isinstance(label, float)
    assert graph.ndata['h'].shape[1] == N_DEFAULT_ATOM_FEATURES
    assert not np.isnan(label)

  shutil.rmtree(dataset.raw_path)


def test_tox21_dataset():
  """Test Tox21Dataset class and temporary Tox21 data in disk.
  """
  dataset = Tox21Dataset(verbose=False, force_reload=True)

  home_path = pathlib.Path.home()

  assert dataset.name == 'tox21'
  assert dataset.raw_dir == str(home_path / '.dgl')
  assert dataset.raw_dir == dataset.save_dir
  assert dataset.raw_path == str(home_path / '.dgl' / dataset.name)
  assert dataset.has_cache()

  shutil.rmtree(dataset.raw_path)

  graph, labels, mask = dataset[0]

  assert labels.shape[0] == mask.shape[0] == len(dataset.tasks)
  assert graph.ndata['h'].shape[1] == N_DEFAULT_ATOM_FEATURES


def test_labeled_graph_data_loader():
  dataset = Tox21Dataset(verbose=False)

  batch_size = 4

  graphs, labels, masks = [], [], []
  for index in range(batch_size):
    graph, label, mask = dataset[index]
    graphs.append(graph)
    labels.append(label)
    masks.append(mask)
  graphs = dgl.batch(graphs)
  labels = mx.nd.array(labels)
  masks = mx.nd.array(masks)

  shutil.rmtree(dataset.raw_path)

  data_loader = mx.gluon.data.DataLoader(
      dataset, batch_size=batch_size, shuffle=False, last_batch='discard',
      batchify_fn=batchify_labeled_masked_graphs)
  batch_graphs, batch_labels, batch_masks = next(iter(data_loader))

  assert batch_graphs.number_of_nodes() == graphs.number_of_nodes()
  assert batch_graphs.number_of_edges() == graphs.number_of_edges()
  assert batch_labels.shape == labels.shape
  assert batch_masks.shape == masks.shape


def test_splitter():
  data = mx.gluon.data.ArrayDataset(list(range(10)))
  train_data, valid_data = train_test_split(data, 0.2, False)
  assert list(train_data) == list(range(8))
  assert list(valid_data) == [8, 9]

  train_data, valid_data = train_test_split(data, 0.2, True)
  assert not (set(train_data) & set(valid_data))

  train_data, valid_data, test_data = train_valid_test_split(data, 0.2, 0.1)
  assert (len(train_data), len(valid_data), len(test_data)) == (7, 2, 1)
