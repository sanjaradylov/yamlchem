"""Test molecular graphs.
"""

import mxnet as mx
from rdkit.Chem import MolFromSmiles

from yamlchem.utils.graph import build_graph_from_molecule


def test_build_graph_from_molecule():
  r"""Build a DGL graph from a molecule and check correspondences of the
  true edge pairs with the ones returned by the graph.
  """
  smiles = 'CC(=O)O'
  molecule = MolFromSmiles(smiles)
  true_source_list = mx.nd.array([0, 1, 1, 2, 1, 3], dtype='int32')
  true_destination_list = mx.nd.array([1, 0, 2, 1, 3, 1], dtype='int32')

  dgl_graph = build_graph_from_molecule(molecule, add_self_loops=False)
  source_list, destination_list = dgl_graph.all_edges()

  assert dgl_graph.number_of_nodes() == molecule.GetNumAtoms()
  # An undirected graph contains edges of form (u, v) and (v, u), therefore
  # the number of the graph edges should be 2 times the number of bonds.
  assert dgl_graph.number_of_edges() == 2 * molecule.GetNumBonds()

  # Quite peculiar way to check if all pairs of elements are equal...
  # but mxnet ndarrays cannot be transformed into lists to use comparison
  # explicitly.
  assert mx.nd.prod(true_source_list == source_list)
  assert mx.nd.prod(true_destination_list == destination_list)
