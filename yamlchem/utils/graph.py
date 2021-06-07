"""The :mod:`yamlchem.utils.graph` implements utilities to construct molecular
graph objects.

Functions:
  build_graph_from_molecule: Construct a DGL graph from an RDKit molecule.
  from_smiles_to_molecule: Convert a SMILES string into an RDKit molecule.
"""

__all__ = (
    'build_graph_from_molecule',
    'from_smiles_to_molecule',
)


from typing import Dict, List, Optional, Type

import dgl
import mxnet as mx
from rdkit.Chem import MolFromSmiles
from rdkit.Chem.rdchem import Bond, Mol

from ..feature.graph import CompoundFeaturizer


def build_graph_from_molecule(
    molecule: Mol,
    *,
    add_self_loops: bool = False,
    node_featurizer: Optional[CompoundFeaturizer] = None,
    edge_featurizer: Optional[CompoundFeaturizer] = None,
    dtype: Type = mx.np.int32,
) -> dgl.DGLGraph:
  """Constructs a DGL graph from an RDKit molecule.

  Args:
    molecule: RDKit molecule.
    add_self_loops: (Optional, defaults to False).
      Whether to include self-loops.
    node_featurizer: (Optional). The node featurizer, which is called on
      `molecule` and attached to a graph.
    edge_featurizer: (Optional). The edge featurizer, which is called on
      `molecule` and attached to a graph.
    dtype: (Optional, defaults to 'int32'). Node and edge data type.

  Examples:
    >>> from rdkit.Chem import MolFromSmiles
    >>> m = MolFromSmiles('CC(=O)O')
    >>> g = build_graph_from_molecule(m, add_self_loops=True)
    >>> g.number_of_nodes(), g.number_of_edges()
    (4, 10)
  """
  # Build a DGL graph from source and destination lists.
  # noinspection PyArgumentList
  n_nodes: int = molecule.GetNumAtoms()
  dgl_graph = dgl.graph(([], []), num_nodes=n_nodes, idtype=dtype)

  source_list: List[int] = []
  destination_list: List[int] = []
  # noinspection PyArgumentList
  n_bonds = molecule.GetNumBonds()

  for bond_index in range(n_bonds):
    bond: Bond = molecule.GetBondWithIdx(bond_index)

    # noinspection PyArgumentList
    start_node: int = bond.GetBeginAtomIdx()
    # noinspection PyArgumentList
    end_node: int = bond.GetEndAtomIdx()

    source_list.extend([start_node, end_node])
    destination_list.extend([end_node, start_node])

  if add_self_loops:
    source_list.extend(range(n_nodes))
    destination_list.extend(range(n_nodes))

  dgl_graph.add_edges(source_list, destination_list)

  if node_featurizer is not None:
    node_features: Dict[str, mx.nd.NDArray] = node_featurizer(molecule)
    dgl_graph.ndata.update(node_features)

  if edge_featurizer is not None:
    edge_features: Dict[str, mx.nd.NDArray] = edge_featurizer(molecule)
    dgl_graph.edata.update(edge_features)

  return dgl_graph


def from_smiles_to_molecule(
    smiles: str,
    raise_error: bool = False,
    **mol_kwargs,
) -> Optional[Mol]:
  """Converts a SMILES string into an RDKit molecule.
  """
  molecule = MolFromSmiles(smiles, **mol_kwargs)
  if raise_error and molecule is None:
    raise TypeError(f'molecule {smiles} is not valid')
  return molecule
