"""Test objects that build molecular features.
"""

import mxnet as mx
import pytest
from rdkit.Chem import MolFromSmiles

from yamlchem.feature import MoleculeTransformer
from yamlchem.feature.fingerprint import ECFP
from yamlchem.feature.graph import (
    AtomFeaturizer,
    BondFeaturizer,
    get_canonical_atom_featurizers,
    get_canonical_bond_featurizers,
    OneHotEncoder,
    OneHotFeaturizer,
    PropertyFeaturizer,
)


def test_one_hot_encoding():
  """Test OneHotEncoder.
  """
  valid_labels = list('CHS')
  label = 'C'
  encoder = OneHotEncoder(valid_labels, encode_unknown=False)
  assert encoder(label) == [True, False, False]

  unknown_label = 'Li'
  encoder.encode_unknown = True
  assert encoder(unknown_label) == [False, False, False, True]

  encoder.encode_unknown = False
  assert encoder(unknown_label) == [False, False, False]

  encoder.valid_features = list('CO')
  assert encoder(label) == [True, False]
  assert encoder(unknown_label) == [False, False]


def test_property_atom_encoding():
  """Test PropertyFeaturizer.
  """
  molecule = MolFromSmiles('CCO')
  atom = molecule.GetAtomWithIdx(0)
  featurizer = PropertyFeaturizer('GetTotalNumHs')
  assert featurizer(atom) == [3]


def test_one_hot_atom_encoding():
  """Test OneHotFeaturizer.
  """
  molecule = MolFromSmiles('CCO')
  atom = molecule.GetAtomWithIdx(0)
  encoder = OneHotEncoder(list('COH'), encode_unknown=False)
  featurizer = OneHotFeaturizer(encoder, 'GetSymbol')
  # Features: ['C', 'O', 'H'], input: 'C'.
  assert featurizer(atom) == [True, False, False]

  encoder = OneHotEncoder(list(range(3)), encode_unknown=False)
  featurizer = OneHotFeaturizer(encoder, 'GetDegree')
  # Features: [0, 1, 2], input: degree of 1st 'C' = 1.
  assert featurizer(atom) == [False, True, False]


def test_atom_featurizer():
  """Test AtomFeaturizer.
  """
  molecule = MolFromSmiles('CC(=O)O')
  feature_name = 'atom_features'
  default_atom_featurizers = get_canonical_atom_featurizers()
  featurizer = AtomFeaturizer(feature_name)
  featurizer.add(*default_atom_featurizers)
  feature_space = featurizer(molecule)[feature_name]

  n_atoms = molecule.GetNumAtoms()
  assert n_atoms == feature_space.shape[0]

  for atom_idx in range(n_atoms):
    current_feature_idx = 0

    for featurizer in default_atom_featurizers:
      atom = molecule.GetAtomWithIdx(atom_idx)
      true_feature = mx.nd.array(featurizer(atom))
      generated_feature = feature_space[
          atom_idx,
          current_feature_idx:current_feature_idx+len(true_feature)]
      current_feature_idx += len(true_feature)

      assert mx.nd.prod(generated_feature == true_feature)


def test_bond_featurizer():
  """Test BondFeaturizer.
  """
  molecule = MolFromSmiles('CC(=O)O')
  feature_name = 'bond_features'
  default_bond_featurizers = get_canonical_bond_featurizers()
  featurizer = BondFeaturizer(feature_name)
  featurizer.add(*default_bond_featurizers)
  feature_space = featurizer(molecule)[feature_name]

  n_bonds = molecule.GetNumBonds()
  assert 2 * n_bonds == feature_space.shape[0]

  for bond_idx in range(n_bonds):
    current_feature_idx = 0

    for featurizer in default_bond_featurizers:
      bond = molecule.GetBondWithIdx(bond_idx)
      true_feature = mx.nd.array(featurizer(bond))
      generated_feature = feature_space[
          2*bond_idx,
          current_feature_idx:current_feature_idx+len(true_feature)]
      current_feature_idx += len(true_feature)

      assert mx.nd.prod(generated_feature == true_feature)


def test_ecfp():
  mp = ECFP(radius=0, n_bits=0)
  smiles_strings = ('CCO', 'N#N', 'C#N')
  molecules = MoleculeTransformer().fit_transform(smiles_strings)

  with pytest.raises(ValueError):
    mp.fit(molecules)

  mp.radius = 2
  mp.n_bits = 2048
  fingerprints = mp.fit_transform(molecules)

  assert fingerprints.shape[0] == len(molecules)
  assert fingerprints.shape[1] == mp.n_bits
