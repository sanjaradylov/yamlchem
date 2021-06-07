"""Atom-based featurization utilities.

Constants:
  ATOMS: The set of valid atomic symbols.
  N_DEFAULT_ATOM_FEATURES: Canonical atom feature dimension.

Classes:
  AtomFeaturizer: Calculation of atomic features for every atom in a compound.

Functions:
  get_canonical_atom_featurizers: Get the sequence of canonical atom
    featurizers.
  get_default_one_hot_encoder: Get the default atom-based one-hot encoding.
"""

__all__ = (
    'AtomFeaturizer',
    'ATOMS',
    'N_DEFAULT_ATOM_FEATURES',
    'get_canonical_atom_featurizers',
    'get_default_one_hot_encoder',
)

from numbers import Real
from typing import List, Tuple

from rdkit.Chem import rdchem

from ._common import OneHotEncoder, OneHotFeaturizer, PropertyFeaturizer
from .base import CompoundFeaturizer


N_DEFAULT_ATOM_FEATURES = 74


ATOMS = frozenset([
    'Ac', 'Ag', 'Al', 'Am', 'Ar', 'As', 'At', 'Au', 'B', 'Ba', 'Be', 'Bh',
    'Bi', 'Bk', 'Br', 'C', 'Ca', 'Cd', 'Ce', 'Cf', 'Cl', 'Cm', 'Co', 'Cr',
    'Cs', 'Cu', 'Db', 'Dy', 'Er', 'Es', 'Eu', 'F', 'Fe', 'Fm', 'Fr', 'Ga',
    'Gd', 'Ge', 'H', 'He', 'Hf', 'Hg', 'Ho', 'Hs', 'I', 'In', 'Ir', 'K',
    'Kr', 'La', 'Li', 'Lr', 'Lu', 'Md', 'Mg', 'Mn', 'Mo', 'Mt', 'N', 'Na',
    'Nb', 'Nd', 'Ne', 'Ni', 'No', 'Np', 'O', 'Os', 'P', 'Pa', 'Pb', 'Pd',
    'Pm', 'Po', 'Pr', 'Pt', 'Pu', 'Ra', 'Rb', 'Re', 'Rf', 'Rh', 'Rn',
    'Ru', 'S', 'Sb', 'Sc', 'Se', 'Sg', 'Si', 'Sm', 'Sn', 'Sr', 'Ta', 'Tb',
    'Tc', 'Te', 'Th', 'Ti', 'Tl', 'Tm', 'U', 'V', 'W', 'Xe', 'Y', 'Yb',
    'Zn', 'Zr'
])


def get_default_one_hot_encoder() -> OneHotEncoder:
  """Returns the default atom-based one-hot encoding.
  """
  return OneHotEncoder(
      valid_features=tuple(sorted(ATOMS)),
      encode_unknown=False,
  )


def get_canonical_atom_featurizers() -> Tuple[PropertyFeaturizer, ...]:
  """Returns a tuple of canonical atom featurizers:
  - aromaticity check;
  - formal charge;
  - number of radical electrons;
  - atomic symbol one-hot encoding;
  - atom degree one-hot encoding;
  - implicit valence one-hot encoding;
  - number of Hs one-hot encoding;
  - hybridization one-hot encoding.
  Embedded feature space has dimension 74.
  """
  return (
      # Aromaticity check.
      PropertyFeaturizer('GetIsAromatic'),

      # Formal charge.
      PropertyFeaturizer('GetFormalCharge'),

      # Number of radical electrons.
      PropertyFeaturizer('GetNumRadicalElectrons'),

      # Atomic symbol one-hot encoding.
      OneHotFeaturizer(
          encoder=OneHotEncoder(
              valid_features=(
                  'Ag', 'Al', 'As', 'Au', 'B', 'Br', 'C', 'Ca', 'Cd', 'Cl',
                  'Co', 'Cr', 'Cu', 'F', 'Fe', 'Ge', 'H', 'Hg', 'I', 'In',
                  'K', 'Li', 'Mg', 'Mn', 'N', 'Na', 'Ni', 'O', 'P', 'Pb',
                  'Pd', 'Pt', 'S', 'Sb', 'Se', 'Si', 'Sn', 'Ti', 'Tl', 'V',
                  'Yb', 'Zn', 'Zr'
              ),
          ),
          attribute='GetSymbol',
      ),

      # Degree one-hot encoding.
      OneHotFeaturizer(
          encoder=OneHotEncoder(valid_features=range(11)),
          attribute='GetDegree',
      ),

      # Implicit valence one-hot encoding.
      OneHotFeaturizer(
          encoder=OneHotEncoder(valid_features=range(7)),
          attribute='GetImplicitValence',
      ),

      # Number of Hs one-hot encoding.
      OneHotFeaturizer(
          encoder=OneHotEncoder(valid_features=range(5)),
          attribute='GetTotalNumHs',
      ),

      # Hybridization one-hot encoding.
      OneHotFeaturizer(
          encoder=OneHotEncoder(
              valid_features=(
                  rdchem.HybridizationType.SP,
                  rdchem.HybridizationType.SP2,
                  rdchem.HybridizationType.SP3,
                  rdchem.HybridizationType.SP3D,
                  rdchem.HybridizationType.SP3D2,
              ),
          ),
          attribute='GetHybridization',
      ),
  )


class AtomFeaturizer(CompoundFeaturizer):
  """Featurizes every atom of a compound and stacks them into a
  two-dimensional feature space.

  Examples:
    >>> from rdkit.Chem import MolFromSmiles
    >>> molecule = MolFromSmiles('CCO')
    >>> default_atom_featurizers = get_canonical_atom_featurizers()
    >>> featurizer = AtomFeaturizer(name='atom')
    >>> featurizer.add(*default_atom_featurizers)
    >>> feature_space = featurizer(molecule)['atom']
    >>> feature_space.shape  # 3 atoms and 74 stacked features.
    (3, 74)

  See also:
    1. `yamlchem.feature.graph.BondFeaturizer`
    2. `yamlchem.feature.graph.CompoundFeaturizer`
  """

  def featurize(self, molecule: rdchem.Mol) -> List[List[Real]]:
    """Calls featurizers on every atom of `molecule`.

    Notes:
      One does not have to call this method explicitly as it is an
      abstract method of `yamlchem.feature.graph.CompoundFeaturizer`.
    """
    feature_space: List[List[Real]] = []
    # noinspection PyArgumentList
    n_atoms: int = molecule.GetNumAtoms()

    for atom_idx in range(n_atoms):
      atom: rdchem.Atom = molecule.GetAtomWithIdx(atom_idx)
      feature_vector: List[Real] = self.concatenate(atom)
      feature_space.append(feature_vector)

    return feature_space
