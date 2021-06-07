"""Bond-based featurization utilities.

Constants:
  N_DEFAULT_BOND_FEATURES: Canonical bond feature dimension.

Classes:
  BondFeaturizer: Calculation of bond features for every bond in a compound.

Functions:
  get_canonical_bond_featurizers: Get the set of canonical bond featurizers.
"""

__all__ = (
    'BondFeaturizer',
    'get_canonical_bond_featurizers',
    'N_DEFAULT_BOND_FEATURES',
)

from numbers import Real
from typing import List, Tuple

from rdkit.Chem import rdchem

from .base import CompoundFeaturizer
from ._common import OneHotEncoder, OneHotFeaturizer, PropertyFeaturizer


N_DEFAULT_BOND_FEATURES = 12


# The set of canonical bond featurizers.
def get_canonical_bond_featurizers() -> Tuple[PropertyFeaturizer, ...]:
  """Returns a tuple of canonical bond featurizers:
  - if bond is conjugated;
  - if bond is in a ring;
  - bond type one-hot encoding;
  - stereo configuration one-hot encoding.
  Embedded feature space has dimension 12.
  """
  return (
      # Check if bond is conjugated.
      PropertyFeaturizer('GetIsConjugated'),

      # Check if bond is in a ring.
      PropertyFeaturizer('IsInRing'),

      # Bond type one-hot encoding.
      OneHotFeaturizer(
          encoder=OneHotEncoder(
              valid_features=(
                  rdchem.BondType.SINGLE,
                  rdchem.BondType.DOUBLE,
                  rdchem.BondType.TRIPLE,
                  rdchem.BondType.AROMATIC,
              ),
          ),
          attribute='GetBondType',
      ),

      # Stereo configuration one-hot encoding.
      OneHotFeaturizer(
          encoder=OneHotEncoder(
              valid_features=(
                  rdchem.BondStereo.STEREONONE,
                  rdchem.BondStereo.STEREOANY,
                  rdchem.BondStereo.STEREOZ,
                  rdchem.BondStereo.STEREOE,
                  rdchem.BondStereo.STEREOCIS,
                  rdchem.BondStereo.STEREOTRANS,
              ),
          ),
          attribute='GetStereo',
      ),
  )


class BondFeaturizer(CompoundFeaturizer):
  """Featurizes every bond of a compound, double the results, and stacks
  them into a two-dimensional feature space.

  Args:
    See the documentation of `yamlchem.feature.graph.CompoundFeaturizer`.

  Examples:
    >>> from rdkit.Chem import MolFromSmiles
    >>> molecule = MolFromSmiles('CC(=O)C')
    >>> default_bond_featurizers = get_canonical_bond_featurizers()
    >>> featurizer = BondFeaturizer(name='bond')
    >>> featurizer.add(*default_bond_featurizers)
    >>> feature_space = featurizer(molecule)['bond']
    >>> feature_space.shape  # 3 pairs of bonds and 12 stacked features.
    (6, 12)

  See also:
    1. `yamlchem.feature.graph.AtomFeaturizer`
    2. `yamlchem.feature.graph.CompoundFeaturizer`
  """

  def featurize(self, molecule: rdchem.Mol) -> List[List[Real]]:
    """Calls featurizers on every bond of `molecule`.

    Args:
      molecule: RDKit molecule.

    Returns:
      The list of feature vectors for every atom.

    Notes:
      One does not have to call this method explicitly as it is an
      abstract method of `yamlchem.feature.graph.CompoundFeaturizer`.
    """
    feature_space: List[List[Real]] = []
    # noinspection PyArgumentList
    n_bonds: int = molecule.GetNumBonds()

    for bond_idx in range(n_bonds):
      bond: rdchem.Bond = molecule.GetBondWithIdx(bond_idx)
      feature_vector: List[Real] = self.concatenate(bond)
      feature_space.extend([feature_vector]*2)

    return feature_space
