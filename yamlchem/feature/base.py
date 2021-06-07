"""Common featurization utilities.

Exceptions:
  InvalidSMILESError

Functions:
  check_compounds_valid: Converts SMILES compounds into a list of RDKit
    molecules.

Classes:
  MoleculeTransformer: Scikit-learn-compatible transformer to convert SMILES
    strings into RDKit molecules.
"""

__all__ = (
    'check_compounds_valid',
    'InvalidSMILESError',
    'MoleculeTransformer',
)


from typing import List, Sequence

import numpy as np

from rdkit.Chem import Mol, MolFromSmiles
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import check_array


class InvalidSMILESError(ValueError):
  """For incorrect SMILES strings recognized by RDKit."""


def check_compounds_valid(
    compounds: Sequence[str],
    invalid: str = 'skip',
    **converter_kwargs,
) -> List[Mol]:
  """Converts SMILES compounds into a list of RDKit molecules.

  Args:
    compounds: Sequence of SMILES strings.
    invalid: (Optional; one of 'nan', 'raise', 'skip' (default)).
      Whether to a) replace invalid SMILES with `numpy.NaN`s,
      b) raise `InvalidSMILESError`, or c) ignore them.
    converter_kwargs: Optional. Key-word arguments for
      `rdkit.Chem.MolFromSmiles`.
  """
  molecules: List[Mol] = []

  for compound in compounds:
    molecule = MolFromSmiles(compound, **converter_kwargs)
    if molecule is not None and compound:
      molecules.append(molecule)
    elif invalid == 'nan':
      molecules.append(np.NaN)
    elif invalid == 'raise':
      raise InvalidSMILESError(
          f'cannot convert {compound!r} into molecule: invalid compound')
    elif invalid == 'skip':
      continue

  return molecules


class MoleculeTransformer(BaseEstimator, TransformerMixin):
  """Converts SMILES compounds into RDKit molecules. Invalid compounds are
  replaced with `numpy.NaN`s.

  Use prior to scikit-learn's FeatureUnions and/or Pipelines that accept RDKit
  molecules.

  Args:
    converter_kwargs: Optional; key-word arguments for
      `rdkit.Chem.MolFromSmiles`.
  """

  def __init__(self, **converter_kwargs):
    self.converter_kwargs = converter_kwargs

  # noinspection PyUnusedLocal
  # pylint: disable=unused-argument
  def fit(self, compounds: Sequence[str], _=None) -> 'MoleculeTransformer':
    """Sets up the number of passed features.
    """
    # noinspection PyAttributeOutsideInit
    self.n_features_in_ = 1

    return self

  def transform(self, compounds: Sequence[str]) -> List[Mol]:
    """Checks `compounds` validity and returns a list of RDKit molecules.

    Args:
      compounds: A sequence of SMILES strings. Invalid strings will be replaced
        by NaNs.
    """
    check_array(
        compounds, accept_large_sparse=False, dtype='object', ensure_2d=False)
    return check_compounds_valid(
        compounds, invalid='nan', **self.converter_kwargs)
