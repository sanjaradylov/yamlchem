"""The :mod:`yamlchem.feature.fingerprint` implements molecular fingerprint
featurizers.

Classes:
  ECFP or MorganFingerprints:
    Applies the Morgan algorithm to a set of compounds to get circular
    fingerprints.
"""

__all__ = (
    'ECFP',
    'MorganFingerprints',
)


from typing import Sequence, Union

import numpy as np
import scipy.sparse as sparse

from rdkit.Chem import Mol
from rdkit.Chem.AllChem import GetMorganFingerprintAsBitVect
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import check_scalar


class ECFP(BaseEstimator, TransformerMixin):
  """Applies the Morgan algorithm to a set of compounds to get circular
  fingerprints.

  Args:
    radius: (Optional, defaults to 2). The radius of fingerprint.
    n_bits: (Optional, defaults to 1024). The number of bits.
    return_type: (Optional; one of 'ndarray' (default), 'csr_sparse',
      'bitvect_list').
      Whether to return csr-sparse matrix, numpy array, or list of rdkit bit
      vectors.

  Examples:
    >>> from sklearn.pipeline import make_pipeline
    >>> from yamlchem.feature import MoleculeTransformer
    >>> molecule_tr = MoleculeTransformer()
    >>> ecfp_tr = ECFP(n_bits=1024)
    >>> pipe = make_pipeline(molecule_tr, ecfp_tr)

  References:
    D. Rogers and M. Hahn. (2010). Extended-connectivity fingerprints.
      Journal of chemical information and modeling 50(5):742–754.
  """

  def __init__(
      self,
      *,
      radius: int = 4,
      n_bits: int = 2048,
      return_type: str = 'ndarray',
  ):
    self.radius = radius
    self.n_bits = n_bits
    self.return_type = return_type

  # noinspection PyUnusedLocal
  # pylint: disable=unused-argument
  def fit(self, molecules: Sequence[Mol], _=None) -> 'ECFP':
    """Checks formal parameters' values.
    """
    check_scalar(self.radius, 'radius', int, min_val=1)
    check_scalar(self.n_bits, 'number of bits', int, min_val=1)

    valid_return_types = {'ndarray', 'csr_sparse', 'bitvect_list'}
    if self.return_type not in valid_return_types:
      raise ValueError(
          f'`return_type` must be in {valid_return_types}, '
          f'not {self.return_type!r}'
      )
    # noinspection PyAttributeOutsideInit
    self.n_features_in_ = 1

    return self

  def transform(self, molecules: Sequence[Mol]) \
      -> Union[np.array, sparse.csr_matrix]:
    """Return circular fingerprints as bit vectors.
    """
    fingerprints = [
        GetMorganFingerprintAsBitVect(molecule, self.radius, self.n_bits)
        for molecule in molecules
    ]

    if self.return_type == 'ndarray':
      return np.array(fingerprints, dtype=np.uint8)
    elif self.return_type == 'csr_sparse':
      return sparse.csr_matrix(fingerprints, dtype=np.uint8)
    elif self.return_type == 'bitvect_list':
      return fingerprints


MorganFingerprints = ECFP
