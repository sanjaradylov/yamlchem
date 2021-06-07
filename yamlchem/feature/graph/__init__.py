"""The :mod:`yamlchem.feature.graph` implements graph featurization
techniques.

ABC Classes:
  Featurizer: Abstract base class for featurization.
  CompoundFeaturizer: Abstract base class for the creation of feature spaces.

Classes:
  PropertyFeaturizer: Atom/Bond property calculator.
  AtomFeaturizer: Calculation of atomic features for every atom in a compound.
  BondFeaturizer: Calculation of bond features for every bond in a compound.
  OneHotFeaturizer: One-hot encoder of the properties of molecule constituents
    (atoms or bonds).
  OneHotEncoder: One-hot-encoder functor.

Functions:
  get_canonical_atom_featurizers: Get the sequence of canonical atom
    featurizers.
  get_canonical_bond_featurizers: Get the set of canonical bond featurizers.
  get_default_one_hot_encoder: Get the default atom-based one-hot encoding.

Constants:
  ATOMS: The set of valid atomic symbols.
  N_DEFAULT_ATOM_FEATURES: Canonical atom feature dimension.
  N_DEFAULT_BOND_FEATURES: Canonical bond feature dimension.
"""

__all__ = (
    'ATOMS',
    'AtomFeaturizer',
    'BondFeaturizer',
    'CompoundFeaturizer',
    'Featurizer',
    'get_canonical_atom_featurizers',
    'get_canonical_bond_featurizers',
    'get_default_one_hot_encoder',
    'N_DEFAULT_ATOM_FEATURES',
    'OneHotEncoder',
    'OneHotFeaturizer',
    'PropertyFeaturizer',
)

from ._atom import (
    AtomFeaturizer,
    ATOMS,
    get_canonical_atom_featurizers,
    get_default_one_hot_encoder,
    N_DEFAULT_ATOM_FEATURES,
)
from .base import (
    CompoundFeaturizer,
    Featurizer,
)
from ._bond import (
    BondFeaturizer,
    get_canonical_bond_featurizers,
)
from ._common import (
    OneHotEncoder,
    OneHotFeaturizer,
    PropertyFeaturizer,
)
