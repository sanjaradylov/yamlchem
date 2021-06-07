"""The :mod:`yamlchem.data.sets` module includes tools to download, process,
and serialize various data sets.

Classes:
  ESOLDataset: Load and process ESOL regression data.
  Tox21Dataset: Load and process Tox21 multi-task classification data.
"""

__all__ = (
    'ESOLDataset',
    'Tox21Dataset',
)

from ._esol import ESOLDataset
from ._tox21 import Tox21Dataset
