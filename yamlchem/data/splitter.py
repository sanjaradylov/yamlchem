"""The :mod:`yamlchem.data.splitter` module implements utilities to split the
data into training and validation/test data.

Classes:
  KFoldSplitter: K-fold cross-validation split.

Functions:
  train_test_split
  train_valid_test_split
"""

__all__ = (
    'KFoldSplitter',
    'train_test_split',
    'train_valid_test_split',
)

import array
import itertools
import math
import random
from typing import Generator, Tuple

from mxnet.gluon.data import ArrayDataset, Dataset


def train_test_split(
    dataset: Dataset,
    test_ratio: float = 0.1,
    shuffle: bool = True,
    use_same_dataset_class: bool = True,
) -> Tuple[Dataset, Dataset]:
  """Split `dataset` into training and test sets.

  Args:
    dataset: Gluon dataset instance.
    test_ratio: (Optional, defaults to 0.1). The ratio of test samples.
    shuffle: (Optional, defaults to True).
      Whether to shuffle indices before splitting.
    use_same_dataset_class: (Optional, defaults to True).
      Whether to return instances of the same class as `dataset`. Otherwise,
      return two `mxnet.gluon.data.ArrayDataset`s.

  Returns:
    Two disjoint Gluon datasets.

  Examples:
    >>> from mxnet.gluon.data import ArrayDataset
    >>> from yamlchem.data.sets import ESOLDataset
    >>> data = ArrayDataset(ESOLDataset())
    >>> train_data, test_data = train_test_split(data)
    >>> len(train_data) / len(data), len(test_data) / len(data)
    (0.9, 0.1)
  """
  if use_same_dataset_class:
    cls = dataset.__class__
  else:
    cls = ArrayDataset

  n_samples = len(dataset)
  n_train = math.floor((1 - test_ratio) * n_samples)

  if not shuffle:
    train_dataset = cls([dataset[i] for i in range(n_train)])
    test_dataset = cls([dataset[i] for i in range(n_train, n_samples)])
  else:
    idx = array.array('L', list(range(n_samples)))
    random.shuffle(idx)
    train_dataset = cls([dataset[i] for i in idx[:n_train]])
    test_dataset = cls([dataset[i] for i in idx[n_train:n_samples]])

  return train_dataset, test_dataset


def train_valid_test_split(
    dataset,
    valid_ratio: float = 0.1,
    test_ratio: float = 0.1,
    shuffle: bool = True,
    use_same_dataset_class: bool = True,
) -> Tuple[Dataset, ...]:
  """Split `dataset` into training and test sets.

  Args:
    dataset: Gluon dataset instance.
    valid_ratio: (Optional, defaults to 0.1). The ratio of validation samples.
    test_ratio: (Optional, defaults to 0.1). The ratio of test samples.
    shuffle: (Optional, defaults to True).
      Whether to shuffle indices before splitting.
    use_same_dataset_class: (Optional, defaults to True).
      Whether to return instances of the same class as `dataset`. Otherwise,
      return two `mxnet.gluon.data.ArrayDataset`s.

  Returns:
    Three disjoint Gluon datasets.

  Examples:
    >>> from mxnet.gluon.data import ArrayDataset
    >>> from yamlchem.data.sets import ESOLDataset
    >>> data = ArrayDataset(ESOLDataset())
    >>> train_data, valid_data, test_data = train_valid_test_split(
    ...     data, valid_ratio=0.2, test_ratio=0.1)
    >>> n = len(data)
    >>> len(train_data) / n, len(valid_data) / n, len(test_data) / n
    (0.7, 0.2, 0.1)
  """
  train_dataset, test_dataset = train_test_split(
      dataset, test_ratio, shuffle, use_same_dataset_class)
  train_dataset, valid_dataset = train_test_split(
      train_dataset, valid_ratio / (1 - test_ratio),
      shuffle, use_same_dataset_class)
  return train_dataset, valid_dataset, test_dataset


class KFoldSplitter:
  """K-fold cross-validation splitter.

  Args:
    n_folds: (Optional, defaults to 3). The number of cv blocks.

  Examples:
    >>> from mxnet.gluon.data import ArrayDataset
    >>> data = ArrayDataset(list(range(6)))
    >>> splitter = KFoldSplitter(n_folds=3)
    >>> for train_data, valid_data in splitter.split(data):
    ...     print(len(train_data), len(valid_data))
    ...     print(list(train_data), list(valid_data))
    4, 2
    [2, 3, 4, 5], [0, 1]
    4, 2
    [0, 1, 4, 5], [2, 3]
    4, 2
    [0, 1, 2, 3], [4, 5]
  """

  def __init__(self, n_folds: int = 3):
    self.n_folds = n_folds

  def __repr__(self):
    return f'{self.__class__.__name__}(n_folds={self.n_folds})'

  def __eq__(self, other) -> bool:
    return self.n_folds == other.n_folds

  @property
  def n_folds(self) -> int:
    """The number of blocks.
    """
    return self.__n_folds

  @n_folds.setter
  def n_folds(self, n_folds: int):
    """Set a new number of blocks.
    """
    if not isinstance(n_folds, int):
      raise TypeError(f'`n_folds` must be int, not {type(n_folds)}')
    if n_folds < 2:
      raise ValueError(f'`n_folds` must be at least 2, not {n_folds}')
    self.__n_folds = n_folds

  @property
  def training_ratio(self) -> float:
    """The ratio of training samples.
    """
    return (self.n_folds - 1) / self.n_folds

  @property
  def validation_ratio(self) -> float:
    """The ratio of validation samples.
    """
    return 1 / self.n_folds

  def split(self, dataset: Dataset) \
      -> Generator[Tuple[ArrayDataset, ArrayDataset], None, None]:
    """Generate two disjoint datasets `self.n_folds` times.

    Args:
      dataset: Gluon-compatible dataset.

    Yields:
      Tuple of two mxnet.gluon.data.ArrayDataset instances.
    """
    for train_range, valid_range in self._get_idx(dataset):
      yield (ArrayDataset([dataset[k] for k in train_range]),
             ArrayDataset([dataset[k] for k in valid_range]))

  def _get_idx(self, dataset: Dataset) \
      -> Generator[Tuple[range, range], None, None]:
    n_samples = len(dataset)
    n_fold_samples = int(len(dataset) / self.n_folds)

    for k in range(self.n_folds):
      current_fold_i = k * n_fold_samples

      valid_range = range(current_fold_i, current_fold_i + n_fold_samples)
      train_range_l = range(min(valid_range.start, 0), valid_range.start)
      train_range_r = range(valid_range.stop, max(valid_range.stop, n_samples))

      yield itertools.chain(train_range_l, train_range_r), valid_range
