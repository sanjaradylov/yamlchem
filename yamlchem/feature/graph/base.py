"""Base featurization utilities.

Classes:
  Featurizer: Abstract base class for featurization.
  CompoundFeaturizer: Abstract base class for the creation of feature spaces.
"""

__all__ = (
    'CompoundFeaturizer',
    'Featurizer',
)


import abc
import functools
import itertools
import operator
import textwrap
from numbers import Real
from typing import Any, Dict, Iterable, List, Optional, Union

import mxnet as mx


_concat_lists = functools.partial(functools.reduce, operator.iconcat)


class Featurizer(metaclass=abc.ABCMeta):
  """An abstract base class for the featurization functors.

  Inherit from and implement the `featurize` method.

  Args:
    name: (Optional, defaults to the class name).
      The name of a featurizer.
  """

  def __init__(self, name: Optional[str] = None):
    self.__name = name or self.__class__.__name__

  @property
  def name(self) -> str:
    return self.__name

  def __repr__(self) -> str:
    return f'{self.__class__.__name__}({self.name})'

  def __eq__(self, other: 'Featurizer') -> bool:
    return self.name == other.name

  def __call__(self, sample: Any) -> List[Real]:
    """Returns a list of the calculated feature(s).

    Args:
      sample: The input data to featurize.
    """
    features: Union[Iterable[Real], Real] = self.featurize(sample)

    if isinstance(features, Real):
      features = [features]
    if not isinstance(features, list):
      features = list(features)

    return features

  @abc.abstractmethod
  def featurize(self, sample: Any) -> Union[Iterable[Real], Real]:
    """Calculate feature value(s).

    Args:
      sample: The input data to featurize.
    """


def _check_featurizer_type(featurizer: Featurizer):
  if not isinstance(featurizer, Featurizer):
    raise TypeError(
        f'featurizer must be of type Featurizer, not {type(featurizer)}')


class CompoundFeaturizer(Featurizer):
  """An abstract base class for the creation of feature spaces.

  Pass a set of featurizers to process and concatenate calculated features into
  a new feature space.

  Args:
    name: (Optional, defaults to the class name).
      The name of a feature space to create.
  """

  def __init__(self, name: Optional[str] = None):
    super().__init__(name=name)

    self._featurizers: List[Featurizer] = []

  def __repr__(self) -> str:
    featurizers_repr = ', '.join(map(lambda f: f.name, self._featurizers))
    featurizers_repr = f'({textwrap.shorten(featurizers_repr, 75)})'

    return (f'{self.__class__.__name__}(\n'
            f'    name={self.name!r},\n'
            f'    featurizers={featurizers_repr}\n)')

  def __getitem__(self, index: int) -> Featurizer:
    """Returns the featurizer with index `index`.

    Raises:
      IndexError: If the featurizer is not in the list.
    """
    try:
      return self._featurizers[index]
    except IndexError as err:
      err.args = ('featurizers index out of range',)
      raise

  def __setitem__(self, index: int, featurizer: Featurizer):
    """Insert `featurizer` in position `index`.

    Raises:
      IndexError: If `index` is out of range.
      TypeError: If `featurizer` is not of type `Featurizer`.
    """
    _check_featurizer_type(featurizer)

    try:
      self._featurizers[index] = featurizer
    except IndexError as err:
      err.args = ('featurizers assignment index out of range',)
      raise err

  def __delitem__(self, index: int):
    """Delete the featurizer in position `key`.

    Raises:
      IndexError: If `index` is out of range.
    """
    try:
      del self._featurizers[index]
    except IndexError as err:
      err.args = ('featurizers assignment index out of range',)
      raise err

  def __len__(self) -> int:
    """Returns the number of featurizers in the sequence.
    """
    return len(self._featurizers)

  def __contains__(self, featurizer: Featurizer) -> bool:
    """Checks if `featurizer` is in the sequence.
    """
    return featurizer in self._featurizers

  def add(self, featurizer: Featurizer, *featurizers: Featurizer):
    """Appends featurizer(s) to the featurizer composition.

    Raises:
      TypeError: If `featurizer` is not of type `Featurizer`.
    """
    featurizers_chain = list(itertools.chain((featurizer,), featurizers))

    for featurizer_ in featurizers_chain:
      _check_featurizer_type(featurizer_)

    self._featurizers.extend(featurizers_chain)

  def insert(self, index: int, featurizer: Featurizer):
    """Inserts `featurizer` in position `index`.

    Raises:
      TypeError: If `featurizer` is not of type `Featurizer`.
    """
    if index < len(self):
      self.__setitem__(index, featurizer)
    else:
      _check_featurizer_type(featurizer)
      self._featurizers.append(featurizer)

  def pop(self, index: int = -1) -> Featurizer:
    """Deletes the featurizer in position `index`.

    Raises:
      IndexError: If `index` is out of range.
    """
    try:
      return self._featurizers.pop(index)
    except IndexError as err:
      err.args = ('pop index out of range',)
      raise

  def __call__(self, compound: Any) -> Dict[str, mx.nd.NDArray]:
    """Calls `self.featurizers` on `sample` and stack them into a
    two-dimensional feature space.
    """
    feature_stack: List[List[Real]] = self.featurize(compound)
    feature_stack: mx.nd.NDArray = mx.nd.stack(
        *map(mx.nd.array, feature_stack))

    return {self.name: feature_stack}

  @abc.abstractmethod
  def featurize(self, compound: Any) -> List[List[Real]]:
    """Calls `self.featurizers` on every element of `compound`.

    Notes:
      Having two or more featurizers, you can use `self.concatenate` to
      flatten them into a single feature set.
    """

  def concatenate(self, compound: Any) -> List[Real]:
    """Concatenates the results of featurizations performed on `compound`.
    """
    return _concat_lists(f(compound) for f in self._featurizers)
