"""Common featurization utilities not specific to a particular molecular
constituent.

Classes:
  OneHotEncoder: One-hot-encoder functor.
  OneHotFeaturizer: One-hot encoder of the properties of molecule constituents
    (atoms or bonds).
  PropertyFeaturizer: Atom/Bond property calculator.
"""

__all__ = (
    'OneHotEncoder',
    'OneHotFeaturizer',
    'PropertyFeaturizer',
)

import textwrap
from numbers import Real
from typing import List, Optional, Sequence, Union

from rdkit.Chem import rdchem

from .base import Featurizer
from ..._types import LabelT


class OneHotEncoder:
  """One-hot encoder functor.

  Args:
    valid_features: Feature labels.
    encode_unknown: (Optional, defaults to False).
      Whether to encode an unknown specified label during encoding.

  Call args:
    labels: The label/category to encode.

  Call returns:
    One-hot encoded boolean list.

  Examples:
    >>> valid_features = list('CHS')
    >>> known_label, unknown_label = 'C', 'Li'
    >>> encoder = OneHotEncoder(valid_features, encode_unknown=False)
    >>> encoder(known_label)
    [True, False, False]
    >>> encoder(unknown_label)
    [False, False, False]
    >>> encoder.encode_unknown = True
    >>> encoder(unknown_label)
    [False, False, False, True]

  See also:
    `yamlchem.feature.graph.OneHotFeaturizer`
  """

  __slots__ = (
      'valid_features',
      'encode_unknown',
  )

  def __init__(
      self,
      valid_features: Sequence[LabelT],
      encode_unknown: bool = False,
  ):
    self.valid_features = valid_features
    self.encode_unknown = encode_unknown

  def __repr__(self) -> str:
    features_repr = textwrap.shorten(repr(self.valid_features), 75)

    return (f'{self.__class__.__name__}(\n'
            f'    valid_features={features_repr},\n'
            f'    encode_unknown={self.encode_unknown!r}\n)')

  def __call__(self, label: LabelT) -> List[bool]:
    """Returns the one-hot encoded vector.

    Args:
      label: The label/category to encode. Expected to be bool, int, or str.

    Returns:
      Encoded boolean vector.
    """
    encoded_list = [label == feature for feature in self.valid_features]

    if self.encode_unknown and label not in self.valid_features:
      encoded_list.append(True)

    return encoded_list


class PropertyFeaturizer(Featurizer):
  """Calculation of a property `attribute_name` of an atom or bond.

  Args:
    attribute: Method name of the `rdkit.Chem.rdbase.Atom`/`Bond` getter, e.g.
      'GetIsAromatic'. Method call must return objects of type `numbers.Real`
      (int, bool, ...).
    name: (Optional, defaults to the class name). The name of a featurizer.

  Raises:
    AttributeError: If `rdkit.Chem.rdchem.Atom`/`Bond` instance has no
      attribute/method `attribute_name`.

  Examples:
    >>> from rdkit.Chem import MolFromSmiles
    >>> molecule = MolFromSmiles('CCO')
    >>> atom = molecule.GetAtomWithIdx(0)
    >>> featurizer = PropertyFeaturizer('GetTotalNumHs')
    >>> featurizer(atom)
    [3]
  """

  def __init__(self, attribute: str, name: Optional[str] = None):
    super().__init__(name=name)

    self.__attribute = attribute

  @property
  def attribute(self) -> str:
    return self.__attribute

  def __repr__(self) -> str:
    return (f'{self.__class__.__name__}(\n'
            f'    name={self.name!r},\n'
            f'    attribute={self.attribute!r}\n)')

  def __eq__(self, other: 'PropertyFeaturizer') -> bool:
    return self.name == other.name and self.attribute == other.attribute

  def featurize(self, element: Union[rdchem.Atom, rdchem.Bond]) -> Real:
    """Calculates an atomic property.

    Args:
      element: RDKit atom or bond.
    """
    return getattr(element, self.attribute)()


class OneHotFeaturizer(PropertyFeaturizer):
  """One-hot encoding of a property `attribute_name` of an atom or bond.

  Args:
    encoder: One-hot encoder instance.
    attribute: Method name of rdkit.Chem.rdbase.Atom, e.g. 'GetSymbol'.
    name: (Optional, defaults to 'h'). The name of a featurizer.

  Examples:
    >>> from rdkit.Chem import MolFromSmiles
    >>> encoder = OneHotEncoder(['C', 'O', 'H'])
    >>> attribute = 'GetSymbol'
    >>> featurizer = OneHotFeaturizer(encoder, attribute)
    >>> molecule = MolFromSmiles('CCO')
    >>> atom = molecule.GetAtomWithIdx(0)
    >>> featurizer(atom)
    [True, False, False]

  See also:
    `yamlchem.feature.OneHotEncoder`
  """

  def __init__(
      self,
      encoder: OneHotEncoder,
      attribute: str,
      name: Optional[str] = None,
  ):
    super().__init__(attribute=attribute, name=name)

    self._encoder = encoder

  @property
  def encoder(self) -> OneHotEncoder:
    return self._encoder

  def __repr__(self) -> str:
    encoder_repr = textwrap.indent(repr(self.encoder), ' '*4).lstrip()

    return (f'{self.__class__.__name__}(\n'
            f'    name={self.name!r},\n'
            f'    attribute={self.attribute!r},\n'
            f'    encoder={encoder_repr}\n)')

  def featurize(self, element: Union[rdchem.Atom, rdchem.Bond]) -> List[bool]:
    """Returns a one-hot encoded vector of an atom property.

    Args:
      element: RDKit atom or bond.
    """
    return self.encoder(super().featurize(element))
