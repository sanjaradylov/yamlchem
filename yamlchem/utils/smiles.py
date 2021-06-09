"""The :mod:`yamlchem.utils.smiles` implements utils to process SMILES strings.

Classes:
  Tokenizer: Functor to tokenize SMILES strings.
  Vocabulary: Calculates token stats and maps tokens into unique integer IDs.
"""

__all__ = (
    'Tokenizer',
    'Vocabulary',
)

import re
from collections import Counter
from collections.abc import Collection
from dataclasses import dataclass
from itertools import chain
from operator import itemgetter
from typing import (Callable, Dict, FrozenSet, Generator, Iterable, List,
                    Optional, Sequence, Tuple)


@dataclass(init=True, repr=True, eq=True, frozen=False)
class Tokenizer:
  """Tokenizer functor containing sets of valid SMILES symbols grouped by rule
  class (atoms, non-atomic symbols like bonds and branches, and special
  symbols like beginning-of-SMILES and padding).

  Args:
    match_bracket_atoms: (Optional, defaults to False).
      Whether to treat the subcompounds enclosed in [] as separate tokens.

  References:
    Adilov, Sanjar (2021): Neural Language Modeling for Molecule Generation.
    ChemRxiv. Preprint. https://doi.org/10.26434/chemrxiv.14700831.v1
  """

  match_bracket_atoms: bool = False

  # Atomic symbols. (We store the original ones, although lowercase symbols
  # should also be considered during tokenization).
  atoms = frozenset([
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

  # Bonds, charges, etc.
  non_atoms = frozenset([
      '-', '=', '#', ':', '(', ')', '.', '[', ']', '+', '-', '\\', '/',
      '1', '2', '3', '4', '5', '6', '7', '8', '9',
      '@', 'AL', 'TH', 'SP', 'TB', 'OH',
  ])
  _non_atom_max_len = max(map(len, non_atoms))

  # Special tokens not presented in the SMILES vocabulary.
  bos = '{'  # Beginning of SMILES.
  eos = '}'  # End of SMILES.
  pad = '_'  # Padding.
  unk = '*'  # Unknown.
  special = frozenset(bos + eos + pad + unk)

  @classmethod
  def get_all_tokens(cls) -> FrozenSet[str]:
    """Returns a set of all the valid tokens defined in the class.
    """
    return cls.special.union(cls.non_atoms.union(cls.atoms))

  def __call__(self, smiles: str) -> List[str]:
    """Tokenizes `smiles` string.
    """
    if self.match_bracket_atoms:
      token_list: List[str] = []
      for subcompound in self._brackets_re.split(smiles):
        if subcompound.startswith('['):
          token_list.append(subcompound)
        else:
          token_list.extend(self._tokenize(subcompound))
      return token_list
    else:
      return self._tokenize(smiles)

  _brackets_re = re.compile(
      pattern=r"""
          (?P<brackets>     # Begin a capture group.
              \[            # Match opening bracket square to capture an atom.
                  [^\[\]]+  # Match atoms, charges, etc., except '[' and ']'.
              \]            # Match closing bracket square to capture an atom.
          )                 # End a capture group.
      """,
      flags=re.VERBOSE,
  )
  _digits_re = re.compile(r'(?P<digits>\d{2,}).*')

  @classmethod
  def _tokenize(cls, smiles: str) -> List[str]:
    token_list: List[str] = []  # The resulting list of tokens.

    char_no = 0  # Points to the current position.
    while char_no < len(smiles):
      # Check if tokens of length `n_chars` are in our `smiles`.
      for n_chars in range(cls._non_atom_max_len, 1, -1):
        token = smiles[char_no:char_no + n_chars]
        if token in cls.non_atoms:
          token_list.append(token)
          char_no += n_chars
          break
      else:
        # If not, then try processing single- and double-char tokens.
        one_char_token = smiles[char_no]
        two_char_token = smiles[char_no:char_no + 2]
        four_char_token = smiles[char_no:char_no + 4]
        if (
            # Double-char atoms like '[se]'.
            four_char_token.startswith('[')
            and four_char_token.endswith(']')
            and four_char_token[1:].islower()
            and four_char_token[1:-1].title() in cls.atoms
        ):
          token_list.append(four_char_token)
          char_no += 4
        elif (
            # Double-char token that cannot be represented as
            # two separate atoms; 'no' will be treated as two
            # single-char tokens 'n' and 'o', while 'Se' or 'Na' as
            # double-char.
            two_char_token.title() in cls.atoms
            and two_char_token[-1].title() not in cls.atoms
            or
            two_char_token[0].isupper()
            and two_char_token in cls.atoms
        ):
          token_list.append(two_char_token)
          char_no += 2
        elif (
            one_char_token.title() in cls.atoms  # n, o, etc.;
            or one_char_token in cls.non_atoms   # -, #, \., etc.;
            or one_char_token in cls.special     # {, }, _, *.
        ):
          token_list.append(one_char_token)
          char_no += 1
        elif one_char_token.startswith('%'):  # Digits > 9 like %10 or %108.
          match = cls._digits_re.match(smiles, char_no + 1)
          if match is not None:
            tokens = f'%{match.group("digits")}'
            token_list.append(tokens)
            char_no += len(tokens)
          else:
            token_list.append(cls.unk)
            char_no += 1
        # If we didn't find any valid token, append the unknown token.
        else:
          token_list.append(cls.unk)
          char_no += 1

    return token_list


class Vocabulary(Collection):
  """Calculates token statistics and maps tokens into unique integer IDs.

    Args:
      dataset: A sequence of SMILES strings.
      tokenizer: Tokenization mapping (e.g. `yamlchem.utils.smiles.Tokenizer`).
      allowed_tokens: (Optional, defaults to frozenset()).
        The set of allowed tokens. By default include all tokens from `dataset`
      prohibited_tokens: (Optional, defaults to frozenset()).
        The set of prohibited tokens. By default (empty frozenset), include
        either all tokens or `allowed_tokens`.
      min_count: (Optional, defaults to 1)
        The minimum number of token occurrences to consider.
      max_size: (Optional, defaults to None).
        The maximum dimension of a vocabulary.

    Attributes:
      id_to_token: The list of unique tokens.
      token_to_id: The token-to-ID mapping.
      tokenizer: The original tokenizer.
      bos, eos, unk, pad: Special tokens.

  Examples:
    >>> tokenizer = Tokenizer()
    >>> dataset = ['CC(=O)C', 'C#N', 'N#N', 'CCOF']
    >>> vocabulary = Vocabulary(tokenizer=tokenizer, dataset=dataset)
    >>> len(vocabulary)
    8
  """

  def __init__(
      self,
      *,
      tokenizer: Callable[[str], List[str]],
      dataset: Sequence[str],
      allowed_tokens: Sequence[str] = frozenset(),
      prohibited_tokens: Sequence[str] = frozenset(),
      min_count: int = 1,
      max_size: Optional[int] = None,
  ):
    token_iter: Iterable[List[str]] = map(tokenizer, dataset)
    token_counter = Counter(chain(*token_iter))

    if min_count > 1:
      redundant_tokens: FrozenSet[str] = frozenset(
          t for t, c in token_counter.items()
          if c < min_count)
    elif max_size is not None:
      sorted_counter: List[Tuple[str, int]] = sorted(
          token_counter.items(), key=itemgetter(1))
      redundant_tokens: FrozenSet[str] = frozenset(
          t for t, c in sorted_counter[:-max_size])  # pylint: disable=invalid-unary-operand-type
    else:
      redundant_tokens = frozenset()

    allowed_tokens = frozenset(allowed_tokens) \
                     or frozenset(token_counter.keys())
    allowed_tokens -= redundant_tokens
    all_tokens = [t for t in token_counter.keys() if t in allowed_tokens]
    prohibited_tokens = frozenset(prohibited_tokens) | redundant_tokens
    all_tokens = [t for t in all_tokens if t not in prohibited_tokens]

    self._pad, self._unk, self._bos, self._eos = range(4)
    self._id_to_token: List[str] = [Tokenizer.pad, Tokenizer.unk,
                                    Tokenizer.bos, Tokenizer.eos] + all_tokens
    self._token_to_id: Dict[str, int] = {
        t: i for i, t in enumerate(self._id_to_token)}
    self._tokenizer = tokenizer

  @property
  def id_to_token(self) -> List[str]:
    """Returns a list of unique tokens.
    """
    return self._id_to_token

  @property
  def token_to_id(self) -> Dict[str, int]:
    """Maps tokens into unique IDs.
    """
    return self._token_to_id

  @property
  def tokenizer(self):
    """Returns the original tokenizer.
    """
    return self._tokenizer

  @property
  def bos(self) -> int:
    """Beginning-of-SMILES token.
    """
    return self._bos

  @property
  def eos(self) -> int:
    """End-of-SMILES token.
    """
    return self._eos

  @property
  def unk(self) -> int:
    """Unknown token.
    """
    return self._unk

  @property
  def pad(self) -> int:
    """Padding token.
    """
    return self._pad

  def __repr__(self):
    return (f'{self.__class__.__name__}(dim={len(self)}, '
            f'bos={self._bos}, eos={self._eos}, '
            f'unk={self._unk}, pad={self._pad})')

  def __len__(self):
    """Returns the number of unique tokens.
    """
    return len(self._id_to_token)

  def __contains__(self, token) -> bool:
    """Checks if `token` is in the vocabulary.
    """
    return token in self._id_to_token

  def __iter__(self) -> Generator[str, None, None]:
    """"Generates the tokens from the vocabulary.
    """
    return (t for t in self._id_to_token)

  def encode(self, smiles: str) -> List[int]:
    """Maps `smiles` into token IDs.
    """
    return [self._token_to_id.get(token, self.unk)
            for token in self._tokenizer(smiles)]

  def decode(self, token_idx: List[int]) -> str:
    """Maps `token_idx` into a SMILES string.
    """
    return ''.join(self._id_to_token[i] for i in token_idx)

  __call__ = encode
