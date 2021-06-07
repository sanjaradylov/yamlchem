"""Load and process ESOL regression data.
"""

__all__ = (
    'ESOLDataset',
)

import functools
import pathlib
import pickle
from typing import Optional, Tuple

import dgl
import pandas as pd
from dgl.data import utils as dgl_data_utils

from ...feature.graph import AtomFeaturizer, get_canonical_atom_featurizers
from ...utils.graph import build_graph_from_molecule, from_smiles_to_molecule


class ESOLDataset(dgl.data.DGLDataset):
  """Processes, loads, serializes, and saves ESOL regression data.

  Args:
    url: (Optional). The URL to download data from.
    raw_dir: (Optional, defaults to '~/.dgl').
      The directory that will store/already stores the downloaded data.
    save_dir: (Optional, defaults to the value of `raw_dir`).
      The directory to save the processed data.
    force_reload: (Optional, defaults to False).
      Whether to reload data even if it is found on disk.
    verbose: (Optional, defaults to False).
      Whether to print log messages.

  Attributes:
    cache_filename: The file path to store serialized data.
    See the documentation of `dgl.data.DGLDataset`.

  References:
    John S. Delaney. ESOL: Estimating aqueous solubility directly from
    molecular structure. Journal of Chemical Information and Computer
    Sciences, 44(3):1000–1005, 2004.
  """

  # pylint: disable=line-too-long
  _url = 'https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/delaney-processed.csv'
  task_name = 'ESOL predicted log solubility in mols per litre'

  def __init__(
      self,
      *,
      url: Optional[str] = None,
      raw_dir: str = str(pathlib.Path.home() / '.dgl'),
      save_dir: Optional[str] = None,
      force_reload: bool = False,
      verbose: bool = False,
  ):
    # The pandas.Series of SMILES strings.
    self.smiles: Optional[pd.Series] = None
    # The pandas.Series of dgl.DGLGraph instances with node features.
    self.graphs: Optional[pd.Series] = None
    # The pandas.Series of outputs.
    self.labels: Optional[pd.Series] = None

    super().__init__(
        name='esol',
        url=url or self._url,
        raw_dir=raw_dir,
        save_dir=save_dir,
        force_reload=force_reload,
        verbose=verbose,
    )

  @property
  def cache_filename(self) -> pathlib.Path:
    """The file path to store serialized data.
    """
    return pathlib.Path(self.raw_path) / f'{self.name}.pkl'

  def __getitem__(self, index: int) -> Tuple[dgl.DGLGraph, pd.Series]:
    if index < len(self):
      return self.graphs[index], self.labels[index]
    else:
      raise IndexError(f'sample index {index!r} out of range')

  def __len__(self) -> int:
    return self.graphs.shape[0]

  def download(self):
    """Downloads data from `self.url` and saves in `self.raw_path`.
    """
    file_path = str(pathlib.Path(self.raw_path) / f'{self.name}.csv')
    dgl_data_utils.download(self.url, path=file_path, log=self.verbose)

  def process(self):
    """Loads .csv-file, converts SMILES strings to molecular graphs,
    applies canonical atom featurization, and saves the data in
    (`self.graphs`, `self.labels`).
    """
    filename = pathlib.Path(self.raw_path) / f'{self.name}.csv'
    data_frame = pd.read_csv(filename)

    node_featurizer = AtomFeaturizer(name='h')
    node_featurizer.add(*get_canonical_atom_featurizers())
    graph_constructor = functools.partial(
        build_graph_from_molecule, node_featurizer=node_featurizer)
    molecules = data_frame['smiles'].apply(from_smiles_to_molecule)

    self.smiles = data_frame['smiles']
    self.graphs = molecules.apply(graph_constructor)
    self.labels = data_frame[self.task_name]

  def save(self):
    """Serializes (`self.graphs`, `self.labels`) in `self.cache_filename`.
    """
    data = dict(graphs=self.graphs, labels=self.labels)
    # noinspection PyTypeChecker
    with open(self.cache_filename, 'wb') as fh:
      pickle.dump(data, fh, protocol=pickle.HIGHEST_PROTOCOL)

  def load(self):
    """Loads the serialized data from `self.cache_filename` to
    (`self.graphs`, `self.labels`).
    """
    # noinspection PyTypeChecker
    with open(self.cache_filename, 'rb') as fh:
      pickled_data = pickle.load(fh)
      self.graphs = pickled_data['graphs']
      self.labels = pickled_data['labels']

  def has_cache(self) -> bool:
    """Checks if cache exists.
    """
    return self.cache_filename.exists()
