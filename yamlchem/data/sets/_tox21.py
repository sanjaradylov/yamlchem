"""Load and process Tox21 multi-task classification data.
"""

__all__ = (
    'Tox21Dataset',
)

import functools
import gzip
import pathlib
import pickle
import shutil
from typing import List, Optional, Tuple

import dgl
import pandas as pd
from dgl.data import utils as dgl_data_utils

from ...feature.graph import AtomFeaturizer, get_canonical_atom_featurizers
from ...utils.graph import build_graph_from_molecule, from_smiles_to_molecule


class Tox21Dataset(dgl.data.DGLDataset):
  """Processes, loads, serializes, and save Tox21 multi-task classification
  data.

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
  """

  # pylint: disable=line-too-long
  _url = 'http://deepchem.io.s3-website-us-west-1.amazonaws.com/datasets/tox21.csv.gz'
  # Task (target) names.
  tasks: List[str] = [
      'NR-AR', 'NR-AR-LBD', 'NR-AhR', 'NR-Aromatase', 'NR-ER', 'NR-ER-LBD',
      'NR-PPAR-gamma', 'SR-ARE', 'SR-ATAD5', 'SR-HSE', 'SR-MMP', 'SR-p53',
  ]

  def __init__(
      self,
      *,
      url: Optional[str] = None,
      raw_dir: Optional[str] = None,
      save_dir: Optional[str] = None,
      force_reload: bool = False,
      verbose: bool = False,
  ):
    # The pandas.Series of SMILES strings.
    self.smiles: Optional[pd.Series] = None
    # The pandas.Series of dgl.DGLGraph instances with node features.
    self.graphs: Optional[pd.Series] = None
    # The pandas.DataFrame of 12 tasks.
    self.labels: Optional[pd.DataFrame] = None
    # The pandas.DataFrame of task weights (0 if task value is empty,
    # 1 if presented).
    self.masks: Optional[pd.DataFrame] = None

    super().__init__(
        name='tox21',
        url=url or self._url,
        raw_dir=raw_dir or str(pathlib.Path.home() / '.dgl'),
        save_dir=save_dir,
        force_reload=force_reload,
        verbose=verbose,
    )

  @property
  def cache_filename(self) -> pathlib.Path:
    """The file path to store serialized data.
    """
    return pathlib.Path(self.raw_path) / f'{self.name}.pkl'

  def __len__(self) -> int:
    """Returns the number of graphs.
    """
    return self.graphs.shape[0]

  def __getitem__(self, index: int) \
      -> Tuple[dgl.DGLGraph, pd.Series,pd.Series]:
    """Returns a graph, labels, and a label mask in the position `index`.

    Returns:
      graph: dgl.DGLGraph
      labels: pandas.Series, shape = (12,)
      mask: pandas.Series, shape = (12,)
    """
    if index < len(self):
      return (
          self.graphs[index],
          self.labels.iloc[index],
          self.masks.iloc[index],
      )
    else:
      raise IndexError(f'sample index {index!r} out of range')

  def download(self):
    """Downloads Tox21 data from `self.url` and saves in `self.raw_path`.
    """
    gz_filename = f'{self.name}.csv.gz'
    gz_file_path = str(pathlib.Path(self.raw_path) / gz_filename)

    dgl_data_utils.download(self.url, path=gz_file_path, log=self.verbose)

    with gzip.open(gz_file_path, 'rb') as gz_fh_in:
      with open(gz_file_path.rstrip('.gz'), 'wb') as fh_out:
        shutil.copyfileobj(gz_fh_in, fh_out)

  def process(self):
    """Loads Tox21 .csv-file, converts SMILES strings to molecular graphs,
    applies canonical atom featurization, and saves
    (`self.graphs`, `self.labels`, `self.masks`).
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
    self.labels = data_frame[self.tasks]
    self.masks = ~self.labels.isna()
    self.labels = self.labels.fillna(-1)

  def save(self):
    """Serializes (`self.graphs`, `self.labels`, `self.masks`) in
    `self.cache_filename`.
    """
    data = dict(graphs=self.graphs, labels=self.labels, masks=self.masks)
    # noinspection PyTypeChecker
    with open(self.cache_filename, 'wb') as fh:
      pickle.dump(data, fh, protocol=pickle.HIGHEST_PROTOCOL)

  def load(self):
    """Loads the serialized data from `self.cache_filename` to
    (`self.graphs`, `self.labels`, `self.masks`).
    """
    # noinspection PyTypeChecker
    with open(self.cache_filename, 'rb') as fh:
      pickled_data = pickle.load(fh)
      self.graphs = pickled_data['graphs']
      self.labels = pickled_data['labels']
      self.masks = pickled_data['masks']

  def has_cache(self) -> bool:
    """Checks if cache exists.
    """
    return self.cache_filename.exists()
