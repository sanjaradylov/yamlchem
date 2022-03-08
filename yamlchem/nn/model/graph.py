"""The :mod:`yamlchem.nn.model.graph` implements utilities to train GNN
predictors.
"""

__all__ = (
    'train_gnn_predictor',
)

from copy import copy
from datetime import timedelta
from math import ceil
from statistics import mean
from time import time
from typing import List, Optional

import mxnet as mx
from mxnet import autograd

from ..._types import ContextT, OptimizerT


def train_gnn_predictor(
    gnn: mx.gluon.Block,
    feature_name: str,
    dataloader: mx.gluon.data.DataLoader,
    loss_fn: mx.gluon.loss.Loss,
    n_epochs: int = 1,
    optimizer: OptimizerT = None,
    metric: Optional[mx.metric.EvalMetric] = None,
    valid_dataloader: Optional[mx.gluon.data.DataLoader] = None,
    verbose: int = 0,
    ctx: ContextT = None,
):
  """Train and optionally evaluate the model.

  Args:
    gnn: Graph neural network.
    feature_name: The name of graph node features.
    dataloader: Batch data loader.
    loss_fn: Loss function to minimize.
    n_epochs: (Optional, defaults to 1).
      The number of epochs to repeat training.
    optimizer: (Optional, defaults to SGD).
      MXNet-compatible optimizer or string.
    metric: (Optional). Validation metric.
    valid_dataloader: (Optional).
      Validation data loader to use for evaluation.
    verbose: (Optional, defaults to 0). Log verbosity level.
    ctx: (Optional, defaults to CPU). MXNet-compatible context.
  """

  def prepare_batch():
    """Get batch data and change context.
    """
    graph = batch.graph.to(ctx)
    features = graph.ndata[feature_name].as_in_context(ctx)
    labels = batch.label.as_in_context(ctx)
    if batch.mask is None:
      masks = None
    else:
      masks = batch.mask.as_in_context(ctx)

    return graph, features, labels, masks

  ctx = ctx or mx.context.current_context()
  gnn.initialize(ctx=ctx)

  if isinstance(optimizer, str):
    optimizer = mx.optimizer.create(optimizer)
  else:
    optimizer = optimizer or mx.optimizer.Adam()
  trainer = mx.gluon.Trainer(gnn.collect_params(), optimizer)

  for epoch in range(1, n_epochs + 1):
    losses: List[float] = []
    scores = []
    if metric is not None:
      metric.reset()
    start_time = time()

    valid_losses: List[float] = []
    valid_scores = []
    valid_metric = copy(metric)
    if valid_metric is not None:
      valid_metric.reset()

    for batch in dataloader:
      g, h, y, m = prepare_batch()

      with autograd.record():
        y_hat = gnn(g, h)
        loss = loss_fn(y_hat, y, m).mean()
      loss.backward()
      trainer.step(batch_size=1, ignore_stale_grad=True)
      losses.append(loss.asscalar())

      if metric is not None:
        metric.update(preds=y_hat, labels=y.reshape_like(y_hat))
        scores.append(metric.get()[1])

    if valid_dataloader is not None:
      for batch in valid_dataloader:
        g, h, y, m = prepare_batch()
        y_hat = gnn(g, h)
        valid_losses.append(loss_fn(y_hat, y, m).mean().asscalar())

        if metric is not None:
          valid_metric.update(preds=y_hat, labels=y.reshape_like(y_hat))
          valid_scores.append(valid_metric.get()[1])

    if verbose and (epoch % verbose == 0 or epoch == n_epochs):
      mean_loss = mean(losses)
      end_time = timedelta(seconds=ceil(time() - start_time))
      print(f'Epoch: {epoch:>3}, '
            f'Time: {end_time}, '
            f'T.{loss_fn.name}: {mean_loss:.3f}',
            end='')

      if valid_dataloader is not None:
        mean_valid_loss = mean(valid_losses)
        print(f', V.{loss_fn.name}: {mean_valid_loss:.3f}', end='')

      if metric is not None:
        mean_score = mean(scores)
        print(f', T.{metric.name}: {mean_score:.3f}', end='')
        if valid_dataloader is not None:
          mean_valid_score = mean(valid_scores)
          print(f', V.{metric.name}: {mean_valid_score:.3f}')
        else:
          print()
      else:
        print()
