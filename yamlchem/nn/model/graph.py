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
    validation_fraction: float = 0.2,
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
    validation_fraction: (Optional, defaults to 0.2).
      The fraction of `dataloader` to use for evaluation.
    verbose: (Optional, defaults to 0). Log verbosity level.
    ctx: (Optional, defaults to CPU). MXNet-compatible context.
  """
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

    n_batches = len(dataloader)
    valid_data_index = n_batches - int(n_batches * validation_fraction)
    valid_losses: List[float] = []
    valid_scores = []
    valid_metric = copy(metric)
    if valid_metric is not None:
      valid_metric.reset()

    for i, batch in enumerate(dataloader):
      g = batch.graph.to(ctx)
      h = g.ndata[feature_name].as_in_context(ctx)
      y = batch.label.as_in_context(ctx)
      if batch.mask is None:
        m = None
      else:
        m = batch.mask.as_in_context(ctx)

      if i < valid_data_index:
        with autograd.record():
          y_hat = gnn(g, h)
          loss = loss_fn(y_hat, y, m).mean()
        loss.backward()
        trainer.step(batch_size=1, ignore_stale_grad=True)
        losses.append(loss.asscalar())

        if metric is not None:
          metric.update(preds=y_hat, labels=y.reshape_like(y_hat))
          scores.append(metric.get()[1])

      else:
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
            f'T. {loss_fn.name}: {mean_loss:.3f}',
            end='')

      if validation_fraction != 0.0:
        mean_valid_loss = mean(valid_losses)
        print(f', V. {loss_fn.name}: {mean_valid_loss:.3f}', end='')

      if metric is not None:
        mean_score = mean(scores)
        mean_valid_score = mean(valid_scores)
        print(f', T. {metric.name}: {mean_score:.3f}'
              f', V. {metric.name}: {mean_valid_score:.3f}')
      else:
        print('\n')
