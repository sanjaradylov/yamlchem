"""Train RNN language models.

Functions:
  train_rnn_lm: Trains SMILESRNN language model.
  train_rnn_predictor: Trains SMILESRNN-based model for supervised learning.
"""

__all__ = (
    'train_rnn_lm',
    'train_rnn_predictor',
)

from copy import copy
from datetime import timedelta
from math import ceil
from statistics import mean
from time import time
from typing import Callable, List, Optional

import mxnet as mx
from mxnet import autograd

from ..._types import ContextT, OptimizerT
from ..block.rnn import SMILESRNN, SMILESRNNPredictor


def train_rnn_lm(
    model: SMILESRNN,
    dataloader: mx.gluon.data.DataLoader,
    loss_fn: Optional[mx.gluon.loss.Loss] = None,
    n_epochs: int = 1,
    optimizer: OptimizerT = None,
    state_initializer: Callable[..., mx.nd.NDArray] = mx.nd.zeros,
    reinit_state: bool = False,
    detach_state: bool = True,
    verbose: int = 0,
    ctx: ContextT = None,
):
  """Trains and optionally evaluates the RNN language model.
  """
  ctx = ctx or mx.context.current_context()
  model.initialize(force_reinit=True, ctx=ctx)

  loss_fn = loss_fn or mx.gluon.loss.SoftmaxCELoss()
  if isinstance(optimizer, str):
    optimizer = mx.optimizer.create(optimizer)
  else:
    optimizer = optimizer or mx.optimizer.Adam()
  trainer = mx.gluon.Trainer(model.collect_params(), optimizer)
  metric = mx.metric.Perplexity(ignore_label=0)

  for epoch in range(1, n_epochs + 1):
    states: Optional[List[mx.nd.NDArray]] = None
    losses: List[float] = []
    perplexities: List[float] = []
    if metric is not None:
      metric.reset()
    start_time = time()

    for batch in dataloader:
      x = batch.sequence[:,:-1].as_in_context(ctx)
      y = batch.sequence[:, 1:].as_in_context(ctx)
      valid_lens = batch.valid_length.as_in_context(ctx)

      if states is None or reinit_state:
        states = model.rnn.begin_state(batch_size=x.shape[0],
                                       func=state_initializer)
      if detach_state:
        states = [state.detach() for state in states]

      with autograd.record():
        y_hat, states = model(x, states)
        mask = mx.nd.expand_dims(mx.nd.ones(shape=y.shape, ctx=ctx),
                                 axis=-1)
        mask = mx.nd.SequenceMask(mask, valid_lens,
                                  use_sequence_length=True, value=0, axis=1)
        loss = loss_fn(y_hat, y, mask)
      loss.backward()
      trainer.step(valid_lens.sum().asscalar())
      losses.append(loss.mean().asscalar())

      if verbose:
        metric.update(labels=y, preds=mx.nd.softmax(y_hat))
        perplexities.append(metric.get()[1])

    if verbose and (epoch % verbose == 0 or epoch == n_epochs):
      mean_loss = mean(losses)
      mean_perplexity = mean(perplexities)
      end_time = timedelta(seconds=ceil(time() - start_time))
      print(f'Epoch: {epoch:>2},  '
            f'Loss: {mean_loss:.3f},  '
            f'PPL: {mean_perplexity:.3f},  '
            f'Duration: {end_time}')


def train_rnn_predictor(
    model: SMILESRNNPredictor,
    dataloader: mx.gluon.data.DataLoader,
    loss_fn: mx.gluon.loss.Loss,
    optimizer: OptimizerT = None,
    metric: Optional[mx.metric.EvalMetric] = None,
    n_epochs: int = 1,
    state_initializer: Callable[..., mx.nd.NDArray] = mx.nd.zeros,
    reinit_state: bool = False,
    detach_state: bool = True,
    validation_fraction: float = 0.0,
    verbose: int = 0,
    ctx: ContextT = None,
):
  """Trains RNN-based supervised learning model.
  """
  ctx = ctx or mx.context.current_context()
  model.output.initialize(ctx=ctx)

  if isinstance(optimizer, str):
    optimizer = mx.optimizer.create(optimizer)
  else:
    optimizer = optimizer or mx.optimizer.Adam()
  trainer = mx.gluon.Trainer(model.collect_params(), optimizer)

  for epoch in range(1, n_epochs + 1):
    states: Optional[List[mx.nd.NDArray]] = None
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
      sequences = batch.sequence.as_in_context(ctx)
      valid_lens = batch.valid_length.as_in_context(ctx).astype(mx.np.float32)
      labels = batch.label.as_in_context(ctx)
      if states is None or reinit_state:
        states = model.rnn.begin_state(batch_size=sequences.shape[0],
                                       func=state_initializer)
      if detach_state:
        states = [state.detach() for state in states]

      if i < valid_data_index:
        with autograd.record():
          y_hat, states = model(sequences, states, valid_lens)
          loss = loss_fn(y_hat, labels)
        loss.backward()
        trainer.step(valid_lens.sum().asscalar())
        losses.append(loss.mean().asscalar())

        if metric is not None:
          metric.update(preds=y_hat, labels=labels.reshape_like(y_hat))
          scores.append(metric.get()[1])

      else:
        y_hat, states = model(sequences, states, valid_lens)
        valid_losses.append(loss_fn(y_hat, labels).mean().asscalar())

        if metric is not None:
          valid_metric.update(preds=y_hat, labels=labels.reshape_like(y_hat))
          valid_scores.append(valid_metric.get()[1])

    if verbose and (epoch % verbose == 0 or epoch == n_epochs):
      mean_loss = mean(losses)
      end_time = timedelta(seconds=ceil(time() - start_time))
      print(f'Epoch: {epoch:>2}, '
            f'Duration: {end_time}, '
            f'Train {loss_fn.name}: {mean_loss:.3f}',
            end='')

      if validation_fraction != 0.0:
        mean_valid_loss = mean(valid_losses)
        print(f', Valid {loss_fn.name}: {mean_valid_loss:.3f}', end='')

      if metric is not None:
        mean_score = mean(scores)
        mean_valid_score = mean(valid_scores)
        print(f', Train {metric.name}: {mean_score:.3f}'
              f', Valid {metric.name}: {mean_valid_score:.3f}')
      else:
        print('\n')
