"""Common data type annotations used in the project.
"""

from typing import Callable, Hashable, List, TypeVar, Union

from mxnet import context, gluon, nd, optimizer


ActivationT = Union[
    None, str, gluon.nn.Activation, Callable[[nd.NDArray], nd.NDArray]]
ContextT = Union[context.Context, List[context.Context]]
LabelT = TypeVar('LabelT', bound=Hashable)
OptimizerT = Union[None, optimizer.Optimizer, str]
