"""
Tensor Contraction Layers
"""

# Author: Jean Kossaifi
# License: BSD 3 clause

import math

import paddle
import paddle.nn as nn
import tensorly as tl
from tensorly import tenalg

tl.set_backend("paddle")


class TCL(nn.Layer):
    """Tensor Contraction Layer [1]_

    Parameters
    ----------
    input_size : int iterable
        shape of the input, excluding batch size
    rank : int list or int
        rank of the TCL, will also be the output-shape (excluding batch-size)
        if int, the same rank will be used for all dimensions
    verbose : int, default is 1
        level of verbosity

    References
    ----------
    .. [1] J. Kossaifi, A. Khanna, Z. Lipton, T. Furlanello and A. Anandkumar,
            "Tensor Contraction Layers for Parsimonious Deep Nets," 2017 IEEE Conference on Computer Vision and Pattern Recognition Workshops (CVPRW),
            Honolulu, HI, 2017, pp. 1940-1946, doi: 10.1109/CVPRW.2017.243.
    """

    def __init__(
        self,
        input_shape,
        rank,
        verbose=0,
        bias=False,
        device=None,
        dtype=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.verbose = verbose

        if isinstance(input_shape, int):
            self.input_shape = (input_shape,)
        else:
            self.input_shape = tuple(input_shape)

        self.order = len(input_shape)

        if isinstance(rank, int):
            self.rank = (rank,) * self.order
        else:
            self.rank = tuple(rank)

        # Start at 1 as the batch-size is not projected
        self.contraction_modes = list(range(1, self.order + 1))
        for i, (s, r) in enumerate(zip(self.input_shape, self.rank)):
            self.register_parameter(
                f"factor_{i}",
                paddle.base.framework.EagerParamBase.from_tensor(
                    paddle.empty((r, s), dtype=dtype)
                ),
            )

        # self.factors = ParameterList(parameters=factors)
        if bias:
            self.bias = paddle.base.framework.EagerParamBase.from_tensor(
                paddle.empty(self.output_shape, dtype=dtype), requires_grad=True
            )
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    @property
    def factors(self):
        return [getattr(self, f"factor_{i}") for i in range(self.order)]

    def forward(self, x):
        """Performs a forward pass"""
        x = tenalg.multi_mode_dot(x, self.factors, modes=self.contraction_modes)

        if self.bias is not None:
            return x + self.bias
        else:
            return x

    def reset_parameters(self):
        """Sets the parameters' values randomly

        Todo
        ----
        This may be renamed to init_from_random for consistency with TensorModules
        """
        for i in range(self.order):
            init_kaimingUniform = paddle.nn.initializer.KaimingUniform(
                negative_slope=math.sqrt(5), nonlinearity="leaky_relu"
            )
            init_kaimingUniform(getattr(self, f"factor_{i}"))
        if self.bias is not None:
            bound = 1 / math.sqrt(self.input_shape[0])
            init_uniform = paddle.nn.initializer.Uniform(low=-bound, high=bound)
            init_uniform(self.bias)
