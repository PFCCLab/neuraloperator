import importlib
from typing import Literal

import paddle


def segment_csr(
    src: paddle.Tensor,
    indptr: paddle.Tensor,
    reduce: Literal["mean", "sum"],
    use_scatter: bool = True,
):
    """segment_csr reduces all entries of a CSR-formatted
    matrix by summing or averaging over neighbors.

    Used to reduce features over neighborhoods
    in neuralop.layers.IntegralTransform

    Parameters
    ----------
    src : paddle.Tensor
        tensor of features for each point
    indptr : paddle.Tensor
        splits representing start and end indices
        of each neighborhood in src
    reduce : Literal['mean', 'sum']
        how to reduce a neighborhood. if mean,
        reduce by taking the average of all neighbors.
        Otherwise take the sum.
    """
    if reduce not in ["mean", "sum"]:
        raise ValueError("reduce must be one of 'mean', 'sum'")

    # TODO: support paddle_scatter
    if (
        paddle.device.is_compiled_with_cuda
        and importlib.util.find_spec("paddle_scatter")
        and use_scatter
    ):
        """only import paddle_scatter when cuda is available"""
        import paddle_scatter.segment_csr as scatter_segment_csr

        return scatter_segment_csr(src, indptr, reduce=reduce)

    else:
        n_nbrs = indptr[1:] - indptr[:-1]  # end indices - start indices
        output_shape = list(src.shape)
        output_shape[0] = indptr.shape[0] - 1

        out = paddle.zeros(output_shape)

        for i, start in enumerate(indptr[:-1]):
            if start == src.shape[0]:  # if the last neighborhoods are empty, skip
                break
            for j in range(n_nbrs[i]):
                out[i] += src[start + j]
            if reduce == "mean":
                # out[i] /= n_nbrs[i]  # [TODO] torch code, why need to convert to complex64 on paddle?
                out[i] /= n_nbrs[i].astype(paddle.complex64)
        return out
