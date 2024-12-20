from ..data_transforms import DefaultDataProcessor
from ..output_encoder import UnitGaussianNormalizer
from ..transforms import PositionalEmbedding2D
import paddle


def test_DefaultDataProcessor():
    if paddle.device.is_compiled_with_cuda():
        device = 'gpu'
    else:
        device = 'cpu'
    paddle.device.set_device(device=device)

    x = paddle.randn((1, 2, 64, 64))
    y = paddle.randn((1, 2, 64, 64))

    pos_encoder = PositionalEmbedding2D(grid_boundaries=[[0, 1], [0, 1]])
    normalizer = UnitGaussianNormalizer(mean=paddle.zeros((1, 2, 1, 1)),
                                        std=paddle.ones((1, 2, 1, 1)),
                                        eps=1e-5)

    pipeline = DefaultDataProcessor(
        in_normalizer=normalizer,
        out_normalizer=normalizer,
        positional_encoding=pos_encoder
    )

    data = {'x': x, 'y': y}  # data on cpu at this point

    xform_data = pipeline.preprocess(data)

    # model outputs will be on device by default
    out = paddle.randn((1, 2, 64, 64))

    _, inv_xform_data = pipeline.postprocess(out, xform_data)

    assert paddle.allclose(inv_xform_data['y'].cpu(), data['y']).item(), 'error'
