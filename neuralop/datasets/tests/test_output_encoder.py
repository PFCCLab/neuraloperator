from ..output_encoder import UnitGaussianNormalizer
import paddle


def test_UnitGaussianNormalizer():
    x = paddle.rand([4, 3, 4, 5, 6])*2.5
    mean = paddle.mean(x, axis=[0, 2, 3, 4], keepdim=True)
    std = paddle.std(x, axis=[0, 2, 3, 4], keepdim=True)

    # Init normalizer with ground-truth mean and std
    normalizer = UnitGaussianNormalizer(mean=mean, std=std)
    x_normalized = normalizer.transform(x)
    x_unnormalized = normalizer.inverse_transform(x_normalized)

    eps = 1e-5
    assert paddle.allclose(x_unnormalized, x, atol=1e-6).item(), 'x_normalized is error'
    assert paddle.mean(x_normalized) <= eps
    assert (paddle.std(x_normalized) - 1) <= eps

    # Init by fitting whole data at once
    normalizer = UnitGaussianNormalizer(dim=[0, 2, 3, 4])
    normalizer.fit(x)
    x_normalized = normalizer.transform(x)
    x_unnormalized = normalizer.inverse_transform(x_normalized)

    eps = 1e-3
    assert paddle.allclose(x_unnormalized, x, atol=1e-6).item(), 'x_unnormalized is error'
    assert paddle.mean(x_normalized) <= eps
    assert (paddle.std(x_normalized) - 1) <= eps

    assert paddle.allclose(normalizer.mean, mean).item(), 'normalizer.mean is error'
    assert paddle.allclose(normalizer.std, std, rtol=1e-3, atol=1e-3).item(), 'normalizer.std is error'

    # Incrementally compute mean and var
    normalizer = UnitGaussianNormalizer(dim=[0, 2, 3, 4])
    normalizer.partial_fit(x, batch_size=2)
    x_normalized = normalizer.transform(x)
    x_unnormalized = normalizer.inverse_transform(x_normalized)

    eps = 1e-3
    assert paddle.allclose(x_unnormalized, x, atol=1e-6).item(), 'x_unnormalized is error'
    assert paddle.mean(x_normalized) <= eps
    assert (paddle.std(x_normalized) - 1) <= eps

    assert paddle.allclose(normalizer.mean, mean).item(), 'normalizer.mean is error'
    print(normalizer.std, std)
    assert paddle.allclose(normalizer.std, std, rtol=1e-2, atol=1e-2).item(), 'normalizer.std is error'
