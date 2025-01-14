import paddle
import time
from tensorly import tenalg

tenalg.set_backend("einsum")
from pathlib import Path

from configmypy import ConfigPipeline, YamlConfig
from neuralop import get_model


def test_from_config():
    """Test forward/backward from a config file"""
    # Read the configuration
    config_name = "default"
    config_path = Path(__file__).parent.as_posix()
    pipe = ConfigPipeline(
        [
            YamlConfig(
                "./test_config.yaml", config_name=config_name, config_folder=config_path
            ),
        ]
    )
    config = pipe.read_conf()
    config_name = pipe.steps[-1].config_name

    batch_size = config.data.batch_size
    size = config.data.size

    if paddle.device.cuda.device_count() >= 1:
        device = "gpu"
    else:
        device = "cpu"

    paddle.device.set_device(device=device)

    model = get_model(config)
    model = model

    in_data = paddle.randn([batch_size, 3, size, size])
    print(model.__class__)
    print(model)

    t1 = time.time()
    out = model(in_data)
    t = time.time() - t1
    print(f"Output of size {out.shape} in {t}.")

    loss = out.sum()
    loss.backward()
