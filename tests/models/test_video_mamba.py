import torch
from omegaconf import OmegaConf

from project.models.video_mamba import VideoMamba


def test_video_mamba_forward_shape():
    hparams = OmegaConf.create(
        {
            "model": {
                "model_class_num": 6,
                "mamba_dim": 32,
                "mamba_layers": 2,
            }
        }
    )
    model = VideoMamba(hparams)
    video = torch.randn(2, 3, 4, 32, 32)
    output = model(video)
    output.sum().backward()
    assert output.shape == (2, 6)
