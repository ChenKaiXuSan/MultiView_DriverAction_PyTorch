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
    assert model.classifier.weight.grad is not None
    assert output.shape == (2, 6)


def test_video_mamba_single_frame():
    hparams = OmegaConf.create({"model": {"model_class_num": 4, "mamba_dim": 16}})
    model = VideoMamba(hparams)
    video = torch.randn(1, 3, 1, 16, 16)
    output = model(video)
    assert output.shape == (1, 4)
