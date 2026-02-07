import torch
from omegaconf import OmegaConf

from project.models.video_transformer import VideoTransformer


def test_video_transformer_forward_shape():
    hparams = OmegaConf.create(
        {
            "model": {
                "model_class_num": 5,
                "transformer_dim": 32,
                "transformer_layers": 2,
                "transformer_heads": 4,
                "transformer_ff_dim": 64,
            }
        }
    )
    model = VideoTransformer(hparams)
    video = torch.randn(2, 3, 4, 32, 32)
    output = model(video)
    output.sum().backward()
    assert model.classifier.weight.grad is not None
    assert output.shape == (2, 5)


def test_video_transformer_single_frame():
    hparams = OmegaConf.create({"model": {"model_class_num": 3, "transformer_dim": 16}})
    model = VideoTransformer(hparams)
    video = torch.randn(1, 3, 1, 16, 16)
    output = model(video)
    assert output.shape == (1, 3)
