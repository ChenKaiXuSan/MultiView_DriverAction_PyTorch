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
    assert output.shape == (2, 5)
