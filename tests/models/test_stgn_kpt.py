import torch
from omegaconf import OmegaConf

from project.models.stgn_kpt import STGCNKeypoint


def test_stgn_keypoint_forward_shape():
    hparams = OmegaConf.create(
        {
            "model": {
                "model_class_num": 4,
                "stgn_hidden_dim": 16,
                "stgn_layers": 2,
                "stgn_num_kpts": 6,
            }
        }
    )
    model = STGCNKeypoint(hparams)
    kpts = torch.randn(2, 5, 6, 3)
    out = model(kpts)
    assert out.shape == (2, 4)
