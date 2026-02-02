import pytest
import torch
from omegaconf import OmegaConf

from project.models.pose_kpt_fusion_res_3dcnn import PoseKptFusionRes3DCNN


@pytest.fixture
def sample_hparams():
    return OmegaConf.create(
        {
            "model": {
                "model_class_num": 5,
                "kpt_in_dim": 3,
                "kpt_hidden_dim": 16,
                "kpt_dropout": 0.0,
                "kpt_fusion_strategy": "gated",
                "kpt_gate_hidden_dim": 8,
            }
        }
    )


def test_pose_kpt_fusion_gated_forward_shape(sample_hparams):
    model = PoseKptFusionRes3DCNN(hparams=sample_hparams)
    model.eval()

    video = torch.randn(2, 3, 8, 224, 224)
    kpt = torch.randn(2, 8, 6, 3)

    with torch.no_grad():
        output = model(video, kpt)

    assert output.shape == (2, sample_hparams.model.model_class_num)


def test_pose_kpt_fusion_weighted_forward_shape(sample_hparams):
    weighted_hparams = sample_hparams.copy()
    weighted_hparams.model.kpt_fusion_strategy = "weighted"
    model = PoseKptFusionRes3DCNN(hparams=weighted_hparams)
    model.eval()

    video = torch.randn(1, 3, 8, 224, 224)
    kpt = torch.randn(1, 8, 6, 3)

    with torch.no_grad():
        output = model(video, kpt)

    assert output.shape == (1, weighted_hparams.model.model_class_num)
