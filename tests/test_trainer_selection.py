from omegaconf import OmegaConf

from project.trainer.single import select_single_trainer_cls
from project.trainer.multi import select_multi_trainer_cls
from project.trainer.baseline.train_3dcnn import Res3DCNNTrainer
from project.trainer.early.train_early_fusion import EarlyFusion3DCNNTrainer
from project.trainer.late.train_late_fusion import LateFusion3DCNNTrainer
from project.trainer.mid.train_se_attn import SEAttnTrainer


def test_select_single_trainer_rgb_3dcnn():
    hparams = OmegaConf.create(
        {"train": {"view": "single"}, "model": {"input_type": "rgb", "backbone": "3dcnn"}}
    )
    trainer_cls = select_single_trainer_cls(hparams)
    assert trainer_cls is Res3DCNNTrainer


def test_select_multi_trainer_early_fusion():
    hparams = OmegaConf.create(
        {
            "train": {"view": "multi"},
            "model": {
                "input_type": "rgb",
                "backbone": "3dcnn",
                "fuse_method": "avg",
            },
        }
    )
    trainer_cls = select_multi_trainer_cls(hparams)
    assert trainer_cls is EarlyFusion3DCNNTrainer


def test_select_multi_trainer_mid_fusion():
    hparams = OmegaConf.create(
        {
            "train": {"view": "multi"},
            "model": {
                "input_type": "rgb",
                "backbone": "3dcnn",
                "fuse_method": "se_atn",
            },
        }
    )
    trainer_cls = select_multi_trainer_cls(hparams)
    assert trainer_cls is SEAttnTrainer


def test_select_multi_trainer_late_fusion():
    hparams = OmegaConf.create(
        {
            "train": {"view": "multi"},
            "model": {
                "input_type": "rgb",
                "backbone": "3dcnn",
                "fuse_method": "late",
            },
        }
    )
    trainer_cls = select_multi_trainer_cls(hparams)
    assert trainer_cls is LateFusion3DCNNTrainer
