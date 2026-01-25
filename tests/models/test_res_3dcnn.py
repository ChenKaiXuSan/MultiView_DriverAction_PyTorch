import pytest
import torch
from omegaconf import OmegaConf

# Res3DCNNが定義されているモジュールからインポートしてください
# ここでは script_name.py と仮定しています
from project.models.res_3dcnn import Res3DCNN


@pytest.fixture
def sample_hparams():
    """テスト用のハイパーパラメータを作成するフィクスチャ"""
    return OmegaConf.create({"model": {"model_class_num": 9}})


def test_res3dcnn_initialization(sample_hparams):
    """モデルの初期化が正しく行われるかテスト"""
    model = Res3DCNN(hparams=sample_hparams)
    assert model.model_class_num == 3
    assert model.fuse_method == "late"
    assert model.model is not None


def test_res3dcnn_forward_shape(sample_hparams):
    """
    フォワードパスの出力シェイプが (B, Class_Num) であるかテスト
    入力: (B, C, T, H, W) -> (2, 3, 8, 224, 224)
    """
    # 1. モデルの準備
    model = Res3DCNN(hparams=sample_hparams)
    model.eval()  # テスト時は評価モードに設定

    # 2. ダミーデータの作成 (Batch=2, Channel=3, Time=8, Height=224, Width=224)
    batch_size = 2
    video_input = torch.randn(batch_size, 3, 8, 224, 224)

    # 3. 推論の実行
    with torch.no_grad():
        output = model(video_input)

    # 4. 検証
    expected_shape = (batch_size, sample_hparams.model.model_class_num)
    assert (
        output.shape == expected_shape
    ), f"期待されるシェイプは {expected_shape} ですが、{output.shape} でした。"


@pytest.mark.parametrize("class_num", [4,8,9])
def test_res3dcnn_different_classes(sample_hparams, class_num):
    """異なるクラス数設定でも正しく動作するかテスト"""
    sample_hparams.model.model_class_num = class_num
    model = Res3DCNN(hparams=sample_hparams)

    video_input = torch.randn(1, 3, 4, 112, 112)
    output = model(video_input)

    assert output.shape == (1, class_num)
