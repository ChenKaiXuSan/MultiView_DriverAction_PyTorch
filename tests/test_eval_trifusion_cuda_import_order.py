from pathlib import Path
import unittest


class EvalTriFusionCudaImportOrderTests(unittest.TestCase):
    def test_torch_imports_before_hydra_to_preserve_cuda_detection(self) -> None:
        source_path = (
            Path(__file__).resolve().parents[1]
            / "TriPoseFusion"
            / "eval"
            / "eval_trifusion_pesudo_gt.py"
        )
        source = source_path.read_text(encoding="utf-8")

        torch_import = source.index("import torch")
        hydra_import = source.index("import hydra")

        self.assertLess(torch_import, hydra_import)


if __name__ == "__main__":
    unittest.main()
