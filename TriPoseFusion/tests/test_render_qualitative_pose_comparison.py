from pathlib import Path
import tempfile
import unittest

import numpy as np

from TriPoseFusion.eval.render_qualitative_pose_comparison import (
    infer_trifusion_pose,
    load_prediction_pose,
    render_svg,
    resolve_env_folder,
)
from TriPoseFusion.eval.render_qualitative_pose_samples import choose_sample_indices
from TriPoseFusion.eval.render_qualitative_pose_official_sam3d import (
    MODEL52_BODY_ANCHOR_INDICES,
    align_full_pose_to_model_reference,
    align_pose_scale_translation,
    align_pose_scale_with_anchor_translation,
    compact_official_keypoint_colors,
    compact_official_skeleton,
    comparison_reference_pose,
    full_official_keypoint_colors,
    full_official_skeleton,
    select_hand_complete_frame_indices,
    reference_center_and_radius,
)


class QualitativePoseComparisonTest(unittest.TestCase):
    def test_load_prediction_pose_uses_matching_frame_id(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "predictions.npz"
            predictions = np.arange(3 * 4 * 3, dtype=np.float32).reshape(3, 4, 3)
            np.savez(path, P_final=predictions, frame_ids=np.asarray(["10", "20", "30"]))

            pose = load_prediction_pose(path, frame_id="20", frame_index=0)

            np.testing.assert_array_equal(pose, predictions[1])

    def test_render_svg_contains_four_pose_panels(self):
        poses = {
            "Single-view": np.asarray([[0, 0, 0], [1, 0, 0], [1, 1, 0]], dtype=np.float32),
            "Median fusion": np.asarray([[0, 0, 0], [1, 0, 0], [1, 1, 0]], dtype=np.float32),
            "TriPoseFusion": np.asarray([[0, 0, 0], [1, 0, 0], [1, 1, 0]], dtype=np.float32),
            "Pseudo reference": np.asarray([[0, 0, 0], [1, 0, 0], [1, 1, 0]], dtype=np.float32),
        }

        svg = render_svg(
            poses,
            edges=[(0, 1), (1, 2)],
            valid_masks={"Pseudo reference": np.asarray([True, True, True])},
            title="subject=01 env=Day_High frame=20",
            style="sam3d-body",
        )

        self.assertIn("<svg", svg)
        for label in poses:
            self.assertIn(label, svg)
        self.assertIn("<line", svg)
        self.assertIn("<circle", svg)
        self.assertIn("#f6f8fb", svg)

    def test_sam3d_style_marks_left_right_and_axes(self):
        pose = np.zeros((52, 3), dtype=np.float32)
        pose[51] = [0.0, 0.0, 0.0]
        pose[5] = [-1.0, 0.0, 0.0]
        pose[6] = [1.0, 0.0, 0.0]
        pose[28] = [-1.2, -0.4, 0.0]
        pose[7] = [1.2, -0.4, 0.0]
        poses = {
            "Single-view": pose,
            "Median fusion": pose,
            "TriPoseFusion (inferred)": pose,
            "Pseudo reference": pose,
        }

        svg = render_svg(poses, title="demo", style="sam3d-body")

        self.assertIn('class="axis axis-x"', svg)
        self.assertIn('class="axis axis-y"', svg)
        self.assertIn('class="axis axis-z"', svg)
        self.assertIn('class="limb side-left"', svg)
        self.assertIn('class="limb side-right"', svg)

    def test_resolve_env_folder_accepts_paper_label(self):
        self.assertEqual(resolve_env_folder("Day_High"), "昼多い")

    def test_choose_sample_indices_spreads_samples(self):
        self.assertEqual(choose_sample_indices(num_frames=9, samples_per_env=3), [0, 4, 8])
        self.assertEqual(choose_sample_indices(num_frames=2, samples_per_env=3), [0, 1])

    def test_infer_trifusion_pose_uses_confidence_weighted_fusion(self):
        view_pose = np.asarray(
            [[[[0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [20.0, 0.0, 0.0]]]],
            dtype=np.float32,
        )
        view_conf = np.asarray([[[1.0, 2.0, 1.0]]], dtype=np.float32)

        inferred = infer_trifusion_pose(view_pose, view_conf)

        np.testing.assert_allclose(inferred, np.asarray([[10.0, 0.0, 0.0]], dtype=np.float32))

    def test_official_sam3d_compact_skeleton_bridges_body_to_hands(self):
        pose_info = {
            "keypoint_info": {
                5: {"name": "left_shoulder", "id": 5, "color": [1, 2, 3]},
                6: {"name": "right_shoulder", "id": 6, "color": [4, 5, 6]},
                41: {"name": "right_wrist", "id": 41, "color": [7, 8, 9]},
                62: {"name": "left_wrist", "id": 62, "color": [10, 11, 12]},
                67: {"name": "left_acromion", "id": 67, "color": [13, 14, 15]},
                68: {"name": "right_acromion", "id": 68, "color": [16, 17, 18]},
                69: {"name": "neck", "id": 69, "color": [19, 20, 21]},
            },
            "skeleton_info": {},
        }

        edges = compact_official_skeleton(pose_info)
        edge_pairs = {(a, b) for a, b, _ in edges}
        colors = compact_official_keypoint_colors(pose_info)

        self.assertIn((49, 48), edge_pairs)
        self.assertIn((50, 27), edge_pairs)
        self.assertEqual(colors[5], (1 / 255.0, 2 / 255.0, 3 / 255.0))

    def test_comparison_reference_pose_uses_pseudo_reference(self):
        ref = np.ones((3, 3), dtype=np.float32)
        poses = {"Single-view": np.zeros((3, 3), dtype=np.float32), "Pseudo reference": ref}

        self.assertIs(comparison_reference_pose(poses), ref)

    def test_reference_center_and_radius_uses_reference_not_outlier_prediction(self):
        ref = np.asarray([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=np.float32)
        outlier = np.asarray([[100.0, 0.0, 0.0]], dtype=np.float32)
        poses = {"Single-view": outlier, "Pseudo reference": ref}

        center, radius = reference_center_and_radius(poses)

        np.testing.assert_allclose(center, np.asarray([1.0, 0.0, 0.0], dtype=np.float32))
        self.assertLess(radius, 2.0)

    def test_full_official_skeleton_keeps_true_arm_links(self):
        pose_info = {
            "keypoint_info": {
                5: {"name": "left_shoulder", "id": 5, "color": [1, 2, 3]},
                7: {"name": "left_elbow", "id": 7, "color": [4, 5, 6]},
                62: {"name": "left_wrist", "id": 62, "color": [7, 8, 9]},
            },
            "skeleton_info": {
                0: {"link": ("left_shoulder", "left_elbow"), "color": [0, 255, 0]},
                1: {"link": ("left_elbow", "left_wrist"), "color": [0, 255, 0]},
            },
        }

        edges = full_official_skeleton(pose_info)
        colors = full_official_keypoint_colors(pose_info)

        self.assertIn((5, 7), {(a, b) for a, b, _ in edges})
        self.assertIn((7, 62), {(a, b) for a, b, _ in edges})
        self.assertEqual(colors[5], (1 / 255.0, 2 / 255.0, 3 / 255.0))

    def test_select_hand_complete_frame_indices_prefers_both_hands(self):
        valid_mask = np.zeros((4, 70), dtype=bool)
        valid_mask[0, 42:63] = True
        valid_mask[1, 21:42] = True
        valid_mask[2, 21:63] = True
        valid_mask[3, 21:42] = True
        valid_mask[3, 42:63] = True
        aligned_ids = ["0", "1", "2", "3"]
        gt_lookup = {fid: idx for idx, fid in enumerate(aligned_ids)}

        selected = select_hand_complete_frame_indices(
            aligned_ids,
            gt_lookup,
            valid_mask,
            samples_per_env=2,
            min_hand_valid_ratio=0.8,
        )

        self.assertEqual(selected, [2, 3])

    def test_align_full_pose_to_model_reference_scales_common_joints(self):
        full_pose = np.zeros((70, 3), dtype=np.float32)
        full_pose[0] = [0.0, 0.0, 0.0]
        full_pose[1] = [2.0, 0.0, 0.0]
        full_pose[69] = [4.0, 0.0, 0.0]
        model_pose = np.zeros((52, 3), dtype=np.float32)
        model_pose[0] = [0.0, 0.0, 0.0]
        model_pose[1] = [1.0, 0.0, 0.0]
        model_pose[51] = [2.0, 0.0, 0.0]

        aligned, scale = align_full_pose_to_model_reference(full_pose, model_pose)

        self.assertAlmostEqual(scale, 0.5)
        np.testing.assert_allclose(aligned[1], model_pose[1])
        np.testing.assert_allclose(aligned[69], model_pose[51])

    def test_align_pose_scale_translation_matches_reference_scale_without_rotation(self):
        source = np.asarray(
            [[10.0, 0.0, 0.0], [14.0, 0.0, 0.0], [12.0, 2.0, 0.0]],
            dtype=np.float32,
        )
        target = np.asarray(
            [[1.0, 5.0, 0.0], [3.0, 5.0, 0.0], [2.0, 6.0, 0.0]],
            dtype=np.float32,
        )

        aligned, scale = align_pose_scale_translation(source, target)

        self.assertAlmostEqual(scale, 0.5)
        np.testing.assert_allclose(aligned, target, atol=1e-6)

    def test_align_pose_scale_with_anchor_translation_keeps_scale_and_aligns_body_anchor(self):
        source = np.asarray(
            [[10.0, 0.0, 0.0], [14.0, 0.0, 0.0], [30.0, 0.0, 0.0], [34.0, 0.0, 0.0]],
            dtype=np.float32,
        )
        target = np.asarray(
            [[1.0, 5.0, 0.0], [3.0, 5.0, 0.0], [5.0, 5.0, 0.0], [7.0, 5.0, 0.0]],
            dtype=np.float32,
        )

        aligned, scale = align_pose_scale_with_anchor_translation(
            source,
            target,
            anchor_indices=[0, 1],
        )

        source_centered = source - source.mean(axis=0, keepdims=True)
        target_centered = target - target.mean(axis=0, keepdims=True)
        expected_scale = np.sqrt(np.sum(target_centered * target_centered) / np.sum(source_centered * source_centered))
        self.assertAlmostEqual(scale, expected_scale)
        np.testing.assert_allclose(aligned[[0, 1]].mean(axis=0), target[[0, 1]].mean(axis=0), atol=1e-6)
        self.assertGreater(np.linalg.norm(aligned[[2, 3]].mean(axis=0) - target[[2, 3]].mean(axis=0)), 0.1)
        self.assertEqual(MODEL52_BODY_ANCHOR_INDICES, (5, 6, 51))


if __name__ == "__main__":
    unittest.main()
