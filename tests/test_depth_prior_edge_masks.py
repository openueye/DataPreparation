import argparse
import json

import numpy as np

from data_preparation.depth_prior.edge_masks import apply_depth_mask, generate_edge_masks


def test_generate_edge_masks_keeps_lidar_anchors_and_writes_report(tmp_path):
    scene = tmp_path / "scene"
    completed_dir = scene / "completed"
    anchor_dir = scene / "anchors"
    completed_dir.mkdir(parents=True)
    anchor_dir.mkdir(parents=True)

    completed = np.asarray(
        [
            [2.0, 2.0, 2.0],
            [2.0, 8.0, 2.0],
            [2.0, 2.0, 2.0],
        ],
        dtype=np.float32,
    )
    anchor = np.zeros((3, 3), dtype=np.float32)
    anchor[1, 1] = 8.0
    np.save(completed_dir / "000001.npy", completed)
    np.save(anchor_dir / "000001.npy", anchor)

    report_by_label = generate_edge_masks(
        argparse.Namespace(
            scene_dir=scene,
            completed_depths_dir=completed_dir,
            anchor_depths_dir=anchor_dir,
            percentiles=[50.0],
            target_min=1.0,
            target_max=50.0,
            overwrite=True,
        )
    )

    mask_path = scene / "masks_spnet_fused9_anchor_preserved_edge50" / "000001.npy"
    report_path = scene / "depth_prior_report_spnet_fused9_anchor_preserved_edge50_mask.json"
    mask = np.load(mask_path).astype(bool)
    report = json.loads(report_path.read_text(encoding="utf-8"))

    assert mask[1, 1]
    assert report_by_label["edge50"]["anchor_ratio_summary"]["mean"] == 1 / 9
    assert report["method"] == "spnet-fused9-anchor-preserved-edge50-mask"
    assert report["per_image"][0]["mask_path"] == str(mask_path)


def test_apply_depth_mask_zeroes_rejected_pixels_and_reports_anchor_error(tmp_path):
    scene = tmp_path / "scene"
    input_dir = scene / "completed"
    mask_dir = scene / "masks"
    anchor_dir = scene / "anchors"
    output_dir = scene / "masked"
    for path in (input_dir, mask_dir, anchor_dir):
        path.mkdir(parents=True)

    depth = np.asarray([[2.0, 3.0], [4.0, 5.0]], dtype=np.float32)
    mask = np.asarray([[1, 0], [1, 1]], dtype=np.uint8)
    anchor = np.asarray([[2.0, 0.0], [0.0, 5.0]], dtype=np.float32)
    np.save(input_dir / "000001.npy", depth)
    np.save(mask_dir / "000001.npy", mask)
    np.save(anchor_dir / "000001.npy", anchor)

    report = apply_depth_mask(
        argparse.Namespace(
            scene_dir=scene,
            input_depths_dir=input_dir,
            input_masks_dir=mask_dir,
            anchor_depths_dir=anchor_dir,
            output_depths_dir=output_dir,
            output_report=scene / "masked_report.json",
            method="test-edge-mask",
            overwrite=True,
        )
    )

    masked = np.load(output_dir / "000001.npy")
    assert masked.tolist() == [[2.0, 0.0], [4.0, 5.0]]
    assert report["anchor_max_abs_error_summary"]["max"] == 0.0
    assert report["valid_ratio_summary"]["mean"] == 0.75
