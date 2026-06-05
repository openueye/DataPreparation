import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from data_preparation.depth_prior.project import export_depth_priors


def _write_scene(root: Path, *, duplicate_stems: bool = False) -> None:
    (root / "images").mkdir(parents=True)
    (root / "sparse" / "0").mkdir(parents=True)
    Image.new("RGB", (4, 4), color=(128, 128, 128)).save(root / "images" / "000001.png")
    if duplicate_stems:
        Image.new("RGB", (4, 4), color=(128, 128, 128)).save(root / "images" / "000001.jpg")
    (root / "sparse" / "0" / "cameras.txt").write_text("1 PINHOLE 4 4 2 2 2 2\n", encoding="utf-8")
    image_lines = ["1 1 0 0 0 0 0 0 1 000001.png", ""]
    if duplicate_stems:
        image_lines.extend(["2 1 0 0 0 0 0 0 1 000001.jpg", ""])
    (root / "sparse" / "0" / "images.txt").write_text("\n".join(image_lines), encoding="utf-8")
    (root / "sparse" / "0" / "points3D.txt").write_text("1 0 0 2 255 255 255 0\n", encoding="utf-8")


def _write_prepared_scene(
    root: Path,
    *,
    frame_id: str = "000001",
    raw_cloud_id: str = "000123",
    dt_ns: int = 500_000,
    include_cloud: bool = True,
    tf_payload: dict | None = None,
) -> None:
    (root / "associations").mkdir(parents=True)
    (root / "lidar" / "raw_frames").mkdir(parents=True)
    (root / "calib").mkdir(parents=True)
    (root / "associations" / "frame_associations.csv").write_text(
        "\n".join(
            [
                "frame_id,image_timestamp_ns,raw_cloud_id,raw_cloud_timestamp_ns,image_to_raw_cloud_dt_ns,raw_cloud_dt_outlier",
                f"{frame_id},1000000000,{raw_cloud_id},{1000000000 + dt_ns},{dt_ns},False",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    if include_cloud:
        np.savez_compressed(
            root / "lidar" / "raw_frames" / f"{raw_cloud_id}.npz",
            xyz=np.asarray(
                [
                    [0.0, 0.0, 0.0],
                    [0.0, 0.0, 2.0],
                    [1.0, 0.0, 1.0],
                    [0.0, 0.0, -2.0],
                ],
                dtype=np.float32,
            ),
        )
    if tf_payload is None:
        tf_payload = {
            "selected_direction": "camera_from_lidar",
            "T_camera_from_lidar": [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 1], [0, 0, 0, 1]],
        }
    (root / "calib" / "tf_chain.json").write_text(json.dumps(tf_payload), encoding="utf-8")


def _raw_frame_args(scene: Path, prepared: Path, depths: Path, **overrides):
    values = {
        "scene_dir": scene,
        "prepared_scene_dir": prepared,
        "output_depths_dir": depths,
        "chunk_size": 1000,
        "overwrite": True,
        "min_depth": 0.0,
        "max_depth": 0.0,
        "max_raw_cloud_dt_ms": 20.0,
        "method": "raw-frame",
        "fusion_window": 5,
        "fusion_mode": "centered",
        "output_masks_dir": None,
        "output_confidence_dir": None,
        "write_confidence": False,
        "overlay_count": 0,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def _write_fusion_scene(root: Path, frame_count: int = 5) -> None:
    (root / "images").mkdir(parents=True)
    (root / "sparse" / "0").mkdir(parents=True)
    for idx in range(frame_count):
        Image.new("RGB", (8, 6), color=(128, 128, 128)).save(root / "images" / f"{idx:06d}.png")
    (root / "sparse" / "0" / "cameras.txt").write_text("1 PINHOLE 8 6 2 2 4 3\n", encoding="utf-8")
    image_lines = []
    for idx in range(frame_count):
        image_lines.extend([f"{idx + 1} 1 0 0 0 0 0 0 1 {idx:06d}.png", ""])
    (root / "sparse" / "0" / "images.txt").write_text("\n".join(image_lines), encoding="utf-8")
    (root / "sparse" / "0" / "points3D.txt").write_text("1 0 0 2 255 255 255 0\n", encoding="utf-8")


def _write_fusion_prepared_scene(
    root: Path,
    *,
    frame_count: int = 5,
    camera_x: list[float] | None = None,
) -> None:
    (root / "associations").mkdir(parents=True)
    (root / "lidar" / "raw_frames").mkdir(parents=True)
    (root / "calib").mkdir(parents=True)
    (root / "poses").mkdir(parents=True)
    camera_x = camera_x or [0.0 for _ in range(frame_count)]

    association_lines = [
        "frame_id,image_timestamp_ns,image_path,rectified_image_path,raw_cloud_id,raw_cloud_timestamp_ns,image_to_raw_cloud_dt_ns,raw_cloud_dt_outlier"
    ]
    pose_header = [
        "frame_id",
        "image_timestamp_ns",
        "odom_prev_timestamp_ns",
        "odom_next_timestamp_ns",
        "odom_interp_alpha",
        "tx",
        "ty",
        "tz",
        "qx",
        "qy",
        "qz",
        "qw",
    ]
    pose_header.extend(f"T_odom_from_camera_{r}{c}" for r in range(4) for c in range(4))
    pose_lines = [",".join(pose_header)]
    for idx in range(frame_count):
        frame_id = f"{idx:06d}"
        ts = 1_000_000_000 + idx * 100_000_000
        raw_id = f"{idx:06d}"
        association_lines.append(
            f"{frame_id},{ts},images/{frame_id}.jpg,images_rectified/{frame_id}.jpg,{raw_id},{ts + 1_000_000},1000000,False"
        )
        transform = np.eye(4, dtype=np.float64)
        transform[0, 3] = camera_x[idx]
        pose_values = [
            frame_id,
            str(ts),
            str(ts),
            str(ts),
            "0.0",
            str(camera_x[idx]),
            "0.0",
            "0.0",
            "0.0",
            "0.0",
            "0.0",
            "1.0",
        ]
        pose_values.extend(str(float(value)) for value in transform.reshape(-1))
        pose_lines.append(",".join(pose_values))
        np.savez_compressed(
            root / "lidar" / "raw_frames" / f"{raw_id}.npz",
            xyz=np.asarray(
                [
                    [0.0, 0.0, 2.0],
                    [0.0, 0.0, 4.0],
                    [1.0, 0.0, 2.0],
                    [20.0, 0.0, 2.0],
                    [0.0, 0.0, -2.0],
                ],
                dtype=np.float32,
            ),
        )
    (root / "associations" / "frame_associations.csv").write_text("\n".join(association_lines) + "\n", encoding="utf-8")
    (root / "poses" / "camera_poses.csv").write_text("\n".join(pose_lines) + "\n", encoding="utf-8")
    (root / "calib" / "tf_chain.json").write_text(
        json.dumps(
            {
                "selected_direction": "camera_from_lidar",
                "T_camera_from_lidar": [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
            }
        ),
        encoding="utf-8",
    )
    (root / "calib" / "camera_rectified.json").write_text(
        json.dumps(
            {
                "camera_model": "PINHOLE",
                "width": 8,
                "height": 6,
                "K_like": [[2.0, 0.0, 4.0], [0.0, 2.0, 3.0], [0.0, 0.0, 1.0]],
            }
        ),
        encoding="utf-8",
    )


def test_depth_range_filter_removes_near_outlier(tmp_path):
    scene = tmp_path / "scene"
    prepared = tmp_path / "prepared"
    _write_scene(scene)
    _write_prepared_scene(prepared)

    report = export_depth_priors(
        _raw_frame_args(scene, prepared, tmp_path / "depths", min_depth=2.0)
    )

    depth = np.load(tmp_path / "depths" / "000001.npy")
    assert np.isclose(depth[2, 2], 3.0)
    assert depth[2, 3] == 0.0
    assert report["depth_filter"]["min_depth"] == 2.0
    image_stats = report["per_image"][0]
    assert image_stats["valid_depth_min"] == 3.0
    assert image_stats["valid_depth_max"] == 3.0
    assert image_stats["valid_depth_mean"] == 3.0


def test_duplicate_image_stems_are_rejected(tmp_path):
    scene = tmp_path / "scene"
    prepared = tmp_path / "prepared"
    _write_scene(scene, duplicate_stems=True)
    _write_prepared_scene(prepared)

    try:
        export_depth_priors(_raw_frame_args(scene, prepared, tmp_path / "depths"))
    except ValueError as exc:
        assert "Image stems are not unique" in str(exc)
    else:
        raise AssertionError("Expected duplicate output stems to be rejected")


def test_image_dimension_mismatch_is_rejected(tmp_path):
    scene = tmp_path / "scene"
    prepared = tmp_path / "prepared"
    _write_scene(scene)
    _write_prepared_scene(prepared)
    Image.new("RGB", (5, 4), color=(128, 128, 128)).save(scene / "images" / "000001.png")

    try:
        export_depth_priors(_raw_frame_args(scene, prepared, tmp_path / "depths"))
    except ValueError as exc:
        assert "COLMAP camera dimensions do not match image file" in str(exc)
    else:
        raise AssertionError("Expected mismatched image dimensions to be rejected")


def test_raw_frame_projection_writes_depth_with_zbuffer_and_report(tmp_path):
    scene = tmp_path / "scene"
    prepared = tmp_path / "prepared"
    _write_scene(scene)
    _write_prepared_scene(prepared)

    report = export_depth_priors(_raw_frame_args(scene, prepared, tmp_path / "depths"))

    depth_path = tmp_path / "depths" / "000001.npy"
    depth = np.load(depth_path)
    assert depth.shape == (4, 4)
    assert depth.dtype == np.float32
    assert np.isclose(depth[2, 2], 1.0)
    assert np.isclose(depth[2, 3], 2.0)
    assert report["method"] == "raw-frame"
    assert report["per_image"][0]["raw_cloud_id"] == "000123"
    assert report["per_image"][0]["image_to_raw_cloud_dt_ns"] == 500_000
    assert report["per_image"][0]["valid_pixels"] == 2


def test_raw_frame_default_behavior_unchanged(tmp_path):
    scene = tmp_path / "scene"
    prepared = tmp_path / "prepared"
    _write_scene(scene)
    _write_prepared_scene(prepared)

    report = export_depth_priors(_raw_frame_args(scene, prepared, tmp_path / "depths"))

    assert report["method"] == "raw-frame"
    assert (tmp_path / "depths" / "000001.npy").exists()
    assert (scene / "depth_prior_report.json").exists()
    assert not (scene / "depth_prior_report_local_fused_5f.json").exists()


def test_local_fused_pose_transform_direction_and_zbuffer(tmp_path):
    scene = tmp_path / "scene"
    prepared = tmp_path / "prepared"
    _write_fusion_scene(scene, frame_count=3)
    _write_fusion_prepared_scene(prepared, frame_count=3, camera_x=[0.0, 1.0, 0.0])

    report = export_depth_priors(
        _raw_frame_args(scene, prepared, None, method="local-fused", fusion_window=3)
    )

    depth = np.load(scene / "depths_lidar_fused_3f" / "000001.npy")
    assert np.isclose(depth[3, 3], 2.0)
    assert np.isclose(depth[3, 4], 2.0)
    assert np.isclose(depth[3, 5], 2.0)
    assert report["method"] == "local-fused"
    target = report["per_image"][1]
    assert target["source_frame_ids"] == ["000000", "000001", "000002"]
    assert target["zbuffer_collision_pixels"] >= 1
    assert target["zbuffer_conflict_pixels"] >= 1


def test_local_fused_window_5_naming_masks_and_boundary_fallback(tmp_path):
    scene = tmp_path / "scene"
    prepared = tmp_path / "prepared"
    _write_fusion_scene(scene, frame_count=5)
    _write_fusion_prepared_scene(prepared, frame_count=5)

    report = export_depth_priors(
        _raw_frame_args(scene, prepared, None, method="local-fused", fusion_window=5, write_confidence=True)
    )

    assert (scene / "depths_lidar_fused_5f" / "000000.npy").exists()
    assert (scene / "masks_lidar_fused_5f" / "000000.npy").exists()
    assert (scene / "confidence_lidar_fused_5f" / "000000.npy").exists()
    assert (scene / "depth_prior_report_local_fused_5f.json").exists()
    first = report["per_image"][0]
    assert first["actual_source_count"] == 3
    assert first["source_frame_ids"] == ["000000", "000001", "000002"]
    assert first["fusion_time_window_ns"] == 200_000_000
    assert "valid_ratio_summary" in report
    assert "depth_percentiles" in first
    assert "outlier_depth_pixels" in first


def test_local_fused_rejects_even_fusion_window(tmp_path):
    scene = tmp_path / "scene"
    prepared = tmp_path / "prepared"
    _write_fusion_scene(scene, frame_count=3)
    _write_fusion_prepared_scene(prepared, frame_count=3)

    try:
        export_depth_priors(_raw_frame_args(scene, prepared, None, method="local-fused", fusion_window=4))
    except ValueError as exc:
        assert "fusion-window must be a positive odd integer" in str(exc)
    else:
        raise AssertionError("Expected even fusion window to be rejected")


def test_raw_frame_projection_rejects_missing_synchronized_cloud(tmp_path):
    scene = tmp_path / "scene"
    prepared = tmp_path / "prepared"
    _write_scene(scene)
    _write_prepared_scene(prepared, include_cloud=False)

    try:
        export_depth_priors(_raw_frame_args(scene, prepared, tmp_path / "depths"))
    except FileNotFoundError as exc:
        assert "Missing synchronized raw LiDAR cloud" in str(exc)
    else:
        raise AssertionError("Expected missing synchronized cloud to be rejected")


def test_raw_frame_projection_rejects_excessive_sync_delta(tmp_path):
    scene = tmp_path / "scene"
    prepared = tmp_path / "prepared"
    _write_scene(scene)
    _write_prepared_scene(prepared, dt_ns=2_000_000)

    try:
        export_depth_priors(
            _raw_frame_args(scene, prepared, tmp_path / "depths", max_raw_cloud_dt_ms=1.0)
        )
    except ValueError as exc:
        assert "exceeds --max-raw-cloud-dt-ms" in str(exc)
    else:
        raise AssertionError("Expected excessive synchronization delta to be rejected")


def test_raw_frame_projection_rejects_missing_camera_from_lidar_transform(tmp_path):
    scene = tmp_path / "scene"
    prepared = tmp_path / "prepared"
    _write_scene(scene)
    _write_prepared_scene(prepared, tf_payload={"selected_direction": "unknown"})

    try:
        export_depth_priors(_raw_frame_args(scene, prepared, tmp_path / "depths"))
    except ValueError as exc:
        assert "T_camera_from_lidar" in str(exc)
    else:
        raise AssertionError("Expected missing LiDAR-to-camera transform to be rejected")
