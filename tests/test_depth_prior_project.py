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
    }
    values.update(overrides)
    return argparse.Namespace(**values)


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
