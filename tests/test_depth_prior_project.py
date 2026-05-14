import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from data_preparation.depth_prior.project import export_depth_priors


def _write_scene(root: Path) -> None:
    (root / "images").mkdir(parents=True)
    (root / "sparse" / "0").mkdir(parents=True)
    Image.new("RGB", (4, 4), color=(128, 128, 128)).save(root / "images" / "000001.png")
    (root / "sparse" / "0" / "cameras.txt").write_text("1 PINHOLE 4 4 2 2 2 2\n", encoding="utf-8")
    (root / "sparse" / "0" / "images.txt").write_text(
        "1 1 0 0 0 0 0 0 1 000001.png\n\n",
        encoding="utf-8",
    )
    (root / "sparse" / "0" / "points3D.txt").write_text("1 0 0 2 255 255 255 0\n", encoding="utf-8")


def test_non_colmap_source_frame_requires_transform(tmp_path):
    scene = tmp_path / "scene"
    _write_scene(scene)
    points = tmp_path / "points.npy"
    np.save(points, np.asarray([[0.0, 0.0, 2.0]], dtype=np.float32))

    args = argparse.Namespace(
        scene_dir=scene,
        point_cloud=points,
        output_depths_dir=tmp_path / "depths",
        chunk_size=1000,
        overwrite=True,
        source_frame="lidar",
        transform_json=None,
    )

    try:
        export_depth_priors(args)
    except ValueError as exc:
        assert "--transform-json is required" in str(exc)
    else:
        raise AssertionError("Expected non-COLMAP source frame to require explicit transform")


def test_transform_is_applied_and_recorded(tmp_path):
    scene = tmp_path / "scene"
    _write_scene(scene)
    points = tmp_path / "points.npy"
    transform = tmp_path / "T.json"
    np.save(points, np.asarray([[0.0, 0.0, 1.0]], dtype=np.float32))
    transform.write_text(
        json.dumps({"T_colmap_from_source": [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 1], [0, 0, 0, 1]]}),
        encoding="utf-8",
    )

    args = argparse.Namespace(
        scene_dir=scene,
        point_cloud=points,
        output_depths_dir=tmp_path / "depths",
        chunk_size=1000,
        overwrite=True,
        source_frame="lidar",
        transform_json=transform,
    )
    report = export_depth_priors(args)

    depth = np.load(tmp_path / "depths" / "000001.npy")
    assert report["transform"]["transform_applied"] is True
    assert report["transform"]["transform_path"] == str(transform)
    assert np.isclose(depth[2, 2], 2.0)
