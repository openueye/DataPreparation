from pathlib import Path
from importlib.machinery import SourceFileLoader
import sys

import numpy as np
import pytest

from data_preparation.shared.io import require_cv2


def _write_test_image(path: Path) -> None:
    cv2 = require_cv2("test image creation")
    image = np.zeros((3, 4, 3), dtype=np.uint8)
    image[..., 0] = 17
    image[..., 1] = 29
    image[..., 2] = 43
    assert cv2.imwrite(str(path), image)


def _write_colmap_scene(scene_dir: Path) -> None:
    sparse_dir = scene_dir / "sparse" / "0"
    sparse_dir.mkdir(parents=True)
    (scene_dir / "images").mkdir()
    _write_test_image(scene_dir / "images" / "000001.jpg")
    _write_test_image(scene_dir / "images" / "000002.jpg")
    (sparse_dir / "cameras.txt").write_text(
        "\n".join(
            [
                "# Camera list with one line of data per camera:",
                "1 PINHOLE 4 3 2.0 2.0 2.0 1.5",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (sparse_dir / "images.txt").write_text(
        "\n".join(
            [
                "# Image list with two lines of data per image:",
                "#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME",
                "1 1 0 0 0 -1 0 0 1 000001.jpg",
                "",
                "2 1 0 0 0 -2 0 0 1 000002.jpg",
                "",
            ]
        ),
        encoding="utf-8",
    )


def test_lsgslam_adapter_writes_euroc_style_images_depths_and_traj(tmp_path):
    from data_preparation.slam.lsgslam_adapter import export_lsgslam_euroc_scene

    source_scene = tmp_path / "slam_scene"
    depth_dir = source_scene / "depths_spnet_fused9_anchor_preserved_edge90"
    output_dir = tmp_path / "lsgslam_scene"
    _write_colmap_scene(source_scene)
    depth_dir.mkdir()
    np.save(depth_dir / "000001.npy", np.full((3, 4), 1.25, dtype=np.float32))
    np.save(depth_dir / "000002.npy", np.full((3, 4), 2.5, dtype=np.float32))

    metadata = export_lsgslam_euroc_scene(
        source_scene,
        output_dir,
        depths_dir=depth_dir,
        copy_images=True,
        overwrite=True,
    )

    assert metadata["frame_count"] == 2
    assert sorted(path.name for path in (output_dir / "data_rect").glob("*.png")) == ["000001.png", "000002.png"]
    assert sorted(path.name for path in (output_dir / "depth_sceneflow").glob("*.npy")) == ["000001.npy", "000002.npy"]
    assert np.load(output_dir / "depth_sceneflow" / "000002.npy").dtype == np.float32
    traj_lines = (output_dir / "traj.txt").read_text(encoding="utf-8").strip().splitlines()
    assert traj_lines[0] == "000001 1 0 0 1 0 1 0 0 0 0 1 0"
    assert traj_lines[1] == "000002 1 0 0 2 0 1 0 0 0 0 1 0"


def test_parse_segment_run_name_reads_numeric_suffix():
    from data_preparation.slam.lsgslam_adapter import parse_segment_run_name

    parsed = parse_segment_run_name("downtown1_1m_000000_000120_1")

    assert parsed == {
        "scene_prefix": "downtown1_1m",
        "start": 0,
        "end": 120,
        "stride": 1,
    }


def test_parse_segment_run_name_rejects_missing_numeric_suffix():
    from data_preparation.slam.lsgslam_adapter import parse_segment_run_name

    with pytest.raises(ValueError, match="numeric start/end/stride suffix"):
        parse_segment_run_name("0606_1M_clean_spnet_fused9_edge90_depth_lsgslam_fullframes_halfres")


def test_validate_lsgslam_feature_layout_requires_global_features(tmp_path):
    from data_preparation.slam.lsgslam_adapter import validate_lsgslam_feature_layout

    scene_dir = tmp_path / "lsgslam_scene"
    (scene_dir / "data_rect").mkdir(parents=True)
    (scene_dir / "depth_sceneflow").mkdir()
    _write_test_image(scene_dir / "data_rect" / "000001.png")
    np.save(scene_dir / "depth_sceneflow" / "000001.npy", np.ones((3, 4), dtype=np.float32))
    (scene_dir / "traj.txt").write_text("000001 1 0 0 0 0 1 0 0 0 0 1 0\n", encoding="utf-8")

    with pytest.raises(FileNotFoundError, match="global_features"):
        validate_lsgslam_feature_layout(scene_dir)


def test_validate_lsgslam_feature_layout_counts_matching_features(tmp_path):
    from data_preparation.slam.lsgslam_adapter import validate_lsgslam_feature_layout

    scene_dir = tmp_path / "lsgslam_scene"
    for subdir in ("data_rect", "depth_sceneflow", "global_features"):
        (scene_dir / subdir).mkdir(parents=True, exist_ok=True)
    for stem in ("000001", "000002"):
        _write_test_image(scene_dir / "data_rect" / f"{stem}.png")
        np.save(scene_dir / "depth_sceneflow" / f"{stem}.npy", np.ones((3, 4), dtype=np.float32))
        np.save(scene_dir / "global_features" / f"{stem}.npy", np.ones((8,), dtype=np.float32))
    (scene_dir / "traj.txt").write_text(
        "000001 1 0 0 0 0 1 0 0 0 0 1 0\n"
        "000002 1 0 0 1 0 1 0 0 0 0 1 0\n",
        encoding="utf-8",
    )

    summary = validate_lsgslam_feature_layout(scene_dir)

    assert summary == {
        "scene_dir": str(scene_dir),
        "rgb_count": 2,
        "depth_count": 2,
        "feature_count": 2,
        "traj_count": 2,
        "first_stem": "000001",
        "last_stem": "000002",
    }


def test_plan_lsgslam_segments_uses_inclusive_overlapping_boundaries():
    from data_preparation.slam.lsgslam_adapter import plan_lsgslam_segments

    segments = plan_lsgslam_segments(total_frames=668, segment_length=100, stride=1)

    assert segments[0] == {"start": 0, "end": 100, "stride": 1}
    assert segments[1] == {"start": 100, "end": 200, "stride": 1}
    assert segments[-1] == {"start": 600, "end": 667, "stride": 1}
    assert len(segments) == 7


def test_plan_lsgslam_segments_rejects_non_overlapping_or_tiny_segments():
    from data_preparation.slam.lsgslam_adapter import plan_lsgslam_segments

    with pytest.raises(ValueError, match="segment_length"):
        plan_lsgslam_segments(total_frames=20, segment_length=20, stride=1)


def test_render_lsgslam_segment_config_sets_full_pipeline_paths():
    from data_preparation.slam.lsgslam_adapter import render_lsgslam_segment_config

    rendered = render_lsgslam_segment_config(
        base_config_import="configs.downtown1.lsgslam_downtown1_1m_edge90",
        scene_name="downtown1_1m_edge90",
        output_root="/tmp/lsgslam_full",
        start=100,
        end=200,
        stride=1,
        progress_every=50,
    )

    assert 'start_idx = 100' in rendered
    assert 'end_idx = 200' in rendered
    assert 'group_name = "/tmp/lsgslam_full"' in rendered
    assert 'run_name = f"{scene_name}_{start_idx:06d}_{end_idx:06d}_{stride}"' in rendered
    assert 'config["run_loop_closure"] = True' in rendered
    assert 'config["data"]["end"] = end_idx' in rendered


def test_downtown1_segment_configs_are_backend_named():
    from data_preparation.slam.lsgslam_adapter import parse_segment_run_name

    thesis_root = Path(__file__).resolve().parents[3]
    lsg_root = thesis_root / "06_Referrance" / "LSG-SLAM"
    config_paths = [
        lsg_root / "configs" / "downtown1" / "lsgslam_downtown1_1m_edge90_segment_000000_000020_1.py",
        lsg_root / "configs" / "downtown1" / "lsgslam_downtown1_1m_edge90_segment_000020_000040_1.py",
    ]

    for path in config_paths:
        sys.path.insert(0, str(lsg_root))
        module = SourceFileLoader(path.stem, str(path)).load_module()
        sys.path.pop(0)
        config = module.config
        parsed = parse_segment_run_name(config["run_name"])

        assert config["data"]["basedir"] == "/home/haibo/Documents/Thesis/04_ProcessedData/slam/downtown1_1M_lsgslam_euroc"
        assert config["workdir"] == "/home/haibo/Documents/Thesis/05_Outputs/Downtown1/lsgslam_paper_pipeline"
        assert config["data"]["start"] == parsed["start"]
        assert config["data"]["end"] == parsed["end"]
        assert config["data"]["stride"] == parsed["stride"]
        assert config["run_name"] == f"downtown1_1m_edge90_{parsed['start']:06d}_{parsed['end']:06d}_{parsed['stride']}"


def test_downtown1_full100_configs_cover_all_frames_without_smoke_output_collision():
    from data_preparation.slam.lsgslam_adapter import parse_segment_run_name

    thesis_root = Path(__file__).resolve().parents[3]
    lsg_root = thesis_root / "06_Referrance" / "LSG-SLAM"
    config_dir = lsg_root / "configs" / "downtown1" / "full100"
    expected_segments = [
        (0, 100),
        (100, 200),
        (200, 300),
        (300, 400),
        (400, 500),
        (500, 600),
        (600, 667),
    ]

    for start, end in expected_segments:
        path = config_dir / f"lsgslam_downtown1_1m_edge90_full100_segment_{start:06d}_{end:06d}_1.py"
        sys.path.insert(0, str(lsg_root))
        module = SourceFileLoader(path.stem, str(path)).load_module()
        sys.path.pop(0)
        config = module.config
        parsed = parse_segment_run_name(config["run_name"])

        assert parsed == {
            "scene_prefix": "downtown1_1m_edge90_full100",
            "start": start,
            "end": end,
            "stride": 1,
        }
        assert config["workdir"] == "/home/haibo/Documents/Thesis/05_Outputs/Downtown1/lsgslam_paper_pipeline_full100"
        assert config["data"]["sequence"] == "downtown1_1m_edge90"
        assert config["data"]["start"] == start
        assert config["data"]["end"] == end

    loop_path = config_dir / "lsgslam_downtown1_1m_edge90_full100_loop_000000_000667_1.py"
    sys.path.insert(0, str(lsg_root))
    loop_module = SourceFileLoader(loop_path.stem, str(loop_path)).load_module()
    sys.path.pop(0)

    assert loop_module.config["run_name"] == "downtown1_1m_edge90_full100_000000_000667_1"
    assert loop_module.config["data"]["start"] == 0
    assert loop_module.config["data"]["end"] == 667
    assert loop_module.config["data"]["sequence"] == "downtown1_1m_edge90"


def _load_lsg_tool_module(module_name: str):
    thesis_root = Path(__file__).resolve().parents[3]
    path = thesis_root / "06_Referrance" / "LSG-SLAM" / "tools" / "loop_closure" / f"{module_name}.py"
    return SourceFileLoader(module_name, str(path)).load_module()


def test_downtown1_backend_readiness_finds_segments_and_loop_folder(tmp_path):
    module = _load_lsg_tool_module("downtown1_backend_readiness")

    base = tmp_path / "pipeline"
    for name in (
        "downtown1_1m_edge90_000020_000040_1",
        "downtown1_1m_edge90_000000_000020_1",
    ):
        folder = base / name
        folder.mkdir(parents=True)
        (folder / "params.npz").write_bytes(b"placeholder")
    loop_folder = base / "downtown1_1m_edge90_000000_000040_1_loops"
    loop_folder.mkdir()
    np.save(loop_folder / "found_loops.npy", np.array([[10, 1]], dtype=np.int64))

    summary = module.inspect_backend_layout(base, "downtown1_1m_edge90", require_loop=True)

    assert summary["odometry_folders"] == [
        "downtown1_1m_edge90_000000_000020_1",
        "downtown1_1m_edge90_000020_000040_1",
    ]
    assert summary["loop_folder"] == "downtown1_1m_edge90_000000_000040_1_loops"
    assert summary["missing"] == []


def test_downtown1_backend_readiness_rejects_missing_segment_params(tmp_path):
    module = _load_lsg_tool_module("downtown1_backend_readiness")

    base = tmp_path / "pipeline"
    (base / "downtown1_1m_edge90_000000_000020_1").mkdir(parents=True)

    summary = module.inspect_backend_layout(base, "downtown1_1m_edge90", require_loop=False)

    assert "downtown1_1m_edge90_000000_000020_1/params.npz" in summary["missing"]


def test_downtown1_backend_readiness_rejects_multiple_loop_folders(tmp_path):
    module = _load_lsg_tool_module("downtown1_backend_readiness")

    base = tmp_path / "pipeline"
    for name in (
        "downtown1_1m_edge90_000000_000020_1",
        "downtown1_1m_edge90_000000_000020_1_loops",
        "downtown1_1m_edge90_000020_000040_1_loops",
    ):
        folder = base / name
        folder.mkdir(parents=True)
        if not name.endswith("_loops"):
            (folder / "params.npz").write_bytes(b"placeholder")

    with pytest.raises(ValueError, match="Expected exactly one loop folder"):
        module.inspect_backend_layout(base, "downtown1_1m_edge90", require_loop=True)


def test_downtown1_extract_global_features_parser_exposes_required_args():
    module = _load_lsg_tool_module("downtown1_extract_global_features")

    parser = module.build_parser()
    option_strings = {
        option
        for action in parser._actions
        for option in action.option_strings
    }

    assert "--image-dir" in option_strings
    assert "--output-dir" in option_strings
    assert "--checkpoint" in option_strings
    assert "--device" in option_strings
    assert "--overwrite" in option_strings

    parsed = parser.parse_args(
        [
            "--image-dir",
            "/tmp/images",
            "--output-dir",
            "/tmp/global_features",
            "--checkpoint",
            "/tmp/TransVPR_MSLS.pth",
            "--device",
            "cpu",
            "--overwrite",
        ]
    )

    assert str(parsed.image_dir) == "/tmp/images"
    assert str(parsed.output_dir) == "/tmp/global_features"
    assert str(parsed.checkpoint) == "/tmp/TransVPR_MSLS.pth"
    assert parsed.device == "cpu"
    assert parsed.overwrite is True
