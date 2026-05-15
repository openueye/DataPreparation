import numpy as np

from data_preparation.slam.converter import filter_consecutive_static_pose_rows, pose_motion_delta


def _pose_row(frame_id: str, tx: float = 0.0, yaw_deg: float = 0.0):
    yaw = np.deg2rad(yaw_deg)
    c = float(np.cos(yaw))
    s = float(np.sin(yaw))
    matrix = np.asarray(
        [
            [c, -s, 0.0, tx],
            [s, c, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    row = {"frame_id": frame_id}
    for i in range(4):
        for j in range(4):
            row[f"T_odom_from_camera_{i}{j}"] = str(matrix[i, j])
    return row


def test_pose_motion_delta_reports_translation_and_rotation():
    first = np.eye(4, dtype=np.float64)
    second = np.eye(4, dtype=np.float64)
    second[0, 3] = 0.25
    second[:3, :3] = np.asarray(
        [
            [0.0, -1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )

    translation, rotation_deg = pose_motion_delta(first, second)

    assert np.isclose(translation, 0.25)
    assert np.isclose(rotation_deg, 90.0)


def test_pose_motion_delta_projects_non_orthogonal_rotations_before_comparison():
    pose = np.eye(4, dtype=np.float64)
    pose[:3, :3] = np.asarray(
        [
            [1.0, 0.01, 0.0],
            [0.0, 0.99, 0.02],
            [0.0, 0.0, 1.01],
        ],
        dtype=np.float64,
    )

    translation, rotation_deg = pose_motion_delta(pose, pose.copy())

    assert translation == 0.0
    assert rotation_deg == 0.0


def test_filter_consecutive_static_pose_rows_drops_exact_repeats():
    rows = [
        _pose_row("000000"),
        _pose_row("000001"),
        _pose_row("000002", tx=0.1),
        _pose_row("000003", tx=0.1),
    ]

    kept, stats = filter_consecutive_static_pose_rows(rows)

    assert [row["frame_id"] for row in kept] == ["000000", "000002"]
    assert stats["dropped_count"] == 2
    assert [item["frame_id"] for item in stats["dropped_frames"]] == ["000001", "000003"]


def test_filter_consecutive_static_pose_rows_keeps_motion_above_thresholds():
    rows = [
        _pose_row("000000"),
        _pose_row("000001", tx=1e-5),
        _pose_row("000002", tx=1e-5, yaw_deg=0.01),
    ]

    kept, stats = filter_consecutive_static_pose_rows(
        rows,
        min_translation_m=1e-6,
        min_rotation_deg=1e-4,
    )

    assert [row["frame_id"] for row in kept] == ["000000", "000001", "000002"]
    assert stats["dropped_count"] == 0
