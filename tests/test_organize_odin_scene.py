import numpy as np

from data_preparation.shared.organize_odin_scene import CameraCalibration, build_tf_chain


def test_build_tf_chain_keeps_tcl0_as_production_camera_from_lidar_when_inverse_scores_higher():
    t_camera_from_lidar = np.asarray(
        [
            [1.0, 0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0, 2.0],
            [0.0, 0.0, 1.0, 3.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    calibration = CameraCalibration(
        path="calib.yaml",
        matrix_name="Tcl_0",
        camera_name="cam_0",
        camera_params={},
        k_like=[],
        distortion={},
        t_camera_from_lidar=t_camera_from_lidar,
    )

    tf_chain = build_tf_chain(calibration, {"selected_direction": "lidar_from_camera"})

    assert tf_chain["validation_selected_direction"] == "lidar_from_camera"
    assert tf_chain["production_direction"] == "camera_from_lidar"
    assert tf_chain["production_transform_source"] == "calib.yaml:Tcl_0"
    assert tf_chain["debug_inverse_used_for_production"] is False
    assert np.allclose(tf_chain["T_camera_from_lidar"], t_camera_from_lidar)
    assert np.allclose(tf_chain["T_lidar_from_camera"], np.linalg.inv(t_camera_from_lidar))
    assert np.allclose(
        np.asarray(tf_chain["T_camera_from_imu"], dtype=np.float64),
        t_camera_from_lidar @ np.asarray(tf_chain["T_lidar_from_imu"], dtype=np.float64),
    )
    assert np.allclose(
        np.asarray(tf_chain["T_imu_from_camera"], dtype=np.float64),
        np.linalg.inv(np.asarray(tf_chain["T_camera_from_imu"], dtype=np.float64)),
    )
