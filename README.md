# data_preparation

`data_preparation` builds COLMAP-compatible scene folders for 3DGS training from Odin1 ROS bag directories.

The formal routes expose one data input: `--rosbag-dir`. Each route first organizes the ROS bag into a pure-headerstamp intermediate scene under `Thesis/04_ProcessedData/rosbag_prepared/`, then runs the requested COLMAP export path.

## Routes

```text
sfm     ROS bag -> rectified images -> COLMAP SfM scene
hybrid  ROS bag -> SfM cameras/poses + aligned SLAM/LiDAR points
slam    ROS bag -> SLAM poses + SLAM/LiDAR points -> COLMAP text scene
```

Default outputs:

```text
Thesis/04_ProcessedData/rosbag_prepared/<scene>_pure_headerstamp/
Thesis/04_ProcessedData/sfm/<scene>/
Thesis/04_ProcessedData/hybrid_sfm_lidar/<scene>/
Thesis/04_ProcessedData/slam_compat/<scene>/
```

Show the route summary:

```bash
conda run -n 3dgs_train python -m data_preparation
```

Run these commands from the `3dgs_train` conda environment. Runtime dependencies include `numpy`, `opencv-python`, `Pillow`, and `plyfile`.

The organizer looks for `cam_in_ex.txt` in the scene bag directory first, then in the parent `001_rosbags/` directory.

## Usage

### SFM

```bash
conda run -n 3dgs_train python -m data_preparation sfm \
  --scene Downtown1 \
  --rosbag-dir /home/haibo/Documents/Thesis/03_Datasets/001_rosbags/Downtown1 \
  -- --camera-model PINHOLE --matcher sequential --sift-gpu
```

### Hybrid

Hybrid internally runs/reuses the SFM route output for the same scene. The exported `sparse/0/points3D.txt` and `sparse/0/points3D.ply` contain the same voxel-downsampled SLAM/LiDAR points.

```bash
conda run -n 3dgs_train python -m data_preparation hybrid \
  --scene Downtown1 \
  --rosbag-dir /home/haibo/Documents/Thesis/03_Datasets/001_rosbags/Downtown1 \
  --max-points 3000000
```

### SLAM

The exported `sparse/0/points3D.txt` and `sparse/0/points3D.ply` contain the same voxel-downsampled SLAM/LiDAR points.

```bash
python -m data_preparation slam \
  --scene Downtown1 \
  --rosbag-dir /home/haibo/Documents/Thesis/03_Datasets/001_rosbags/Downtown1 \
  --output-dir /home/haibo/Documents/Thesis/04_ProcessedData/slam/downtown1_1M_new \
  --max-points 1000000 \
  -- --copy-images
```

Use `--max-points 0` to disable point-cloud downsampling.

## Depth Prior Projection

Depth priors are generated from synchronized raw LiDAR frames. The command
uses the converted scene's `manifest.json` to find the prepared pure-headerstamp
scene, then reads `associations/frame_associations.csv`,
`lidar/raw_frames/*.npz`, and `calib/tf_chain.json`.

```bash
python -m data_preparation depth-prior-project \
  --scene-dir /home/haibo/Documents/Thesis/04_ProcessedData/slam/downtown1_3M \
  --min-depth 1.0 \
  --max-depth 50.0 \
  --overwrite
```

The output format is `depths/<image_stem>.npy` with `float32` metric
OpenCV/COLMAP z-depth and invalid pixels encoded as `0`.

Calibration note: the finalized raw-frame route projects `/odin1/cloud_raw`
directly using `T_camera_from_lidar`. The available ROS bags do not contain
`/tf` or `/tf_static`, and `/odin1/cloud_raw` is recorded with
`frame_id=odin1_base_link`. Prepared scenes therefore use
`lidar_frame_assumption=odin1_base_link` and
`T_base_from_lidar=identity`. This identity base/lidar assumption is supported
by frame metadata, calibration files, code inspection, multi-frame overlays,
and numerical sanity checks, but it is not independently confirmed by ROS TF.
The route is frozen as `freeze except for calibration note`. Existing stale
global-cloud depth reports must not be used as evidence for the finalized
raw-frame route.
