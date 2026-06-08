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

Canonical CLI reference:

```text
docs/CLI.md
```

Command migration notes:

```text
docs/migration.md
```

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

## Downtown1 depth-prior route

`data_preparation` is the canonical owner for Downtown1 depth-map generation
and depth-prior artifact traceability. The route includes raw-frame LiDAR
projection, local-fused LiDAR projection, SPNet completion, edge90 masking, and
confidence/source-label sidecars. `00_Baselines/02baseline` should only consume
prepared artifacts through `--depths-dir`, `--depth-masks-dir`, and
`--depth-confidence-dir`.

Detailed closure and traceability report:

```text
docs/2026-06-08_downtown1_depth_prior_route_closure_and_traceability.md
```

Canonical post-completion utilities:

```bash
python -m data_preparation depth-prior-edge-masks --help
python -m data_preparation depth-prior-apply-mask --help
python -m data_preparation depth-prior-sidecars --help
```

## Depth Prior Projection

Depth priors are generated from synchronized raw LiDAR frames. The command
uses the converted scene's `manifest.json` to find the prepared pure-headerstamp
scene, then reads `associations/frame_associations.csv`,
`lidar/raw_frames/*.npz`, and `calib/tf_chain.json`.

```bash
python -m data_preparation depth-prior-project \
  --scene-dir /home/haibo/Documents/Thesis/04_ProcessedData/slam/downtown1_3M \
  --method raw-frame \
  --min-depth 1.0 \
  --max-depth 50.0 \
  --overwrite
```

The output format is `depths/<image_stem>.npy` with `float32` metric
OpenCV/COLMAP z-depth and invalid pixels encoded as `0`.

`raw-frame` is the default method and is kept as the frozen single raw LiDAR
frame baseline. It writes only to:

```text
<scene>/depths/
<scene>/depth_prior_report.json
```

### Local Multi-Frame LiDAR Fusion

Use `local-fused` for a pose-compensated local raw LiDAR fusion prior. This
does not overwrite the raw-frame baseline. The default fusion window is 5
centered frames; the window must be a positive odd integer.

```bash
python -m data_preparation depth-prior-project \
  --scene-dir /home/haibo/Documents/Thesis/04_ProcessedData/slam/downtown1_3M \
  --prepared-scene-dir /home/haibo/Documents/Thesis/04_ProcessedData/rosbag_prepared/Downtown1_pure_headerstamp \
  --method local-fused \
  --fusion-window 5 \
  --min-depth 1.0 \
  --max-depth 50.0 \
  --overwrite
```

Default local-fused outputs:

```text
<scene>/depths_lidar_fused_5f/<image_stem>.npy
<scene>/masks_lidar_fused_5f/<image_stem>.npy
<scene>/depth_prior_report_local_fused_5f.json
```

Pass `--write-confidence` to also write:

```text
<scene>/confidence_lidar_fused_5f/<image_stem>.npy
```

The local-fused transform is:

```text
p_target_cam =
  inverse(T_odom_from_target_cam)
  @ T_odom_from_source_cam
  @ T_camera_from_lidar
  @ p_source_lidar
```

The route reads `poses/camera_poses.csv`, `calib/tf_chain.json`,
`calib/camera_rectified.json`, `associations/frame_associations.csv`, and
`lidar/raw_frames/*.npz`. It writes a quality report with source frame ids,
source time offsets, input/front/projected point counts, valid pixels, valid
ratio, depth statistics, z-buffer collisions, conflict/outlier metrics, and
the fusion time window. It does not generate overlays by default; pass
`--overlay-count N` for a small number of visual checks under the scene
directory.

For 02baseline training, keep the depth loss format unchanged and switch depth
priors only through `--use-depth-prior --depths-dir`:

```bash
python train.py \
  --data-dir /home/haibo/Documents/Thesis/04_ProcessedData/slam/downtown1_3M \
  --use-depth-prior \
  --depths-dir /home/haibo/Documents/Thesis/04_ProcessedData/slam/downtown1_3M/depths_lidar_fused_5f \
  --depth-prior-method local-fused \
  --depth-report-path /home/haibo/Documents/Thesis/04_ProcessedData/slam/downtown1_3M/depth_prior_report_local_fused_5f.json \
  --iterations 100 \
  --save-interval 100 \
  --val-interval 0
```

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
