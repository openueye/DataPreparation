# rosbag_to_3dgs

This directory owns ROS2 bag inspection, conversion, and raw extrinsic
projection checks. It is part of `data_preparation`; it is not invoked through
`3DGS_baseline01/run_scene.sh`.

The boundary is:

```text
03_Datasets/001_rosbags/       # raw ROS2 bags and calibration files
data_preparation/rosbag_to_3dgs/
04_ProcessedData/011_scenes_lidar/<scene_name>/  # prepared LiDAR scene
3DGS_baseline01/                 # training consumer only
```

## Entrypoint

Run from the `00_Baselines` directory with the baseline conda environment:

```bash
cd /home/haibo/Documents/Thesis/00_Baselines
PYTHONPATH=/home/haibo/Documents/Thesis/00_Baselines \
python -m data_preparation rosbag-inspect --help

PYTHONPATH=/home/haibo/Documents/Thesis/00_Baselines \
python -m data_preparation rosbag-convert --help

PYTHONPATH=/home/haibo/Documents/Thesis/00_Baselines \
python -m data_preparation rosbag-validate-projection --help
```

## Inspect Bags

```bash
PYTHONPATH=/home/haibo/Documents/Thesis/00_Baselines \
python -m data_preparation rosbag-inspect \
  --root /home/haibo/Documents/Thesis/03_Datasets/001_rosbags \
  --inventory-out /home/haibo/Documents/Thesis/05_Outputs/030_validation/rosbag_reports/bag_inventory.md \
  --feasibility-out /home/haibo/Documents/Thesis/05_Outputs/030_validation/rosbag_reports/feasibility_report.md \
  --summary-json /home/haibo/Documents/Thesis/05_Outputs/030_validation/rosbag_reports/inspection_summary.json
```

Check that the target bag has valid `.db3` files, usable image/odometry/LiDAR
topics, overlapping time ranges, and topic names matching the conversion
parameters.

## Convert One Bag

```bash
PYTHONPATH=/home/haibo/Documents/Thesis/00_Baselines \
python -m data_preparation rosbag-convert \
  --bag-dir /home/haibo/Documents/Thesis/03_Datasets/001_rosbags/Motorworld1 \
  --output-dir /home/haibo/Documents/Thesis/04_ProcessedData/011_scenes_lidar/Motorworld1 \
  --calibration /home/haibo/Documents/Thesis/03_Datasets/001_rosbags/cam_in_ex.txt \
  --export-cloud-frames
```

For a smoke conversion, add `--limit-images 200`.

Expected prepared scene:

```text
04_ProcessedData/011_scenes_lidar/<scene_name>/
├── scene_meta.json
├── images/
├── poses/poses.csv
├── intrinsics/camera.json
├── lidar/global_map.ply
├── lidar/frames/
├── transforms/tf_chain.json
└── metadata/associations.csv
```

## Validate Raw Projection

```bash
PYTHONPATH=/home/haibo/Documents/Thesis/00_Baselines \
python -m data_preparation rosbag-validate-projection \
  --bag-dir /home/haibo/Documents/Thesis/03_Datasets/001_rosbags/Motorworld1 \
  --output-dir /home/haibo/Documents/Thesis/05_Outputs/030_validation/rosbag_extrinsic_checks/Motorworld1 \
  --calibration /home/haibo/Documents/Thesis/03_Datasets/001_rosbags/cam_in_ex.txt \
  --num-samples 8
```

This check uses the exported `K_like` pinhole approximation. It is meant to
catch extrinsic direction and scale mistakes, not to report precise distortion-
aware reprojection error.

## Output Meaning

- `images/*.jpg`: RGB images exported from `sensor_msgs/msg/CompressedImage`.
- `poses/poses.csv`: per-image `T_world_from_camera` rows.
- `intrinsics/camera.json`: camera intrinsics and distortion metadata parsed
  from calibration. FishPoly captures are marked as `image_model:
  raw_distorted` and `colmap_training_ready: false`; a later rectification step
  must create undistorted pinhole images before using the COLMAP-oriented 3DGS
  baseline.
- `lidar/frames/*.npy`: optional per-frame point clouds matched to images.
- `lidar/global_map.ply`: lightweight accumulated point cloud for training
  initialization.
- `transforms/tf_chain.json`: extrinsic direction and frame assumptions.
- `metadata/associations.csv`: image, pose, and LiDAR timestamp associations.

## Handoff To Training

After conversion and quality checks, use the output directory as
`--data-dir ... --data-format slam_lidar` in `3DGS_baseline01`. Do not add rosbag
inspection, conversion, colorization, or projection commands back into the
training wrapper.
