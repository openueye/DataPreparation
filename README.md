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
conda run -n 3dgs_train python -m data_preparation slam \
  --scene Downtown1 \
  --rosbag-dir /home/haibo/Documents/Thesis/03_Datasets/001_rosbags/Downtown1 \
  --max-points 3000000 \
  -- --copy-images
```

Use `--max-points 0` to disable point-cloud downsampling.

## Training

Pass any route output scene to the trainer:

```bash
conda run -n 3dgs_train python train.py \
  --mode train \
  --data-format colmap \
  --data-dir /path/to/output_scene
```
