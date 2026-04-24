# data_preparation

`data_preparation` is the standalone preprocessing toolkit for the baseline
workspace. Its responsibility stops at data inspection, parsing, conversion,
preprocessing, scene-format writing, and preprocessing quality checks.

It does not own training, rendering, viewer, checkpoint, or model logic. The
training repository consumes the processed scenes produced here.

## Directory Layout

```text
data_preparation/
  cli.py                    # unified command dispatcher: python -m data_preparation
  __main__.py               # module entrypoint
  workflows.py              # scene-centric orchestration wrappers
  shared/                   # common calibration, camera, pose, point cloud, timing, I/O helpers
  rosbag_to_colmap/         # ROS2 bag image-only extraction for pure visual COLMAP SfM
  rosbag_to_3dgs/           # ROS2 bag inspection, conversion, and raw projection validation
  slam_to_colmap/           # processed SLAM/LiDAR scene -> COLMAP text layout
  video2colmap/             # raw video -> COLMAP scene preprocessing
  data_quality/             # processed-scene projection checks and LiDAR map colorization
```

## Workflow Boundary

```text
03_Datasets/       # raw acquisition data: rosbags, videos, calibration files
data_preparation/  # inspect, parse, convert, validate, and write scene layouts
04_ProcessedData/  # training-ready or validation-ready scenes
3DGS_baseline01/     # training/rendering/viewer/export consumers
05_Outputs/        # checkpoints, reports, render outputs, validation artifacts
```

Recommended flow:

1. Start from raw inputs in `03_Datasets/`.
2. Run the workflow commands to inspect, prepare, validate, and export scenes.
3. Pass the prepared scene directory to `3DGS_baseline01` for training, viewing,
   or export.

This package does not call `3DGS_baseline01/run_scene.sh`, does not train models,
and does not write checkpoints.

## Unified Entrypoint

Use the conda environment used by the baseline project:

```bash
conda activate 3dgs_train
cd /home/haibo/Documents/Thesis/00_Baselines
export PYTHONPATH="$PWD:${PYTHONPATH}"
python -m data_preparation
```

## Workflow Entrypoint

Default users should start from the scene-centric workflow commands:

```bash
python -m data_preparation inspect --scene Ferrari1

python -m data_preparation prepare \
  --scene Ferrari1 \
  --source rosbag \
  --preset smoke \
  --dry-run

python -m data_preparation prepare \
  --scene Ferrari1 \
  --source rosbag \
  --preset smoke

python -m data_preparation prepare \
  --scene Ferrari1 \
  --source rosbag-sfm \
  --preset full

python -m data_preparation validate \
  --scene Ferrari1 \
  --preset smoke

python -m data_preparation colorize \
  --scene Ferrari1 \
  --preset smoke

python -m data_preparation export \
  --scene Ferrari1 \
  --format colmap-compatible

python -m data_preparation run \
  --scene Ferrari1 \
  --source rosbag \
  --preset full
```

The wrapper layer infers canonical paths from the Thesis layout:

```text
03_Datasets/001_rosbags/<scene>
04_ProcessedData/011_scenes_lidar/<scene>
04_ProcessedData/010_scenes_colmap/<scene>_slam_compat
04_ProcessedData/010_scenes_colmap/<scene>_SFM
05_Outputs/030_validation/<scene>/
```

Presets:

- `smoke`: lightweight conversion/validation defaults for quick checks.
- `full`: full scene preparation and validation defaults.
- `debug`: heavier diagnostics while still using the same backend tools.

For ROS bag SLAM/LiDAR preparation, `smoke` and `full` intentionally write
different scene directories:

```text
04_ProcessedData/011_scenes_lidar/<scene>_smoke  # smoke prepare output
04_ProcessedData/011_scenes_lidar/<scene>        # full canonical training scene
```

Use the `_smoke` scene only for quick validation and disposable experiments.
Use the canonical `<scene>` directory for normal training runs.

For pure visual ablation baselines, use `--source rosbag-sfm`. That workflow
extracts only the ROS bag camera stream, rectifies FishPoly images to an
undistorted `PINHOLE` camera, runs COLMAP SfM, and writes the training-ready
scene under:

```text
04_ProcessedData/010_scenes_colmap/<scene>_SFM
```

The staging extraction and rectification scenes are written under
`05_Outputs/030_validation/<scene>/rosbag_sfm/`. They are not training inputs.
This path intentionally does not use SLAM odometry or LiDAR points.

`prepare` is conservative by default:

- `--dry-run` prints inferred inputs, outputs, calibration, bag directory, output
  directory, and preset-injected backend arguments without running the backend.
- If the target output directory already exists and is non-empty, `prepare`
  refuses to run.
- Pass `--overwrite` only when you intentionally want to reuse or replace the
  existing target output directory.

`inspect --scene <scene>` is scene-scoped at the workflow layer. It scans
`03_Datasets/001_rosbags/<scene>` and writes reports under
`05_Outputs/030_validation/<scene>/inspect/`. The legacy
`rosbag-inspect` command remains available for full-root scans of
`03_Datasets/001_rosbags`.

`colorize` wraps the existing LiDAR colorization backend:

```text
smoke output:
05_Outputs/030_validation/<scene>/colorize/global_map_colorized_preview.ply

full output:
04_ProcessedData/011_scenes_lidar/<scene>/lidar/global_map_colorized.ply
```

For `smoke`, colorization reads `<scene>_smoke` when that prepared scene exists;
otherwise it falls back to the canonical `<scene>` scene and still writes the
preview artifact under validation.

Advanced backend arguments can be passed after `--`, for example:

```bash
python -m data_preparation prepare --scene Ferrari1 --source rosbag --preset smoke -- \
  --cloud-stride 4
```

## Legacy Commands

The original tool commands remain supported and keep their existing parameters:

```bash
python -m data_preparation rosbag-inspect --help
python -m data_preparation rosbag-extract-images --help
python -m data_preparation rosbag-convert --help
python -m data_preparation rosbag-validate-projection --help
python -m data_preparation slam-to-colmap --help
python -m data_preparation video2colmap --help
python -m data_preparation projection-check --help
python -m data_preparation colorize-lidar --help
```

## Output Contracts

COLMAP-ready output:

```text
04_ProcessedData/010_scenes_colmap/<scene_name>/
├── images/
└── sparse/0/
    ├── cameras.bin | cameras.txt
    ├── images.bin  | images.txt
    └── points3D.bin | points3D.txt
```

The final camera model should be undistorted `PINHOLE` or `SIMPLE_PINHOLE`.
SLAM/LiDAR scenes exported from FishPoly or other distorted camera models must
be rectified before they are used as COLMAP-compatible 3DGS training data. The
`slam-to-colmap` converter rejects distorted camera metadata by default; pass
`--allow-pinhole-approximation` only when you intentionally want the previous
K-like pinhole smoke-test export.

SLAM/LiDAR-ready output:

```text
04_ProcessedData/011_scenes_lidar/<scene_name>/
├── images/
├── intrinsics/camera.json
├── poses/poses.csv
├── lidar/global_map.ply
├── scene_meta.json
├── transforms/tf_chain.json
└── metadata/associations.csv
```

Optional prepared outputs such as `lidar/global_map_colorized.ply` are also
owned by this package. Training-time reports and checkpoints belong under
`05_Outputs/`, not inside the prepared scene.

## Tool Groups

- `rosbag_to_3dgs/`
  - inspects ROS2 `.db3` bags and topic health
  - parses ROS2 CDR messages through shared message helpers
  - converts synchronized RGB/odometry/LiDAR messages into a 3DGS-ready scene
  - validates raw bag LiDAR-to-image projection with calibration overlays

- `rosbag_to_colmap/`
  - extracts only compressed image frames from ROS2 bags
  - writes FishPoly camera metadata from `cam_in_ex.txt`
  - intentionally excludes odometry, LiDAR, SLAM poses, and LiDAR point maps

- `video2colmap/`
  - extracts frames from video or consumes a prepared image directory
  - runs COLMAP feature extraction, matching, mapping, and undistortion
  - writes a COLMAP-style `images/` and `sparse/0/` dataset

- `slam_to_colmap/`
  - converts processed SLAM/LiDAR exports into COLMAP-compatible text files
  - reuses shared pose, camera, point cloud, and COLMAP writers

- `data_quality/`
  - checks processed-scene LiDAR/RGB projection quality
  - writes overlay images and JSON reports
  - colorizes global LiDAR maps from synchronized RGB frames

- `shared/`
  - `calibration.py`: camera/LiDAR calibration parsing
  - `camera_models.py`: K-like pinhole projection helpers
  - `poses.py`: pose CSV loading and quaternion/matrix conversions
  - `pointcloud.py`: ROS2 point decoding and PLY read/write helpers
  - `timing.py`: nearest timestamp matching and sync statistics
  - `io.py`: JSON/CSV/image-path/common numeric helpers
  - `colmap_io.py`: COLMAP text model writers

## Boundary Notes

The refactor keeps existing output layouts, COLMAP text semantics, ROS topic
defaults, and projection approximations. Full FishPoly undistortion and full ROS
message coverage remain out of scope for this toolkit pass.
