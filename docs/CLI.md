# data_preparation CLI

Run commands from the `00_Baselines` parent directory or with `data_preparation`
installed on `PYTHONPATH`:

```bash
conda run -n 3dgs_train python -m data_preparation <command> --help
```

The canonical entry point is:

```bash
python -m data_preparation
```

## Canonical Commands

| Command | Purpose | Pipeline role |
| --- | --- | --- |
| `sfm` | Organize an Odin1 ROS bag and run COLMAP/SfM on rectified images. | raw input to visual COLMAP scene |
| `hybrid` | Organize the ROS bag, reuse/run SfM, and export SfM poses with SLAM/LiDAR points. | hybrid COLMAP scene export |
| `slam` | Organize the ROS bag and export SLAM poses/SLAM-LiDAR points to COLMAP text. | SLAM-compatible COLMAP scene export |
| `depth-prior-project` | Project raw-frame or local-fused LiDAR metric depth priors. | sparse LiDAR depth generation |
| `depth-prior-edge-masks` | Generate edge-gated masks for completed depth priors. | post-completion reliability filtering |
| `depth-prior-apply-mask` | Apply masks by zeroing rejected completed-depth pixels. | final completed-depth artifact preparation |
| `depth-prior-sidecars` | Generate confidence and source-label sidecars. | reliability-aware sidecar preparation |
| `lsgslam-export` | Export a COLMAP-compatible scene plus depth priors to LSG-SLAM EuRoC-style RGB-D layout. | external LSG-SLAM adapter |

## Deprecated Direct Backend

`video2colmap` remains callable for compatibility and debugging, but new users
should prefer:

```bash
python -m data_preparation sfm ...
```

The `sfm` route handles ROS bag organization before invoking the video/COLMAP
backend.

## Depth-Prior Command Boundary

Depth-prior generation and artifact preparation belong here:

```text
00_Baselines/data_preparation
```

Training-time consumption belongs in `00_Baselines/02baseline` through:

```text
--depths-dir
--depth-masks-dir
--depth-confidence-dir
```

Do not add new depth-generation logic to `02baseline`.

## Common Help Checks

```bash
conda run -n 3dgs_train python -m data_preparation sfm --help
conda run -n 3dgs_train python -m data_preparation hybrid --help
conda run -n 3dgs_train python -m data_preparation slam --help
conda run -n 3dgs_train python -m data_preparation depth-prior-project --help
conda run -n 3dgs_train python -m data_preparation depth-prior-edge-masks --help
conda run -n 3dgs_train python -m data_preparation depth-prior-apply-mask --help
conda run -n 3dgs_train python -m data_preparation depth-prior-sidecars --help
conda run -n 3dgs_train python -m data_preparation lsgslam-export --help
```
