# data_preparation Migration Notes

This note records command consolidation without changing validated data
semantics, coordinate conventions, output names, or downstream-compatible paths.

## Canonical Entry Point

Use:

```bash
python -m data_preparation <command>
```

## Command Replacements

| Old command or script | New command | Status |
| --- | --- | --- |
| direct `video2colmap/preprocess_video_to_colmap.py` use for normal ROS bag scenes | `python -m data_preparation sfm ...` | deprecated direct backend; still available as `video2colmap` for debugging |
| missing legacy `rosbag-inspect` command | no canonical replacement in this repo | removed from CLI registry because target module is absent |
| missing legacy `rosbag-extract-images` command | `sfm` route for supported ROS bag to COLMAP flow | removed from CLI registry because target module is absent |
| `05_Outputs/.../generate_edge_gated_masks.py` | `python -m data_preparation depth-prior-edge-masks ...` | migrated |
| `05_Outputs/.../apply_mask_to_depths.py` | `python -m data_preparation depth-prior-apply-mask ...` | migrated |
| `00_Baselines/02baseline/tools/generate_depth_confidence_sidecars.py` | `python -m data_preparation depth-prior-sidecars ...` | migrated out of consumer repo |
| direct `slam/lsgslam_adapter.py` invocation | `python -m data_preparation lsgslam-export ...` | now advertised canonical adapter command |

## Preserved Behavior

- Existing scene output paths are preserved.
- Existing depth `.npy`, mask `.npy`, confidence `.npy`, source-label `.npy`,
  and report JSON formats are preserved.
- Existing transform conventions are preserved.
- `02baseline` remains a consumer of prepared depth-prior artifacts only.

## Remaining External Step

SPNet full inference is still represented by:

```text
05_Outputs/depth_prior_experiments/2026-06-06_spnet_fused9_anchor_preserved_1600x1296/generate_spnet_anchor_preserved.py
```

It should be migrated only with a dedicated tested wrapper because it depends on
the Gaussian-LIC SPNet checkout, checkpoint loading, and model runtime.
