import argparse
import json

import numpy as np

from data_preparation.depth_prior.sidecars import generate_sidecars


def test_generate_sidecars_labels_anchors_completed_and_rejected_pixels(tmp_path):
    final_dir = tmp_path / "final"
    completed_dir = tmp_path / "completed"
    anchor_dir = tmp_path / "anchors"
    confidence_dir = tmp_path / "confidence"
    labels_dir = tmp_path / "labels"
    for path in (final_dir, completed_dir, anchor_dir):
        path.mkdir(parents=True)

    final = np.asarray([[2.0, 3.0], [0.0, 0.0]], dtype=np.float32)
    completed = np.asarray([[2.0, 3.0], [4.0, 0.0]], dtype=np.float32)
    anchor = np.asarray([[2.0, 0.0], [0.0, 0.0]], dtype=np.float32)
    np.save(final_dir / "000001.npy", final)
    np.save(completed_dir / "000001.npy", completed)
    np.save(anchor_dir / "000001.npy", anchor)

    report = generate_sidecars(
        argparse.Namespace(
            final_depths_dir=final_dir,
            completed_depths_dir=completed_dir,
            anchor_depths_dir=anchor_dir,
            confidence_dir=confidence_dir,
            source_labels_dir=labels_dir,
            report_path=tmp_path / "sidecar_report.json",
            method="test-confidence",
            anchor_weight=1.0,
            completion_weight=0.2,
        )
    )

    confidence = np.load(confidence_dir / "000001.npy")
    labels = np.load(labels_dir / "000001.npy")
    written_report = json.loads((tmp_path / "sidecar_report.json").read_text(encoding="utf-8"))

    assert np.allclose(confidence, np.asarray([[1.0, 0.2], [0.0, 0.0]], dtype=np.float32))
    assert labels.tolist() == [[1, 2], [4, 0]]
    assert report["label_contract"]["4"] == "completed_rejected_by_final_mask"
    assert written_report["num_images"] == 1
    assert written_report["mean_positive_weight_summary"]["mean"] == 0.6000000238418579
