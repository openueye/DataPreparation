from data_preparation import cli


def test_depth_prior_consolidation_commands_are_registered():
    assert cli.DEPTH_PRIOR_COMMANDS["depth-prior-edge-masks"].module == "data_preparation.depth_prior.edge_masks"
    assert cli.DEPTH_PRIOR_COMMANDS["depth-prior-apply-mask"].module == "data_preparation.depth_prior.edge_masks"
    assert cli.DEPTH_PRIOR_COMMANDS["depth-prior-sidecars"].module == "data_preparation.depth_prior.sidecars"


def test_canonical_command_registry_has_no_missing_legacy_modules():
    assert "rosbag-inspect" not in cli.COMMANDS
    assert "rosbag-extract-images" not in cli.COMMANDS
    assert cli.COMMANDS["lsgslam-export"].module == "data_preparation.slam.lsgslam_adapter"
