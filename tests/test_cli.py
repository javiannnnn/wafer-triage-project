from wm811k.cli import split_cli_args


def test_split_cli_args_separates_arm_names_from_config_overrides():
    arms, overrides = split_cli_args(["gbm", "cnn", "data.map_size=32", "budgets=[0.1,1.0]"])

    assert arms == ["gbm", "cnn"]
    assert overrides == ["data.map_size=32", "budgets=[0.1,1.0]"]


def test_split_cli_args_with_no_arms_means_all_arms():
    arms, overrides = split_cli_args(["seeds=[0]"])

    assert arms == ["gbm", "cnn", "simclr", "fixmatch"]
    assert overrides == ["seeds=[0]"]
